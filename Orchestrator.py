# Orchestrator.py

import os
import operator
from typing import Annotated, TypedDict, Literal, List
import sqlite3
import datetime
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from llm import FAST_LLM
# Import ReAct Agent (reasoning_llm)
from React_agent import run_react_agent
# Import Fast Agent (fast_llm) – NextGen_Chat.py
from Simple_agent import app as fast_agent_app
from Simple_agent import DB_PATH


# load_dotenv()
# # GOOGLE_API_KEY is picked up from env by ChatGoogleGenerativeAI
# llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm_router = FAST_LLM

# =============================================================================
# 2. STATE & ANALYSIS MODELS
# =============================================================================

class OrchestratorState(TypedDict):
    """
    Shared state passed through the Orchestrator graph.
    """
    messages: Annotated[List[BaseMessage], operator.add]
    analysis: dict
    final_response: str
    customer_id: str
    long_term_memory: str


class QueryDeepAnalysis(BaseModel):
    """
    Advanced 7-Factor Analysis for Routing.
    """
    # 1. Complexity
    complexity_score: int = Field(
        description="1-10 Score. 1=Hi/Bye, 10=Multi-step deductive reasoning needed."
    )

    # 2. Sentiment
    sentiment_category: Literal[
        "Joy", "Neutral", "Confusion", "Frustration", "Anger", "Rage", "Anxiety", "Urgency"
    ] = Field(description="Primary emotion category.")
    sentiment_intensity: int = Field(
        description="1-10 Intensity of emotion."
    )

    # 3. Intent
    intent: Literal[
        "General_Info", "Billing_Status", "Technical_Issue", "Complaint", "Sales"
    ] = Field(description="High-level intent of the query.")

    # 4. Knowledge Source
    requires_multi_source: bool = Field(
        description="True if query needs comparison between DB and SOP."
    )

    # 5. Urgency
    urgency_level: Literal["Low", "Medium", "High", "Critical"] = Field(
        description="How urgent the issue appears."
    )

    # 7. Reasoning
    requires_react: bool = Field(
        description=(
            "True if the answer isn't a simple lookup but requires logic "
            "(e.g., 'Why is my bill high?')."
        )
    )


# =============================================================================
# 3. SENTIMENT MAPPER (Orchestrator → Fast LLM)
# =============================================================================

def sentiment_mapper(category: str) -> str:
    """
    Map the Orchestrator's sentiment_category to NextGen_Chat's SENTIMENT_OPTIONS.

    Orchestrator sentiment_category:
        "Joy", "Neutral", "Confusion", "Frustration", "Anger", "Rage", "Anxiety", "Urgency"

    NextGen_Chat SENTIMENT_OPTIONS:
        "Angry", "Happy", "Sad", "Neutral", "Frustrated"
    """
    if not category:
        return "Neutral"

    category = category.strip()

    if category == "Joy":
        return "Happy"

    if category in ["Frustration"]:
        return "Frustrated"

    if category in ["Anger", "Rage"]:
        return "Angry"

    if category in ["Anxiety", "Confusion", "Worried", "Confused"]:
        # “Worried / confused” – we bias toward a softer, caring tone
        return "Sad"

    # "Urgency" or anything unknown → default to Neutral tone.
    return "Neutral"


# =============================================================================
# 4. NODES
# =============================================================================

def brain_classifier_node(state: OrchestratorState):
    """
    The BRAIN: Analyzes the query using the 7 decision factors.
    """
    print("\n🧠 ORCHESTRATOR: Analyzing Query Dimensions...")
    print("\n--- [Node] Brain Node ---")
    query = state["messages"][-1].content
    # print(query)

    system_prompt = """
        You are the Decision Core of an AI Helpdesk. 
        Analyze the user query deeply based on:
        1. Complexity (Is it a simple lookup or a 'Why' question?)
        2. Sentiment (Detect subtle cues of frustration or rage).
        3. Intent.
        4. Need for multi-source reasoning (DB + SOP).
        5. Urgency.
        6. Whether it requires ReAct-style reasoning.
        
        OUTPUT RULES:
        - If user asks 'Why', 'Explain', 'Compare', or has a complex technical issue -> requires_react = True.
        - If user asks 'What is', 'How to', 'Status of' -> requires_react = False.
        - If Sentiment is 'Rage' or Urgency is 'Critical' -> use high intensity & urgency_level accordingly.
    """

    structured_llm = llm_router.with_structured_output(QueryDeepAnalysis)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{query}"),
        ]
    )

    analysis_result = prompt | structured_llm
    analysis = analysis_result.invoke({"query": query})

    print(
        f"   📊 Analysis: "
        f"Complexity={analysis.complexity_score}, "
        f"Sentiment={analysis.sentiment_category}, "
        f"ReAct={analysis.requires_react}, "
        f"Urgency={analysis.urgency_level}"
    )

    # Store as raw dict in state
    return {"analysis": analysis.model_dump()}

def memory_manager(state: OrchestratorState):
    """
    1. Loads existing summary from DB.
    2. Checks if conversation is too long (> 5 messages).
    3. If long: Summarizes older messages -> Updates DB -> Updates State.
    """
    print("\n--- [Node] Memory Manager ---")
    cust_id = state.get("customer_id")
    messages = state["messages"]
    
    # 1. Fetch Existing Summary
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM conversation_memory WHERE customer_id = ?", (cust_id,))
    row = cursor.fetchone()
    existing_summary = row[0] if row else "No previous context."
    conn.close()
    
    # 2. Check Length Logic (Threshold = 5 messages)
    if len(messages) > 5:
        print("   ⚠️ Long Context Detected. Summarizing...")
        
        # We summarize everything EXCEPT the last 2 messages (Keep recent context fresh)
        to_summarize = messages[:-2]
        recent_messages = messages[-2:]
        
        # LLM Summarization
        summary_prompt = f"""
        Summarize the following conversation concisely. Preserve key details like Customer Name, Issues Reported, and Solutions offered.
        
        Previous Summary: {existing_summary}
        
        New Messages:
        {to_summarize}
        """
        new_summary = llm_router.invoke(summary_prompt).content
        
        # 3. Update DB
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO conversation_memory (customer_id, summary, last_updated) VALUES (?, ?, ?)", 
                       (cust_id, new_summary, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()
        
        print(f"   ✅ Memory Updated: {new_summary[:50]}...")
        return {"long_term_memory": new_summary}
    
    return {"long_term_memory": existing_summary}

def fast_agent_node(state: OrchestratorState):
    """
    Fast Path: Delegates to the Fast LLM (NextGen_Chat.py) for SOP/DB-style queries.

    - Uses Orchestrator's analysis to map sentiment into the
      SENTIMENT_OPTIONS expected by NextGen_Chat.
    - NextGen_Chat is responsible for:
        * Routing between simple DB / SOP tools
        * Deciding if escalation is needed
        * Empathetic synthesis using the passed-in sentiment
    """
    print("\n⚡ FAST AGENT: Delegating to Fast LLM (NextGen_Chat)...")

    analysis = state["analysis"]
    raw_sentiment = analysis["sentiment_category"]  # e.g. "Frustration"
    mapped_sentiment = sentiment_mapper(raw_sentiment)  # e.g. "Frustrated"

    fast_state = {
        "messages": state["messages"], # Pass the list of messages, not just the content
        "customer_id": state.get("customer_id"),
        "sentiment": mapped_sentiment,
        "memory": state.get("long_term_memory")
    }

    response_state = fast_agent_app.invoke(fast_state)
    # print(f"FastLLM: {response_state}")
    # Fast LLM returns a state with 'messages' list; last one is the final AIMessage
    final_msg = response_state["messages"][-1].content
    # print(final_msg)
    return {"final_response": final_msg}


def react_agent_node(state: OrchestratorState):
    """
    Slow Path: Routes to the ReAct Agent for deep reasoning and tool use.
    ReAct agent is responsible for:
      - DB investigation,
      - deciding when it is out of information,
      - politely escalating via ticket API (create_ticket_tool) if needed.
    """
    print("\n🐢 REACT AGENT: Delegating to Reasoner...")

    query = state["messages"][-1].content
    analysis = state["analysis"]
    customer_id = state.get("customer_id") # Default if not present
    memory = state.get("long_term_memory", "No History")

    # Call external ReAct agent – it returns a final string message
    # run_react_agent is a regular function, not a compiled LangGraph app, so we call it directly.
    final_msg, result_state = run_react_agent(
        user_query=query,
        sentiment=analysis["sentiment_category"],
        customer_id=customer_id,
        chat_history=memory
    )

    # print(f"ReactAgent: {result_state}")

    return {"final_response": final_msg}


# =============================================================================
# 5. ROUTING LOGIC
# =============================================================================

def router(state: OrchestratorState) -> str:
    """
    Decide whether to use the Fast Agent or the ReAct Agent.

    No ticket creation here -- pure routing:
      - ReAct:
          * complex reasoning
          * or negative/emotional
          * or high/critical urgency
      - Fast:
          * simple / known-path queries
    """
    analysis = state["analysis"]

    complexity = analysis["complexity_score"]
    requires_react = analysis["requires_react"]
    sentiment = analysis["sentiment_category"]
    urgency = analysis["urgency_level"]

    # Use ReAct if:
    # - it explicitly needs reasoning, OR
    # - complexity is high, OR
    # - user is clearly upset, OR
    # - urgency is high / critical.
    if (
        requires_react
        or complexity >= 7
        or sentiment in ["Frustration", "Anger", "Rage"]
        or urgency in ["High", "Critical"]
    ):
        return "react_agent"

    # Otherwise, Fast path is enough.
    return "fast_agent"


# =============================================================================
# 6. GRAPH CONSTRUCTION
# =============================================================================

workflow = StateGraph(OrchestratorState)

workflow.add_node("brain", brain_classifier_node)
workflow.add_node("memory", memory_manager)
workflow.add_node("fast_agent", fast_agent_node)
workflow.add_node("react_agent", react_agent_node)

workflow.set_entry_point("brain")
workflow.add_edge("brain", "memory")

workflow.add_conditional_edges(
    "memory",
    router,
    {
        "fast_agent": "fast_agent",
        "react_agent": "react_agent",
    },
)

workflow.add_edge("fast_agent", END)
workflow.add_edge("react_agent", END)

orchestrator_app = workflow.compile()


# =============================================================================
# 7. RUNNER (Manual CLI Test)
# =============================================================================

# if __name__ == "__main__":
    # print(">>> 🧠 ORCHESTRATOR ONLINE")

    # while True:
    #     user_input = input("Enter your query (or 'exit'): ")
    #     if user_input.lower() == "exit":
    #         break

    #     result = orchestrator_app.invoke(
    #         {
    #             "messages": [HumanMessage(content=user_input)],
    #             "customer_id": "2",
    #         }
    #     )
    #     print(f"\n🤖 Bot Response: {result['final_response']}\n")


def run_interactive_chat():
    print(">>> 🧠 ORCHESTRATOR ONLINE")
    chat_history = []

    while True:
        try:
            user_input = input("Enter your query (or 'exit'): ")
            if user_input.lower() == "exit":
                break
            chat_history.append(HumanMessage(content=user_input))

            result = orchestrator_app.invoke(
                {
                    "messages": chat_history,
                    "customer_id": "2",
                }
            )
            bot_response = result['final_response']
            chat_history.append(AIMessage(content=bot_response))

            print(f"\n🤖 Bot Response: {bot_response}\n")

            if len(chat_history) > 6:
                chat_history = chat_history[-2:]
            
            # print(chat_history)
        except Exception as e:
            print(f"exception: {e}")
    
if __name__ == "__main__":
    run_interactive_chat()