import os
import operator
import json
from typing import Annotated, TypedDict, Literal, List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_community.utilities import SQLDatabase

# Import ReAct Agent
from React_agent import run_react_agent
# Import Fast Agent 
from Simple_agent import app as fast_agent_app
# Import Ticket Tool for direct emergency escalation
from ticket_service import create_ticket_tool

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# =============================================================================
# 1. SETUP RESOURCES (Adopting Simple_agent.py Logic)
# =============================================================================

DB_PATH = "./Database/NextGen.db"
CHROMA_DB_DIR = "./sop_embeddings/chroma_store"

# Re-establishing connections for the Fast Agent (Orchestrator local use)
if not os.path.exists(DB_PATH):
    # Fallback if DB doesn't exist (Simple_agent.py usually creates this)
    print("⚠️ Warning: NextGen.db not found. Ensure Simple_agent.py has run once.")

db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)
# Connect to existing persistence if available, else standard fallback
if os.path.exists(CHROMA_DB_DIR):
    vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    sop_retriever = vectorstore.as_retriever()
else:
    print("⚠️ Warning: Chroma DB not found. Fast Agent RAG will be limited.")
    # Mock retriever for safety if file missing
    class MockRetriever:
        def invoke(self, x): return []
    sop_retriever = MockRetriever()

llm_router = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# =============================================================================
# 2. STATE & ANALYSIS MODELS
# =============================================================================

class OrchestratorState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    analysis: dict
    final_response: str
    customer_id: str

class QueryDeepAnalysis(BaseModel):
    """
    Advanced 7-Factor Analysis for Routing.
    """
    # 1. Complexity
    complexity_score: int = Field(description="1-10 Score. 1=Hi/Bye, 10=Multi-step deductive reasoning needed.")
    # 2. Sentiment
    sentiment_category: Literal[
        "Joy", "Neutral", "Confusion", "Frustration", "Anger", "Rage", "Anxiety", "Urgency"
    ] = Field(description="Primary emotion category.")
    sentiment_intensity: int = Field(description="1-10 Intensity of emotion.")
    # 3. Intent
    intent: Literal["General_Info", "Billing_Status", "Technical_Issue", "Complaint", "Sales"]
    # 4. Knowledge Source
    requires_multi_source: bool = Field(description="True if query needs comparison between DB and SOP.")
    # 5. Urgency
    urgency_level: Literal["Low", "Medium", "High", "Critical"]
    # 7. Reasoning
    requires_react: bool = Field(description="True if the answer isn't a simple lookup but requires logic (e.g., 'Why is my bill high?').")

# =============================================================================
# 3. NODES
# =============================================================================

def brain_classifier_node(state: OrchestratorState):
    """
    The BRAIN: Analyzes the query using the 7 decision factors.
    """
    print("\n🧠 ORCHESTRATOR: Analyzing Query Dimensions...")
    query = state["messages"][-1].content
    
    system_prompt = """You are the Decision Core of an AI Helpdesk. 
    Analyze the user query deeply based on:
    1. Complexity (Is it a simple lookup or a 'Why' question?)
    2. Sentiment (Detect subtle cues of frustration or rage).
    3. Intent.
    
    OUTPUT RULES:
    - If user asks 'Why', 'Explain', 'Compare', or has a complex technical issue -> requires_react = True.
    - If user asks 'What is', 'How to', 'Status of' -> requires_react = False.
    - If Sentiment is 'Rage' or Urgency is 'Critical' -> Mark accordingly for escalation.
    """
    
    structured_llm = llm_router.with_structured_output(QueryDeepAnalysis)
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")])
    
    analysis_result = prompt | structured_llm
    analysis = analysis_result.invoke({"query": query})
    
    print(f"   📊 Analysis: Complexity={analysis.complexity_score}, Sentiment={analysis.sentiment_category}, ReAct={analysis.requires_react}")
    return {"analysis": analysis.dict()}

def fast_agent_node(state: OrchestratorState):
    """
    Fast Path: Uses RAG + Simple DB Lookup (Logic adapted from Simple_agent.py).
    """
    # print("\n⚡ FAST AGENT: Executing Quick Lookup...")
    # query = state["messages"][-1].content
    # customer_id = state.get("customer_id", "CUST_001") # Default for demo
    
    # # 1. RAG Lookup
    # docs = sop_retriever.invoke(query)
    # sop_context = "\n".join([d.page_content for d in docs]) if docs else "No SOP found."
    
    # # 2. Simple DB Lookup (LLM generates SQL for simple retrieval)
    # db_context = "No DB Access needed for this query."
    # if state["analysis"]["intent"] in ["Billing_Status", "Technical_Issue"]:
    #     try:
    #         # Quick "Text-to-SQL" for simple retrieval
    #         sql_prompt = f"Write a SQLite SELECT query for customer_id='{customer_id}' based on: {query}. Schema: {db.get_table_info()}"
    #         sql_query = llm_router.invoke(sql_prompt).content.replace("```sql", "").replace("```", "").strip()
    #         if "SELECT" in sql_query.upper():
    #             db_results = db.run(sql_query)
    #             db_context = f"DB Data: {db_results}"
    #     except:
    #         db_context = "DB Lookup Failed."

    # # 3. Synthesis
    # synth_prompt = f"""You are a helpful assistant.
    # User Query: {query}
    # SOP Context: {sop_context}
    # Account Data: {db_context}
    
    # Answer the user politely and concisely in an Empathetic way.
    # """
    print("\n⚡ FAST AGENT: Delegating to Simple Agent...")
    response = fast_agent_app.invoke({
        "messages": state["messages"],
        "customer_id": state.get("customer_id", "CUST_001")
        })
    # The final message is an AIMessage object in the 'messages' list
    return {"final_response": response['messages'][-1].content}


def react_agent_node(state: OrchestratorState):
    """
    Slow Path: Routes to the ReAct Agent for deep thinking.
    """
    print("\n🐢 REACT AGENT: Delegating to Reasoner...")
    query = state["messages"][-1].content
    
    # Call the external ReAct agent
    response = run_react_agent(query)
    
    return {"final_response": response}

def emergency_ticket_node(state: OrchestratorState):
    """
    Critical Path: Bypasses AI resolution for Rage/Critical issues.
    """
    print("\n🚨 CRITICAL ESCALATION: Creating Ticket Immediately...")
    query = state["messages"][-1].content
    sentiment = state["analysis"]["sentiment_category"]
    
    ticket_msg = create_ticket_tool(
        description=f"AUTO-ESCALATION ({sentiment}): {query}", 
        priority="Urgent"
    )
    
    response = f"I sense that you are extremely upset ({sentiment}). I have bypassed our standard AI and created an urgent ticket for a human supervisor.\n{ticket_msg}"
    return {"final_response": response}

# =============================================================================
# 4. ROUTING LOGIC
# =============================================================================

def router(state: OrchestratorState):
    analysis = state["analysis"]
    
    # 1. Critical Safety Valve
    if analysis["sentiment_category"] == "Rage" or analysis["urgency_level"] == "Critical":
        return "emergency_ticket"
    
    # 2. Complexity Check
    if analysis["requires_react"] or analysis["complexity_score"] >= 7:
        return "react_agent"
    
    # 3. Default Fast Path
    return "fast_agent"

# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

workflow = StateGraph(OrchestratorState)

workflow.add_node("brain", brain_classifier_node)
workflow.add_node("fast_agent", fast_agent_node)
workflow.add_node("react_agent", react_agent_node)
workflow.add_node("emergency_ticket", emergency_ticket_node)

workflow.set_entry_point("brain")

workflow.add_conditional_edges(
    "brain",
    router,
    {
        "fast_agent": "fast_agent",
        "react_agent": "react_agent",
        "emergency_ticket": "emergency_ticket"
    }
)

workflow.add_edge("fast_agent", END)
workflow.add_edge("react_agent", END)
workflow.add_edge("emergency_ticket", END)

orchestrator_app = workflow.compile()

# =============================================================================
# 6. RUNNER
# =============================================================================

if __name__ == "__main__":
    print(">>> 🧠 ORCHESTRATOR ONLINE")

    print("\n--- ORCHESTRATOR TEST RUN ---")
    # print(f"SQL:{result.query}")

    while True:
        user_input = input("Enter your query: ")
        if user_input.lower() == 'exit':
            break
    
        result = orchestrator_app.invoke({
        "messages": [HumanMessage(content=user_input)],
        "customer_id": "CUST_002"})
        print(f"\n🤖 Bot Response: {result['final_response']}\n")
    
    # # Test 1: Simple (Fast Agent)
    # print("\n--- TEST 1: Simple Info ---")
    # res1 = orchestrator_app.invoke({"messages": [HumanMessage(content="What are the customer service hours?")], "customer_id": "CUST_001"})
    # print(f"Bot: {res1['final_response']}")
    
    # # Test 2: Complex (ReAct Agent)
    # print("\n--- TEST 2: Complex Reasoning ---")
    # res2 = orchestrator_app.invoke({"messages": [HumanMessage(content="Why is my bill higher than last month? I didn't change my plan.")], "customer_id": "CUST_002"})
    # print(f"Bot: {res2['final_response']}")
    
    # # Test 3: Rage (Emergency Ticket)
    # print("\n--- TEST 3: Rage Escalation ---")
    # res3 = orchestrator_app.invoke({"messages": [HumanMessage(content="This service is garbage! I have been waiting for 5 hours! I want to sue you!")], "customer_id": "CUST_001"})
    # print(f"Bot: {res3['final_response']}")