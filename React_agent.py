# React_agent.py

import os
import sqlite3
import json
import datetime
from typing import List, Literal, TypedDict, Annotated

from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

from llm import REASONING_LLM  # REASONING_LLM is our reasoning LLM

DB_PATH = "./Database/NextGen1.db"  # Same DB as fast_llm

# ---------------------------------------------------------------------------
# 1. SENTIMENT + STATE
# ---------------------------------------------------------------------------

# Align with NextGen_Chat SENTIMENT_OPTIONS
SENTIMENT_OPTIONS = Literal["Angry", "Happy", "Sad", "Neutral", "Frustrated"]


class ReactState(TypedDict):
    messages: Annotated[List[BaseMessage], list.__add__]
    sentiment: SENTIMENT_OPTIONS
    customer_id: str
    tool_output: str
    escalation_needed: bool
    long_term_memory: str


# Module-level context so tools can see current customer/sentiment
CURRENT_CUSTOMER_ID: str | None = None
CURRENT_SENTIMENT: str | None = None


# ---------------------------------------------------------------------------
# 2. INTELLIGENT TICKET LOGIC (ported from NextGen_Chat.py)
# ---------------------------------------------------------------------------

class TicketClassification(BaseModel):
    issue_type_id: int = Field(
        description="The ID of the most relevant issue type from the provided list."
    )
    reasoning: str = Field(description="Why this issue type was selected.")


def run_ticket_logic(description: str) -> str:
    """
    Implements the same logic as ticket_agent in NextGen_Chat.py:
      1. If an OPEN ticket already exists for this customer, update it.
      2. Otherwise, classify issue_type via LLM, pick best agent, create ticket.
    Returns a string containing 'TICKET_CREATED' or 'TICKET_UPDATED' for the synthesizer.
    """
    global CURRENT_CUSTOMER_ID, CURRENT_SENTIMENT

    cust_id = CURRENT_CUSTOMER_ID
    sentiment = CURRENT_SENTIMENT or "Neutral"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        # LOGIC 1: Check for existing OPEN ticket
        cursor.execute(
            """
            SELECT ticket_id, description, assigned_agent_id
            FROM tickets
            WHERE customer_id = ? AND status = 'Open'
            """,
            (cust_id,),
        )
        existing_ticket = cursor.fetchone()

        if existing_ticket:
            t_id = existing_ticket["ticket_id"]
            old_desc = existing_ticket["description"]
            agent_id = existing_ticket["assigned_agent_id"]

            cursor.execute(
                "SELECT name FROM agents WHERE agent_id = ?", (agent_id,)
            )
            agent_row = cursor.fetchone()
            agent_name = agent_row["name"] if agent_row else "Unknown"

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            new_desc = f"{old_desc} || [Update {timestamp}]: {description}"

            cursor.execute(
                "UPDATE tickets SET description = ? WHERE ticket_id = ?",
                (new_desc, t_id),
            )
            conn.commit()

            msg = (
                f"TICKET_UPDATED: You already have an open ticket #{t_id} with "
                f"{agent_name}. I have added this new information to it."
            )
            print(f"   🔄 {msg}")
            return msg

        # LOGIC 2: No open ticket → create new one

        # 1. Fetch Issue Types
        cursor.execute("SELECT issue_type_id, name, description FROM issue_types")
        issue_types = [dict(row) for row in cursor.fetchall()]

        # Fallback "General" type
        general_type_id = next(
            (it["issue_type_id"] for it in issue_types if "General" in it["name"]),
            1,
        )

        selected_type_id = general_type_id
        selected_type_name = "General Support"

        # 2. LLM Classification
        try:
            structured_llm = REASONING_LLM.with_structured_output(
                TicketClassification
            )
            classification_prompt = f"""
            You are a Ticket Classifier. Match the user complaint to the best Issue Type.

            Available Issue Types:
            {json.dumps(issue_types, indent=2)}

            User Complaint: "{description}"

            Return the exact issue_type_id.
            """
            classification: TicketClassification = structured_llm.invoke(
                classification_prompt
            )

            if classification and classification.issue_type_id:
                selected_type_id = classification.issue_type_id
                selected_type_name = next(
                    (
                        it["name"]
                        for it in issue_types
                        if it["issue_type_id"] == selected_type_id
                    ),
                    "General Support",
                )
        except Exception as llm_e:
            print(f"   ⚠️ Classification Warning: {llm_e}. Using General fallback.")

        print(f"   Using Issue Type: {selected_type_name} (ID: {selected_type_id})")

        # 3. Map Department (same mapping as NextGen_Chat)
        department_mapping = {
            "Billing Dispute": "Billing",
            "Wrong Recharge": "Billing",
            "Plan Benefits Not Added": "Billing",
            "Poor 4G/5G Signal": "Network",
            "Roaming Issue": "Network",
            "Slow Internet": "Broadband",
            "Router Configuration Issue": "Broadband",
            "Broadband Outage": "Broadband",
            "SIM Issue": "General Support",
            "Porting Issue": "General Support",
            "Data Exhausted Fast": "General Support",
        }
        target_department = department_mapping.get(
            selected_type_name, "General Support"
        )

        # 4. Find Best Agent
        cursor.execute(
            """
            SELECT agent_id, name
            FROM agents
            WHERE department = ?
            ORDER BY rating DESC
            LIMIT 1
            """,
            (target_department,),
        )
        best_agent = cursor.fetchone()

        if best_agent:
            assigned_agent_id = best_agent["agent_id"]
            agent_name = best_agent["name"]
        else:
            cursor.execute(
                "SELECT agent_id, name FROM agents ORDER BY rating DESC LIMIT 1"
            )
            fallback = cursor.fetchone()
            assigned_agent_id = fallback["agent_id"]
            agent_name = fallback["name"]

        # 5. Insert Ticket
        priority = "High" if sentiment in ["Angry", "Frustrated"] else "Medium"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute(
            """
            INSERT INTO tickets (
                customer_id, issue_type_id, description, status,
                priority, created_at, assigned_agent_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cust_id,
                selected_type_id,
                description,
                "Open",
                priority,
                created_at,
                assigned_agent_id,
            ),
        )

        ticket_id = cursor.lastrowid
        conn.commit()

        msg = (
            f"TICKET_CREATED: Ticket #{ticket_id} created for "
            f"'{selected_type_name}'. Assigned to {agent_name} ({target_department})."
        )
        print(f"   ✅ {msg}")
        return msg

    except Exception as e:
        print(f"   ❌ Ticket Logic Failed: {e}")
        return "Failed to manage ticket due to system error."
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 3. TOOLS (ReAct) – escalate_issue now uses run_ticket_logic
# ---------------------------------------------------------------------------

@tool
def query_telecom_db(query: str):
    """
    Executes a read-only SQL query against the Telecom Database (NextGen1.db).
    Use this to look up customer plans, outages, invoices, and transactions.
    Tables: customers, outage_areas, customer_usage, invoices, transactions, subscriptions.
    Input must be a valid SQLite SELECT query.
    """
    try:
        if any(kw in query.lower() for kw in ["drop", "delete", "update", "insert"]):
            return "Error: Read-only access allowed."

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(query)
        columns = [description[0] for description in cursor.description]
        results = cursor.fetchall()
        conn.close()

        if not results:
            return "No data found."

        return f"Schema: {columns}\nData: {results}"
    except Exception as e:
        return f"Database Error: {e}"


@tool
def escalate_issue(issue_summary: str, priority: str):
    """
    Escalates the issue to a human agent by creating or updating a support ticket.

    USE THIS WHEN:
    - You have used `query_telecom_db` and still cannot confidently resolve the issue,
      OR the data is missing/inconsistent,
      OR the user's situation clearly needs human follow-up.

    Internally this uses the same DB ticket logic as the fast_llm (NextGen_Chat):
    - Reuses any existing OPEN ticket for this customer.
    - Otherwise classifies and creates a new ticket, assigning the best agent.
    Returns a string containing TICKET_CREATED or TICKET_UPDATED.
    """
    # priority is currently not used directly; we derive from sentiment in run_ticket_logic
    msg = run_ticket_logic(issue_summary)
    return msg


# ---------------------------------------------------------------------------
# 4. BUILD THE REACT AGENT (LangChain agent, reused inside LangGraph)
# ---------------------------------------------------------------------------

def build_react_agent():
    llm = REASONING_LLM
    tools = [query_telecom_db, escalate_issue]

    # Simple ReAct-style prompt
    FULL_REACT_PROMPT = """
        You are a Senior Level 3 Telecom Support Agent.

        ==================== ROLE & BEHAVIOR RULES ====================

        1. You must perform deep technical troubleshooting and root-cause analysis.
        2. You must use tools to investigate issues:
           - Use `query_telecom_db` to check invoices, network outages, usage, plans, etc.
           - Use `escalate_issue` only when necessary.

        3. When to escalate:
           - If database information is missing, unclear, contradictory, or out of date.
           - If you still cannot confidently resolve the issue after investigation.
           - If the user is very upset or the problem requires urgent human intervention.

        4. When escalating:
           - Begin by politely apologizing.
           - Explain that the issue needs human attention.
           - Call `escalate_issue` with a short summary.
           - In FINAL ANSWER: clearly include the Ticket ID and next steps.

        5. NEVER ask the user for their customer_id or phone number.
           The system already provides customer_id in the backend context.

        6. Tone rules:
           - Angry/Frustrated users → Start with empathy and apology.
           - Confused users → Provide clarity and reassurance.
           - Neutral users → Be direct and professional.
           - Happy users → Maintain positive tone.

        7. Out-of-scope questions:
           - If question is unrelated to telecom, politely decline.

        ==================== REACT REASONING FORMAT ====================

        You MUST follow the ReAct format EXACTLY:

        Question: the user's question  
        Thought: your internal reasoning about what to do next  
        Action: the action to take, one of [{tool_names}]  
        Action Input: the input to the action  
        Observation: the result returned by the action  

        (Repeat Thought → Action → Action Input → Observation as needed.)

        When you have enough information:

        Thought: I now know the final answer  
        Final Answer: your final user-facing answer, empathetic and clear  

        ==================== BEGIN ====================

        Chat History: {history}
        Question: {input}
        Thought:{agent_scratchpad}
"""


    agent = create_agent(
        llm,
        tools,
        system_prompt=FULL_REACT_PROMPT,
        # agent_type="react-docstore",
    )
    return agent


react_agent = build_react_agent()  # shared instance


# ---------------------------------------------------------------------------
# 5. NODES
# ---------------------------------------------------------------------------

def react_reasoner_node(state: ReactState):
    """
    Node 1: Run the ReAct agent with tools and get a 'raw' answer.
    This answer may already contain TICKET_CREATED/TICKET_UPDATED from escalate_issue.
    """
    print("\n🧠 [ReAct Node] Deep reasoning and tool use...")
    user_text = state["messages"][-1].content
    memory = state.get("long_term_memory", "No History")
    result = react_agent.invoke(
        {
            "messages": [{"role": "user", "content": user_text}],
            "history": memory
        }
    )
    raw_answer = result["messages"][-1].content
    return {"tool_output": raw_answer, "escalation_needed": False}


def response_synthesizer(state: ReactState):
    """
    Final empathetic response generator (ported from NextGen_Chat.py):
    - Uses sentiment from Orchestrator
    - Uses tool_output (which may include TICKET_CREATED/TICKET_UPDATED)
    - Explains ticket creation or update clearly to the user
    """
    print("--- [Node] Empathetic Synthesizer (ReAct) ---")
    raw_data = state["tool_output"]
    sentiment = state["sentiment"]
    user_text = state["messages"][-1].content
    # memory = state.get("long_term_memory", "No History.")

    # Escalation detection from message content or a future flag
    escalated = (
        state.get("escalation_needed", False)
        or "TICKET_CREATED" in raw_data
        or "TICKET_UPDATED" in raw_data
    )

    base_prompt = "You are a helpful Telecom Assistant for NextGen Telecom Company."

    # Tone rule (same as NextGen_Chat)
    if sentiment in ["Angry", "Frustrated"]:
        tone = "The user is upset. Start with a sincere apology and reassurance."
    elif sentiment == "Happy":
        tone = "The user is happy. Respond with high energy."
    else:
        tone = "The user is neutral. Be professional and polite."

    # Context logic
    if escalated:
        if "TICKET_CREATED" in raw_data:
            context_instr = f"""
ACTION TAKEN: A new support ticket was created because the issue requires human attention.
DETAILS: {raw_data}
INSTRUCTION: Inform the user clearly that a ticket has been created. Mention the Ticket ID and who it is assigned to (if present). Assure them about next steps.
"""
        elif "TICKET_UPDATED" in raw_data:
            context_instr = f"""
ACTION TAKEN: An existing open ticket for this customer was found and updated with new information.
DETAILS: {raw_data}
INSTRUCTION: Tell the user you noticed they already had an open case, so instead of creating a duplicate, you have updated their existing ticket with this new information.
"""
        else:
            context_instr = (
                "Apologize that you cannot resolve the issue directly and suggest they contact support."
            )
    else:
        context_instr = (
            f"Answer the user's question using this specific data/context: {raw_data}"
        )

    full_prompt = f"{base_prompt}\n{tone}\n{context_instr}"

    msg = REASONING_LLM.invoke(f"{full_prompt}\n\nUser Query: {user_text}")
    return {"messages": [msg]}


# ---------------------------------------------------------------------------
# 6. LANGGRAPH
# ---------------------------------------------------------------------------

react_workflow = StateGraph(ReactState)

react_workflow.add_node("react_reasoner", react_reasoner_node)
react_workflow.add_node("react_synthesizer", response_synthesizer)

react_workflow.set_entry_point("react_reasoner")
react_workflow.add_edge("react_reasoner", "react_synthesizer")
react_workflow.add_edge("react_synthesizer", END)

react_app = react_workflow.compile()


# ---------------------------------------------------------------------------
# 7. ENTRY FUNCTION FOR ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_react_agent(user_query: str, sentiment: SENTIMENT_OPTIONS, customer_id: str = "1", chat_history: str = "No History.") -> str:
    """
    Called by Orchestrator.

    - Sets CURRENT_CUSTOMER_ID and CURRENT_SENTIMENT so tools can use them.
    - Builds initial ReactState.
    - Runs the LangGraph.
    - Returns the final synthesized answer (string).
    """
    global CURRENT_CUSTOMER_ID, CURRENT_SENTIMENT

    print(f"\n🧠 [ReAct Graph] Starting for: {user_query} | Sentiment={sentiment} | Customer={customer_id}")
    CURRENT_CUSTOMER_ID = customer_id
    CURRENT_SENTIMENT = sentiment

    initial_state: ReactState = {
        "messages": [HumanMessage(content=user_query)],
        "sentiment": sentiment,
        "customer_id": customer_id,
        "tool_output": "",
        "escalation_needed": False,
        "long_term_memory": chat_history
    }

    result_state = react_app.invoke(initial_state)
    final_msg = result_state["messages"][-1].content
    return final_msg, result_state


# if __name__ == "__main__":
#     # Simple manual test
#     while True:
#         q = input("User: ")
#         if q.lower() in ["exit", "quit"]:
#             break
#         # simulate frustrated user with a router/broadband issue
#         ans = run_react_agent(
#             q,
#             sentiment="Frustrated",
#             customer_id="2",
#         )
#         print(f"\nReAct Bot: {ans}\n")
