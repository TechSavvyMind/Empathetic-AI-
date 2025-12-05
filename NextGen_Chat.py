import sqlite3
import operator
import os
import json
import uuid
import datetime
from typing import Annotated, TypedDict, Union, List, Literal
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from IPython.display import Image, display

load_dotenv()

# !!! IMPORTANT: Set your OpenAI API Key here !!!
# os.environ["OPENAI_API_KEY"] = "sk-..."
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

DB_PATH = "./Database/NextGen.db"
CHROMA_DB_DIR = "./sop_embeddings/chroma_store"
COLLECTION_NAME = "telecom_sops"
JSONL_PATH = "./sop_embeddings/telecom_sop_chunks.jsonl"


# =============================================================================
# 2. DATA LOADING & DATABASE SETUP (Updated with Ticket API Schema)
# =============================================================================

def setup_database_schema():
    """Creates the schema, aligning with your 'models.py' and 'logical_model.pdf'."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. CUSTOMERS
    c.execute('''CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY, name TEXT, phone_number TEXT, 
        account_status TEXT, kyc_status TEXT, pincode TEXT)''')

    # 2. OUTAGE_AREAS
    c.execute('''CREATE TABLE IF NOT EXISTS outage_areas (
        outage_id TEXT PRIMARY KEY, city TEXT, pincode TEXT, 
        issue_description TEXT, expected_resolution TEXT)''')

    # 3. CUSTOMER_USAGE
    c.execute('''CREATE TABLE IF NOT EXISTS customer_usage (
        usage_id TEXT PRIMARY KEY, customer_id TEXT, date TEXT, 
        mobile_data_used_gb REAL, plan_limit_gb REAL)''')

    # 4. INVOICES
    c.execute('''CREATE TABLE IF NOT EXISTS invoices (
        invoice_id TEXT PRIMARY KEY, customer_id TEXT, amount REAL, 
        due_date TEXT, paid_status TEXT, billing_period TEXT)''')

    # 5. TRANSACTIONS
    c.execute('''CREATE TABLE IF NOT EXISTS transactions (
        txn_id TEXT PRIMARY KEY, customer_id TEXT, amount REAL, 
        txn_date TEXT, transaction_status TEXT)''')

    # 6. SUBSCRIPTIONS
    c.execute('''CREATE TABLE IF NOT EXISTS subscriptions (
        subscription_id TEXT PRIMARY KEY, customer_id TEXT, plan_id TEXT, 
        status TEXT, start_date TEXT)''')

    # 7. ISSUE_TYPES (From your models.py)
    c.execute('''CREATE TABLE IF NOT EXISTS issue_types (
        issue_type_id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT, 
        description TEXT)''')

    # 8. AGENTS (From your models.py)
    c.execute('''CREATE TABLE IF NOT EXISTS agents (
        agent_id INTEGER PRIMARY KEY AUTOINCREMENT, 
        name TEXT, 
        department TEXT, 
        shift_timing TEXT, 
        rating REAL)''')

    # 9. TICKETS (Updated to match models.py)
    c.execute('''CREATE TABLE IF NOT EXISTS tickets (
        ticket_id INTEGER PRIMARY KEY AUTOINCREMENT, 
        customer_id TEXT, 
        issue_type_id INTEGER, 
        description TEXT, 
        status TEXT, 
        priority TEXT, 
        created_at TEXT, 
        closed_at TEXT, 
        assigned_agent_id INTEGER,
        FOREIGN KEY(customer_id) REFERENCES customers(customer_id),
        FOREIGN KEY(issue_type_id) REFERENCES issue_types(issue_type_id),
        FOREIGN KEY(assigned_agent_id) REFERENCES agents(agent_id)
    )''')

    # --- Insert Mock Data ---
    # Issue Types
    c.execute(
        "INSERT OR IGNORE INTO issue_types (issue_type_id, name, description) VALUES (1, 'Technical', 'Network or Device issues')")
    c.execute(
        "INSERT OR IGNORE INTO issue_types (issue_type_id, name, description) VALUES (2, 'Billing', 'Invoice and Balance issues')")
    c.execute(
        "INSERT OR IGNORE INTO issue_types (issue_type_id, name, description) VALUES (3, 'General', 'General inquiries')")

    # Agents
    c.execute(
        "INSERT OR IGNORE INTO agents (agent_id, name, department, shift_timing, rating) VALUES (101, 'John Doe', 'Technical', 'Morning', 4.5)")
    c.execute(
        "INSERT OR IGNORE INTO agents (agent_id, name, department, shift_timing, rating) VALUES (102, 'Jane Smith', 'Billing', 'Day', 4.8)")

    # Customers
    c.execute(
        "INSERT OR REPLACE INTO customers VALUES ('CUST_001', 'Alice', '555-0101', 'Active', 'Verified', '10001')")
    c.execute("INSERT OR REPLACE INTO outage_areas VALUES ('OUT_01', 'New York', '10001', 'Fiber Cut', '4 Hours')")
    c.execute("INSERT OR REPLACE INTO customers VALUES ('CUST_002', 'Bob', '555-0102', 'Active', 'Verified', '10002')")
    c.execute(
        "INSERT OR REPLACE INTO invoices VALUES ('INV_001', 'CUST_002', 150.00, '2023-11-01', 'Unpaid', 'Oct-2023')")

    conn.commit()
    conn.close()
    print("✅ Database Schema & Mock Data Ready.")


def load_documents_from_jsonl(file_path: str) -> List[Document]:
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                data = json.loads(line)
                text_content = data.get("text")
                if not text_content: continue
                tags = data.get("tags", [])
                if isinstance(tags, list): tags = ", ".join(tags)
                meta = {"chunk_id": data.get("chunk_id"), "sop_id": data.get("sop_id"), "tags": tags}
                documents.append(Document(page_content=text_content, metadata=meta))
        return documents
    except FileNotFoundError:
        return []
    except Exception as e:
        return []


def setup_vector_store():
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)
    if os.path.exists(CHROMA_DB_DIR) and os.path.isdir(CHROMA_DB_DIR):
        print("✅ Found existing ChromaDB. Loading...")
        return Chroma(persist_directory=CHROMA_DB_DIR, collection_name=COLLECTION_NAME,
                      embedding_function=embedding_function).as_retriever()

    print(f"⚙️ Building new ChromaDB from {JSONL_PATH}...")
    docs = load_documents_from_jsonl(JSONL_PATH)
    if docs:
        vectorstore = Chroma.from_documents(docs, embedding_function, collection_name=COLLECTION_NAME,
                                            persist_directory=CHROMA_DB_DIR)
        return vectorstore.as_retriever()
    else:
        return Chroma.from_texts(["No Content"], embedding_function).as_retriever()


# setup_database_schema()
sop_retriever = setup_vector_store()
db = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# =============================================================================
# 3. STATE & DEFINITIONS
# =============================================================================

# Added 'human_handoff' to routes
ROUTE_OPTIONS = Literal["greeting", "general_inquiry", "db_inquiry", "troubleshoot", "human_handoff"]
SENTIMENT_OPTIONS = Literal["Angry", "Happy", "Sad", "Neutral", "Frustrated"]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sentiment: SENTIMENT_OPTIONS
    route_intent: ROUTE_OPTIONS
    customer_id: str
    tool_output: str
    # New: Track if we need to escalate
    escalation_needed: bool


# --- Pydantic Models ---
class SentimentResponse(BaseModel):
    emotion: SENTIMENT_OPTIONS = Field(description="The emotional tone.")


class RouteResponse(BaseModel):
    step: ROUTE_OPTIONS = Field(description="The classification of the user query.")


class HybridResponse(BaseModel):
    needs_sql: bool = Field(description="True if SOP requires DB.")
    sql_query: str = Field(description="SQL query or empty.")
    direct_answer: str = Field(description="Answer if no SQL needed.")
    # New: Fallback detection
    can_resolve: bool = Field(description="False if the SOP/DB doesn't have the answer.")


class SqlQuery(BaseModel):
    query: str = Field(description="Valid SQL query.")


# =============================================================================
# 4. NODE IMPLEMENTATIONS
# =============================================================================

def sentiment_agent(state: AgentState):
    print("\n--- [Node] Sentiment Agent ---")
    user_text = state["messages"][-1].content
    structured_llm = llm.with_structured_output(SentimentResponse)
    system_prompt = "Classify emotion: Neutral (polite/help), Angry/Frustrated (complaints/errors), Happy."
    chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{text}")]) | structured_llm
    result = chain.invoke({"text": user_text})
    print(f"   Detected Sentiment: {result.emotion}")
    return {"sentiment": result.emotion}


def router_node(state: AgentState):
    print("--- [Node] Router ---")
    user_text = state["messages"][-1].content
    structured_llm = llm.with_structured_output(RouteResponse)

    system_prompt = """Classify intent:
    1. 'greeting': Hello, Hi.
    2. 'general_inquiry': General policies/how-to.
    3. 'db_inquiry': My balance, bill, account status.
    4. 'troubleshoot': Slow internet, billing disputes.
    5. 'human_handoff': "Talk to human", "I want an agent", "Connect me to support".
    """

    chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")]) | structured_llm
    result = chain.invoke({"query": user_text})
    print(f"   Routing to: {result.step.upper()}")

    # If routed directly to handoff, set flag
    is_handoff = (result.step == "human_handoff")
    return {"route_intent": result.step, "escalation_needed": is_handoff}


def greeting_node(state: AgentState):
    return {"tool_output": "Greeting"}


def general_inquiry_node(state: AgentState):
    print("--- [Node] General Inquiry Agent ---")
    user_text = state["messages"][-1].content
    docs = sop_retriever.invoke(user_text)

    # Logic: If no docs found, escalate
    if not docs:
        print("   ❌ No SOP found. Triggering Escalation.")
        return {"tool_output": "No info found", "escalation_needed": True}

    context = "\n".join([d.page_content for d in docs])
    return {"tool_output": f"SOP Info: {context}"}


def db_agent(state: AgentState):
    print("--- [Node] Direct DB Agent ---")
    user_text = state["messages"][-1].content
    cust_id = state.get("customer_id")

    structured_llm = llm.with_structured_output(SqlQuery)
    prompt = f"Schema: {db.get_table_info()}. Write SQL for customer_id='{cust_id}' based on: {user_text}"
    try:
        result = (ChatPromptTemplate.from_template(prompt) | structured_llm).invoke({})
        data = db.run(result.query)
        if not data:
            # Empty result might trigger escalation if it looks like an error
            return {"tool_output": "No records found.", "escalation_needed": False}
        return {"tool_output": f"DB Record: {data}"}
    except Exception as e:
        print(f"   ❌ DB Error: {e}")
        return {"tool_output": str(e), "escalation_needed": True}


def sop_troubleshooter_node(state: AgentState):
    print("--- [Node] Hybrid Troubleshooter ---")
    user_text = state["messages"][-1].content
    cust_id = state.get("customer_id")

    docs = sop_retriever.invoke(user_text)
    context = "\n".join([d.page_content for d in docs])

    if not context:
        print("   ❌ No SOP Context found. Escalating.")
        return {"tool_output": "Unknown Issue", "escalation_needed": True}

    structured_llm = llm.with_structured_output(HybridResponse)
    system_prompt = f"""You are a Technical Troubleshooter.
    Context (SOP): {context}
    DB Schema: {db.get_table_info()}

    1. If you can answer using SOP/DB, set can_resolve=True.
    2. If the query is unclear, irrelevant to Telecom, or SOP is missing, set can_resolve=False.
    3. If SOP needs DB, set needs_sql=True and write query.
    """

    chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")]) | structured_llm
    analysis = chain.invoke({"query": user_text})

    # Check for Fallback
    if not analysis.can_resolve:
        print("   ⚠️ Agent cannot resolve query. Escalating.")
        return {"tool_output": "Agent unable to resolve", "escalation_needed": True}

    final_context = f"SOP Guidelines: {context}\n"
    if analysis.needs_sql:
        try:
            sql_result = db.run(analysis.sql_query)
            final_context += f"\nLIVE DATA: {sql_result}"
        except Exception as e:
            final_context += f"\nDB Error: {e}"

    return {"tool_output": final_context}


# --- NEW NODE: TICKET CREATION (ESCALATION) ---
def ticket_agent(state: AgentState):
    """Creates a ticket in the DB when LLM fails."""
    print("--- [Node] Ticket Creation Agent ---")

    cust_id = state.get("customer_id")
    description = state["messages"][-1].content
    sentiment = state.get("sentiment", "Neutral")

    # 1. Determine Priority based on Sentiment
    priority = "High" if sentiment in ["Angry", "Frustrated"] else "Medium"

    # 2. Determine Issue Type (Simple heuristic or LLM)
    # Mapping: 1=Technical, 2=Billing, 3=General
    issue_type_id = 3
    if "bill" in description.lower() or "charge" in description.lower():
        issue_type_id = 2
    elif "internet" in description.lower() or "slow" in description.lower():
        issue_type_id = 1

    # 3. Insert into DB (Mimicking crud.py create_ticket)
    created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO tickets (customer_id, issue_type_id, description, status, priority, created_at, assigned_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (cust_id, issue_type_id, description, "Open", priority, created_at, 1))

        ticket_id = cursor.lastrowid
        conn.commit()
        conn.close()

        output_msg = f"TICKET_CREATED: ID #{ticket_id}. Priority: {priority}. Assigned to Agent 1."
        print(f"   ✅ {output_msg}")
        return {"tool_output": output_msg}

    except Exception as e:
        print(f"   ❌ Ticket Creation Failed: {e}")
        return {"tool_output": "Failed to create ticket system error."}


def response_synthesizer(state: AgentState):
    print("--- [Node] Empathetic Synthesizer ---")
    raw_data = state["tool_output"]
    sentiment = state["sentiment"]
    user_text = state["messages"][-1].content
    escalated = state.get("escalation_needed", False) or "TICKET_CREATED" in raw_data

    base_prompt = "You are a helpful Telecom Assistant."

    if sentiment in ["Angry", "Frustrated"]:
        tone = "The user is upset. Start with a sincere apology. Be concise but reassuring. Acknowledge their frustration."
    elif sentiment == "Happy":
        tone = "The user is happy. Respond with high energy and gratitude."
    else:
        tone = "The user is neutral. Be professional, polite, and direct."

    if escalated:
        if "TICKET_CREATED" in raw_data:
            context_instr = f"Inform the user that because the issue is complex, you have created a support ticket. \nTicket Details: {raw_data}"
        else:
            # Fallback if ticket creation failed or wasn't triggered yet (shouldn't happen with correct edges)
            context_instr = "Apologize that you cannot help and suggest calling support."
    else:
        context_instr = f"Answer using: {raw_data}"

    full_prompt = f"{base_prompt}\n{tone}\n{context_instr}"
    msg = llm.invoke(f"{full_prompt}\n\nQuery: {user_text}")
    return {"messages": [msg]}


# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

workflow = StateGraph(AgentState)

workflow.add_node("sentiment_scanner", sentiment_agent)
workflow.add_node("router", router_node)
workflow.add_node("greeting_agent", greeting_node)
workflow.add_node("general_agent", general_inquiry_node)
workflow.add_node("db_agent", db_agent)
workflow.add_node("hybrid_agent", sop_troubleshooter_node)
workflow.add_node("ticket_agent", ticket_agent)  # NEW
workflow.add_node("synthesizer", response_synthesizer)

workflow.set_entry_point("sentiment_scanner")
workflow.add_edge("sentiment_scanner", "router")


def route_decision(state):
    # Immediate handoff check
    if state.get("escalation_needed"):
        return "ticket_agent"
    return state["route_intent"]


workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "greeting": "greeting_agent",
        "general_inquiry": "general_agent",
        "db_inquiry": "db_agent",
        "troubleshoot": "hybrid_agent",
        "human_handoff": "ticket_agent",  # Explicit request
        "ticket_agent": "ticket_agent"  # Fallback safety
    }
)


def check_escalation(state):
    # After a tool runs, check if it failed
    if state.get("escalation_needed"):
        return "ticket_agent"
    return "synthesizer"


# Tools -> Check -> Synthesizer OR Ticket Agent
workflow.add_conditional_edges("general_agent", check_escalation,
                               {"synthesizer": "synthesizer", "ticket_agent": "ticket_agent"})
workflow.add_conditional_edges("db_agent", check_escalation,
                               {"synthesizer": "synthesizer", "ticket_agent": "ticket_agent"})
workflow.add_conditional_edges("hybrid_agent", check_escalation,
                               {"synthesizer": "synthesizer", "ticket_agent": "ticket_agent"})

# Greeting and Ticket Agent always go to synthesizer
workflow.add_edge("greeting_agent", "synthesizer")
workflow.add_edge("ticket_agent", "synthesizer")
workflow.add_edge("synthesizer", END)

app = workflow.compile()

# Graph Structure
display(Image(app.get_graph().draw_mermaid_png()))

# =============================================================================
# 6. TEST SCENARIOS
# =============================================================================

if __name__ == "__main__":
    print(">>> 🤖 TELECOM BOT: HUMAN HANDOFF ENABLED")

    # TEST 1: Unknown Issue (Triggers Escalation)
    print("\n--- TEST 1: Unknown Query (Expect Ticket) ---")
    try:
        # A query that has NO SOP coverage
        res1 = app.invoke({"messages": [HumanMessage(content="My satellite dish on the roof is leaking water.")],
                           "customer_id": "1"})
        print(f"Bot: {res1['messages'][-1].content}")
    except Exception as e:
        print(f"Test 1 Error: {e}")

    # TEST 2: Explicit Handoff
    print("\n--- TEST 2: Explicit Human Request ---")
    try:
        res2 = app.invoke(
            {"messages": [HumanMessage(content="I want to talk to a human agent now!")], "customer_id": "2"})
        print(f"Bot: {res2['messages'][-1].content}")
    except Exception as e:
        print(f"Test 2 Error: {e}")