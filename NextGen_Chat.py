import sqlite3
import operator
import os
import json
import uuid
import datetime
import httpx
import re
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
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

client = httpx.Client(verify=False)

tiktoken_cache_dir = "tiktoken_cache"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir
assert os.path.exists(os.path.join(tiktoken_cache_dir, "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"))

DB_PATH = "./Database/NextGen1.db"
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
    embedding_function = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model="azure/genailab-maas-text-embedding-3-large",
        api_key=OPENAI_API_KEY, # Replace with your actual API key
        http_client=client
    )
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

llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",
    model="genailab-maas-gpt-4o",
    api_key=OPENAI_API_KEY,
    http_client=client
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
    # NEW: Store the summarized context
    long_term_memory: str


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


class TicketClassification(BaseModel):
    issue_type_id: int = Field(description="The ID of the most relevant issue type from the provided list.")
    reasoning: str = Field(description="Why this issue type was selected.")

# =============================================================================
# 4. NODE IMPLEMENTATIONS
# =============================================================================

# --- NEW: MEMORY MANAGER NODE ---
def memory_manager(state: AgentState):
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
        new_summary = llm.invoke(summary_prompt).content
        
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
    2. 'general_inquiry': General policies/how-to which no need DataBase.
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
    memory = state.get("long_term_memory", "")
    cust_id = state.get("customer_id")

    docs = sop_retriever.invoke(user_text)
    context = "\n".join([d.page_content for d in docs])

    if not context:
        print("   ❌ No SOP Context found. Escalating.")
        return {"tool_output": "Unknown Issue", "escalation_needed": True}

    structured_llm = llm.with_structured_output(HybridResponse)
    system_prompt = f"""You are a Technical Troubleshooter.
    History: {memory}
    Context (SOP): {context}
    DB Schema: {db.get_table_info()}
    Customer ID: {cust_id}

    1. If you can answer using SOP/DB, set can_resolve=True.
    2. If the query is unclear, irrelevant to Telecom, or SOP is missing, set can_resolve=False.
    3. If SOP needs DB, set needs_sql=True and extract query.
    """

    chain = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{query}")]) | structured_llm
    analysis = chain.invoke({"query": user_text})

    # # Check for Fallback
    # if not analysis.can_resolve:
    #     print("   ⚠️ Agent cannot resolve query. Escalating.")
    #     return {"tool_output": "Agent unable to resolve", "escalation_needed": True}

    # final_context = f"SOP Guidelines: {context}\n"
    # if analysis.needs_sql:
    #     try:
    #         sql_result = db.run(analysis.sql_query)
    #         final_context += f"\nLIVE DATA: {sql_result}"
    #     except Exception as e:
    #         final_context += f"\nDB Error: {e}"

    # return {"tool_output": final_context}

    # --- REGEX FALLBACK (Safety Net) ---
    sql_query = analysis.sql_query
    needs_execution = analysis.needs_sql
    
    # If LLM said "False" but text clearly has SQL, override it
    if not needs_execution:
        sql_match = re.search(r"SELECT .* FROM .* WHERE .*", context, re.IGNORECASE)
        if sql_match:
            print("   ⚠️ Regex detected SQL that LLM missed. Executing...")
            sql_query = sql_match.group(0).replace(":customer_id", f"'{state['customer_id']}'").replace("?", f"'{state['customer_id']}'")
            needs_execution = True

    final_context = f"SOP Guidelines (Internal): {context}\n"
    
    if needs_execution and sql_query:
        print(f"   ⚙️ Executing SQL: {sql_query}")
        try:
            sql_result = db.run(sql_query)
            final_context += f"\nLIVE SYSTEM DATA: {sql_result}"
            print(f"   ✅ Data: {sql_result}")
        except Exception as e:
            print(f"   ❌ SQL Failed: {e}")
            final_context += f"\nData Check Failed: {e}"
    else:
        print("   ℹ️ Pure Text Answer.")
            
    return {"tool_output": final_context}


# --- UPDATED: INTELLIGENT TICKET AGENT ---
def ticket_agent(state: AgentState):
    """
    1. Checks for EXISTING Open tickets. If found, updates them.
    2. If NO open ticket, uses LLM to classify and create a new one.
    """
    print("--- [Node] Intelligent Ticket Agent ---")
    
    cust_id = state.get("customer_id")
    description = state["messages"][-1].content
    memory = state.get("long_term_memory", "No history")
    sentiment = state.get("sentiment", "Neutral")
    
    # Combined description ensures Human Agent knows the full story
    full_description = f"USER COMPLAINT: {description}\n\nCONTEXT_HISTORY: {memory}"

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # --- LOGIC 1: CHECK FOR EXISTING OPEN TICKET ---
        cursor.execute(
            "SELECT ticket_id, description, assigned_agent_id FROM tickets WHERE customer_id = ? AND status = 'Open'", 
            (cust_id,)
        )
        existing_ticket = cursor.fetchone()
        
        if existing_ticket:
            # UPDATE EXISTING TICKET
            t_id = existing_ticket['ticket_id']
            old_desc = existing_ticket['description']
            agent_id = existing_ticket['assigned_agent_id']
            
            # Fetch agent name for context
            cursor.execute("SELECT name FROM agents WHERE agent_id = ?", (agent_id,))
            agent_row = cursor.fetchone()
            agent_name = agent_row['name'] if agent_row else "Unknown"
            
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            new_desc = f"{old_desc} || [Update {timestamp}]: {description}"
            
            cursor.execute("UPDATE tickets SET description = ? WHERE ticket_id = ?", (new_desc, t_id))
            conn.commit()
            
            msg = f"TICKET_UPDATED: You already have an open ticket #{t_id} with {agent_name}. I have added this new information to it."
            print(f"   🔄 {msg}")
            return {"tool_output": msg}

        # --- LOGIC 2: CREATE NEW TICKET (If no existing one) ---
        
        # 1. Fetch Issue Types
        cursor.execute("SELECT issue_type_id, name, description FROM issue_types")
        issue_types = [dict(row) for row in cursor.fetchall()]
        
        # Identify "General" as a fallback ID
        general_type_id = next((it['issue_type_id'] for it in issue_types if "General" in it['name']), 1)

        # 2. LLM Classification
        selected_type_id = general_type_id # Default initialization
        selected_type_name = "General Support"
        
        try:
            structured_llm = llm.with_structured_output(TicketClassification)
            classification_prompt = f"""
            You are a Ticket Classifier. Match the user complaint to the best Issue Type.
            
            Available Issue Types:
            {json.dumps(issue_types, indent=2)}
            
            User Complaint: "{description}"
            
            Return the exact issue_type_id.
            """
            classification = structured_llm.invoke(classification_prompt)
            
            # Validation Check
            if classification and classification.issue_type_id:
                selected_type_id = classification.issue_type_id
                selected_type_name = next((it['name'] for it in issue_types if it['issue_type_id'] == selected_type_id), "General Support")
            else:
                print("   ⚠️ LLM returned None for ID. Using Fallback.")
                
        except Exception as llm_e:
            print(f"   ⚠️ Classification Warning: {llm_e}. Using General Fallback.")
            # Keep defaults set above

        print(f"   Using Issue Type: {selected_type_name} (ID: {selected_type_id})")

        # 3. Map Department
        department_mapping = {
            "Billing Dispute": "Billing", "Wrong Recharge": "Billing", "Plan Benefits Not Added": "Billing",
            "Poor 4G/5G Signal": "Network", "Roaming Issue": "Network",
            "Slow Internet": "Broadband", "Router Configuration Issue": "Broadband", "Broadband Outage": "Broadband",
            "SIM Issue": "General Support", "Porting Issue": "General Support", "Data Exhausted Fast": "General Support"
        }
        target_department = department_mapping.get(selected_type_name, "General Support")

        # 4. Find Best Agent
        cursor.execute("SELECT agent_id, name FROM agents WHERE department = ? ORDER BY rating DESC LIMIT 1", (target_department,))
        best_agent = cursor.fetchone()
        
        if best_agent:
            assigned_agent_id = best_agent['agent_id']
            agent_name = best_agent['name']
        else:
            cursor.execute("SELECT agent_id, name FROM agents ORDER BY rating DESC LIMIT 1")
            fallback = cursor.fetchone()
            assigned_agent_id = fallback['agent_id']
            agent_name = fallback['name']

        # 5. Insert Ticket
        priority = "High" if sentiment in ["Angry", "Frustrated"] else "Medium"
        created_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute('''
            INSERT INTO tickets (customer_id, issue_type_id, description, status, priority, created_at, assigned_agent_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (cust_id, selected_type_id, full_description, "Open", priority, created_at, assigned_agent_id))
        
        ticket_id = cursor.lastrowid
        conn.commit()
        
        msg = f"TICKET_CREATED: Ticket #{ticket_id} created for '{selected_type_name}'. Assigned to {agent_name} ({target_department})."
        print(f"   ✅ {msg}")
        return {"tool_output": msg}
        
    except Exception as e:
        print(f"   ❌ Ticket Logic Failed: {e}")
        return {"tool_output": "Failed to manage ticket due to system error."}
    finally:
        conn.close()


def response_synthesizer(state: AgentState):
    print("--- [Node] Empathetic Synthesizer ---")
    raw_data = state["tool_output"]
    sentiment = state["sentiment"]
    user_text = state["messages"][-1].content
    
    # Check if this was an escalation (Ticket Created OR Updated)
    escalated = state.get("escalation_needed", False) or \
                "TICKET_CREATED" in raw_data or \
                "TICKET_UPDATED" in raw_data
    
    base_prompt = "You are a helpful Telecom Assistant."
    
    # 1. Tone Setting
    if sentiment in ["Angry", "Frustrated"]:
        tone = "The user is upset. Start with a sincere apology and reassurance."
    elif sentiment == "Happy":
        tone = "The user is happy. Respond with high energy."
    else:
        tone = "The user is neutral. Be professional and polite."

    # 2. Context Logic
    if escalated:
        if "TICKET_CREATED" in raw_data:
            context_instr = f"""
            ACTION TAKEN: A new support ticket was created because the issue requires human attention.
            DETAILS: {raw_data}
            INSTRUCTION: Inform the user clearly that a ticket has been created. Mention the Ticket ID and the Agent Name assigned. Assure them they are in good hands.
            """
        elif "TICKET_UPDATED" in raw_data:
            context_instr = f"""
            ACTION TAKEN: Found an existing open ticket for this customer and added their new comments to it.
            DETAILS: {raw_data}
            INSTRUCTION: Tell the user you noticed they already had an open case, so instead of creating a duplicate, you have updated their existing ticket with this new information. This is efficient and helpful.
            """
        else:
            # Fallback if escalation happened but no ticket info (rare error case)
            context_instr = "Apologize that you cannot resolve the issue directly and suggest they call our hotline."
    else:
        # Standard Greeting / SOP / DB Answer
        context_instr = f"Answer the user's question using this specific data/context: {raw_data}"

    full_prompt = f"{base_prompt}\n{tone}\n{context_instr}\nNEVER output SQL queries to the user."
    
    msg = llm.invoke(f"{full_prompt}\n\nUser Query: {user_text}")
    return {"messages": [msg]}

# =============================================================================
# 5. GRAPH CONSTRUCTION
# =============================================================================

workflow = StateGraph(AgentState)

workflow.add_node("memory_manager", memory_manager) # Entry Point
workflow.add_node("sentiment_scanner", sentiment_agent)
workflow.add_node("router", router_node)
workflow.add_node("greeting_agent", greeting_node)
workflow.add_node("general_agent", general_inquiry_node)
workflow.add_node("db_agent", db_agent)
workflow.add_node("hybrid_agent", sop_troubleshooter_node)
workflow.add_node("ticket_agent", ticket_agent)  # NEW
workflow.add_node("synthesizer", response_synthesizer)

workflow.set_entry_point("memory_manager")
workflow.add_edge("memory_manager", "sentiment_scanner")
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
# display(Image(app.get_graph().draw_mermaid_png()))

# =============================================================================
# 6. INTERACTIVE LOOP (Fixes Memory Issue)
# =============================================================================

def run_interactive_chat():
    print(">>> 🤖 TELECOM BOT LIVE SESSION (Type 'exit' to quit)")
    
    # 1. State must be maintained here if not using a Checkpointer
    chat_history = [] 
    customer_id = "2"
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]: break
        
        # Append to local history
        chat_history.append(HumanMessage(content=user_input))
        
        # Invoke Graph with accumulated history
        inputs = {
            "messages": chat_history, 
            "customer_id": customer_id,
        }
        
        try:
            result = app.invoke(inputs)
            bot_response = result["messages"][-1].content
            
            # Append Bot response to history so it grows
            chat_history.append(AIMessage(content=bot_response))
            
            print(f"Bot: {bot_response}")
            
            print(f"Chat History Length: {len(chat_history)}")
            # Debug: Check Memory
            if len(chat_history) > 6:
                chat_history = chat_history[-2:]
                # print(f"   [Debug] Current Summary: {result.get('long_term_memory')}")

        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_interactive_chat()
    # print(">>> 🤖 TELECOM BOT: HUMAN HANDOFF ENABLED")

    # while True:
    #     user_input = input("Enter your query: ")
    #     if user_input.lower() == 'exit':
    #         break
    #     # TEST 1: Unknown Issue (Triggers Escalation)
    #     print("\n--- TEST 1: Unknown Query (Expect Ticket) ---")
    #     try:
    #         # A query that has NO SOP coverage
    #         res1 = app.invoke({"messages": [HumanMessage(content=user_input)],
    #                         "customer_id": "1"})
            
    #         print(f"Bot: {res1['messages'][-1].content}")
    #         # print(f"STATE: {res1}")
    #     except Exception as e:
    #         print(f"Test 1 Error: {e}")

    # # TEST 2: Explicit Handoff
    # print("\n--- TEST 2: Explicit Human Request ---")
    # try:
    #     res2 = app.invoke(
    #         {"messages": [HumanMessage(content="I want to talk to a human agent now!")], "customer_id": "2"})
    #     print(f"Bot: {res2['messages'][-1].content}")
    # except Exception as e:
    #     print(f"Test 2 Error: {e}")