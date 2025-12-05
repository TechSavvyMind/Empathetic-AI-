import os
import sqlite3
from typing import List
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_classic import hub  # ← CHANGED: langchain-classic for hub
from ticket_service import create_ticket_tool

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# --- CONFIGURATION ---
DB_PATH = "./Database/NextGen.db"  # Path to the Telecom Data (from Simple_agent)

# --- TOOLS FOR REACT AGENT ---

@tool
def query_telecom_db(query: str):
    """
    Executes a read-only SQL query against the Telecom Database (NextGen.db).
    Use this to look up customer plans, outages, invoices, and transactions.
    Tables: customers, outage_areas, customer_usage, invoices, transactions, subscriptions.
    Input must be a valid SQLite SELECT query.
    """
    try:
        # Security check (simple)
        if "drop" in query.lower() or "delete" in query.lower() or "update" in query.lower():
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
    Escalates the issue to a human agent by creating a support ticket.
    Use this if you cannot resolve the issue after analyzing the database.
    """
    # Mapping priority to Issue Type implicitly for simplicity
    issue_type = 2 if "technical" in issue_summary.lower() else 3 if "bill" in issue_summary.lower() else 1
    return create_ticket_tool(description=issue_summary, priority=priority, issue_type_id=issue_type)

# --- AGENT CONSTRUCTION ---

def build_react_agent():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)  # ← Fixed typo
    
    tools = [query_telecom_db, escalate_issue]
    
    # Pull ReAct prompt from Hub (works with v1)
    prompt = hub.pull("hwchase17/react")
    
    # Create custom system prompt for v1 agent
    system_prompt = """
    You are a Senior Level 3 Support Agent.
    Your job is to perform Root Cause Analysis.
    
    1. **Think**: Break down the user's complaint.
    2. **Investigate**: Use 'query_telecom_db' to check tables (customers, invoices, outage_areas, etc.).
    3. **Analyze**: Compare the user's claim against the DB data.
    4. **Act**: If you find the root cause, explain it. If it's a valid system error you can't fix, use 'escalate_issue'.
    
    Example: User says "Internet slow". 
    - Check 'customers' for status.
    - Check 'outage_areas' for their pincode.
    - Check 'customer_usage' for data caps.
    
    Always use tools before responding. Think step-by-step.
    """
    
    # v1.0 create_agent API
    agent = create_agent(
        llm, 
        tools, 
        system_prompt=system_prompt  # ← v1 uses system_prompt parameter
    )
    
    return agent  # ← v1 agent is directly invocable, no AgentExecutor needed

# Shared instance
react_agent = build_react_agent()  # ← Renamed for clarity

def run_react_agent(user_query: str):
    """Entry point for the Orchestrator."""
    print(f"\n🧠 [ReAct Agent] Starting deep analysis for: {user_query}")
    # v1.0 input format uses messages
    result = react_agent.invoke({
        "messages": [{"role": "user", "content": user_query}]
    })
    # Extract final output from messages
    return result["messages"][-1].content  # ← v1 returns AIMessage in messages list