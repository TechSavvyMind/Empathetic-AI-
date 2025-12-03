import sqlite3
import operator
import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict, Union, List, Literal
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.utilities import SQLDatabase
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

load_dotenv()

# !!! IMPORTANT: Set your OpenAI API Key here !!!
# os.environ["OPENAI_API_KEY"] = "sk-..."
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# =============================================================================
# 1. SETUP (Same as before)
# =============================================================================
DB_PATH = "./Database/NextGen.db"
SOP_FILE_PATH = "sop_document.txt"
CHROMA_DB_DIR = "./chroma_db"


def setup_resources():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        conn.cursor().execute(
            "CREATE TABLE IF NOT EXISTS customers (customer_id TEXT, full_name TEXT, data_balance_gb REAL)")
        conn.cursor().execute("INSERT INTO customers VALUES ('CUST_001', 'Alice', 45.5)")
        conn.commit()
        conn.close()

    db_connection = SQLDatabase.from_uri(f"sqlite:///{DB_PATH}")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", api_key=GOOGLE_API_KEY)

    if os.path.exists(CHROMA_DB_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embedding_function)
    else:
        vectorstore = Chroma.from_texts(["Router reset: hold power 10s"], embedding=embedding_function,
                                        persist_directory=CHROMA_DB_DIR)

    return db_connection, vectorstore.as_retriever()


db, sop_retriever = setup_resources()
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# =============================================================================
# 2. STRICT TYPE DEFINITIONS
# =============================================================================

SENTIMENT_OPTIONS = Literal["Angry", "Happy", "Sad", "Neutral", "Frustrated"]
ROUTE_OPTIONS = Literal["db_inquiry", "general_inquiry"]


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    sentiment: SENTIMENT_OPTIONS
    route_intent: ROUTE_OPTIONS
    customer_id: str
    # New Field: Stores the raw fact/data before the final empathetic polish
    tool_output: str


# --- PYDANTIC MODELS ---
class SentimentResponse(BaseModel):
    emotion: SENTIMENT_OPTIONS = Field(description="The emotional tone of the user's text.")


class RouteResponse(BaseModel):
    step: ROUTE_OPTIONS = Field(description="The next logical step.")


class SqlQuery(BaseModel):
    query: str = Field(description="The valid SQL query.")


# =============================================================================
# 3. NODE IMPLEMENTATIONS
# =============================================================================

def sentiment_agent(state: AgentState):
    print("\n--- [Node] Sentiment Agent ---")
    user_text = state["messages"][-1].content

    structured_llm = llm.with_structured_output(SentimentResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert sentiment analyzer."),
        ("human", "User Text: {text}"),
    ])

    result = prompt | structured_llm
    sentiment_data = result.invoke({"text": user_text})

    print(f"   Detected: {sentiment_data.emotion}")
    return {"sentiment": sentiment_data.emotion}


def router_node(state: AgentState):
    print("--- [Node] Router ---")
    user_text = state["messages"][-1].content

    structured_llm = llm.with_structured_output(RouteResponse)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Route to 'db_inquiry' for account data or 'general_inquiry' for info."),
        ("human", "{query}"),
    ])

    result = prompt | structured_llm
    route_data = result.invoke({"query": user_text})

    print(f"   Routing to: {route_data.step}")
    return {"route_intent": route_data.step}


def db_agent(state: AgentState):
    """Fetches Raw Data ONLY. Does not generate final response."""
    print("--- [Node] DB Agent (Worker) ---")
    user_text = state["messages"][-1].content
    cust_id = state.get("customer_id", "CUST_001")

    structured_llm = llm.with_structured_output(SqlQuery)
    system_prompt = f"Schema: {db.get_table_info()}. Write SQL for customer_id='{cust_id}'."

    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{question}")])
    query_result = (prompt | structured_llm).invoke({"question": user_text})

    try:
        db_result = db.run(query_result.query)
        raw_output = f"Database Data: {db_result}"
    except Exception as e:
        raw_output = f"Database Error: {e}"

    # Return raw data to state, NOT a message yet
    return {"tool_output": raw_output}


def sop_agent(state: AgentState):
    """Fetches Raw Context ONLY. Does not generate final response."""
    print("--- [Node] SOP Agent (Worker) ---")
    user_text = state["messages"][-1].content

    docs = sop_retriever.invoke(user_text)
    context = "\n".join([d.page_content for d in docs])

    if not context:
        context = "No specific policy found in SOP."

    # Return raw context to state
    return {"tool_output": context}


# --- NEW NODE: The Empathetic Responder ---
def response_synthesizer(state: AgentState):
    """Combines Sentiment + Raw Data -> Final Empathetic Response"""
    print("--- [Node] Empathetic Responder ---")

    user_sentiment = state["sentiment"]

    raw_data = state["tool_output"]
    print(f"Tool Output: {raw_data}")

    user_query = state["messages"][-1].content

    system_prompt = f"""You are a helpful and empathetic Telecom Assistant.

    User Sentiment: {user_sentiment}
    Data/Context Provided: {raw_data}

    Guidelines:
    1. If sentiment is 'Angry' or 'Frustrated', start with a genuine apology and reassurance.
    2. If sentiment is 'Happy', be cheerful.
    3. Use the Data Provided to answer the User Query accurately.
    4. Do not mention "SQL" or "Database". Speak naturally.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{query}")
    ])

    chain = prompt | llm
    final_response = chain.invoke({"query": user_query})

    return {"messages": [final_response]}


# =============================================================================
# 4. GRAPH CONSTRUCTION
# =============================================================================

workflow = StateGraph(AgentState)
workflow.add_node("sentiment_scanner", sentiment_agent)
workflow.add_node("router", router_node)
workflow.add_node("db_tool", db_agent)
workflow.add_node("sop_tool", sop_agent)
# Add the new node
workflow.add_node("response_synthesizer", response_synthesizer)

workflow.set_entry_point("sentiment_scanner")
workflow.add_edge("sentiment_scanner", "router")

workflow.add_conditional_edges(
    "router",
    lambda x: x["route_intent"],
    {"db_inquiry": "db_tool", "general_inquiry": "sop_tool"}
)

# Both tools now go to the Synthesizer instead of END
workflow.add_edge("db_tool", "response_synthesizer")
workflow.add_edge("sop_tool", "response_synthesizer")
workflow.add_edge("response_synthesizer", END)

app = workflow.compile()

if __name__ == "__main__":
    print(">>> Running Empathetic Bot...")

    # Test: Angry user needing data
    res = app.invoke({
        "messages": [HumanMessage(content="I am furious! Why is my internet is too slow??")],
        "customer_id": "CUST_001"
    })
    print(f"Bot: {res['messages'][-1].content}")