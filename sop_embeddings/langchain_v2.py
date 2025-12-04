
# main_langchain.py
from __future__ import annotations

import os
import json
import shutil
from typing import List, Dict, Tuple

from dotenv import load_dotenv
load_dotenv()

# LangChain v1 imports
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Prefer the split package; fall back to community wrapper if not installed
try:
    from langchain_chroma import Chroma  # modern wrapper (recommended)
except ImportError:
    from langchain_community.vectorstores import Chroma  # legacy wrapper


# ----------------------------
# Config (mirrors your original constants)
# ----------------------------
PERSIST_DIR = os.getenv("PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "telecom_sops")
INPUT_JSONL = os.getenv("INPUT_JSONL", "telecom_sop_chunks.jsonl")

# Keep in sync with your original script
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "text-embedding-004")


# ----------------------------
# Data Loading & Normalization
# ----------------------------
def load_chunks(jsonl_path: str) -> List[Dict]:
    """Load rows from the JSONL and normalize tags to List[str]."""
    rows: List[Dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            # basic schema validation
            for key in ("chunk_id", "sop_id", "text"):
                if key not in row:
                    raise ValueError(f"Missing key '{key}' in: {row}")

            # normalize tags -> List[str]
            tags = row.get("tags", [])
            if isinstance(tags, str):
                tags = [t.strip() for t in tags.split(",") if t.strip()]
            elif isinstance(tags, list):
                tags = [t.strip() for t in tags if isinstance(t, str) and t.strip()]
            else:
                tags = []

            row["tags"] = tags
            rows.append(row)

    print(f"[INFO] Loaded {len(rows)} chunks from {jsonl_path}")
    return rows


def tags_to_flags(tags: List[str]) -> Dict[str, bool]:
    """
    Convert tags to tag_<normalized>: True
    e.g., 'Outage Notice' -> {'tag_outage_notice': True}
    """
    flags: Dict[str, bool] = {}
    for t in tags:
        key = f"tag_{t.lower().replace(' ', '_').replace('-', '_')}"
        flags[key] = True
    return flags


def rows_to_documents(rows: List[Dict]) -> Tuple[List[Document], List[str]]:
    """Convert raw rows to LangChain Documents; keep ids aligned to chunk_id."""
    docs: List[Document] = []
    ids: List[str] = []

    for r in rows:
        meta = {"sop_id": r["sop_id"]}
        meta.update(tags_to_flags(r["tags"]))  # boolean flags only

        doc = Document(page_content=r["text"], metadata=meta)
        docs.append(doc)
        ids.append(r["chunk_id"])

    return docs, ids


# ----------------------------
# Vector Store (Chroma)
# ----------------------------
def get_vectorstore(embedding: GoogleGenerativeAIEmbeddings) -> Chroma:
    """
    Create/load a persistent Chroma collection via LangChain wrapper.
    """
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
        embedding_function=embedding,
    )
    return vs


def upsert_chunks(vs: Chroma, docs: List[Document], ids: List[str]) -> None:
    """
    Add documents to Chroma with your chunk_ids, then persist.
    """
    vs.add_documents(docs, ids=ids)
    vs.persist()
    print(f"[INFO] Upserted {len(docs)} chunks into '{COLLECTION_NAME}'")
    print(f"[INFO] Data stored under {PERSIST_DIR}")


# ----------------------------
# Example Queries (server-side boolean filters)
# ----------------------------
def example_query(vs: Chroma, query_text: str, filter_: Dict, k: int = 5) -> None:
    print(f"\n[QUERY] {query_text}\n filter={filter_}")
    hits = vs.similarity_search(query_text, k=k, filter=filter_)
    for i, d in enumerate(hits, start=1):
        print(f"[{i}] sop_id={d.metadata.get('sop_id')}")
        print(d.page_content[:300].replace("\n", " "))
        print("")


# ----------------------------
# Optional: RAG Chain with Gemini (LCEL)
# ----------------------------
def build_rag_chain(vs: Chroma):
    """
    Minimal RAG:
      question -> retrieve -> prompt -> Gemini -> answer
    """
    retriever = vs.as_retriever(search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a telecom support assistant. Use the SOP context faithfully. "
             "If context is insufficient, ask a clarifying question."),
            ("human",
             "Question: {question}\n\n"
             "Relevant SOP context:\n{context}\n\n"
             "Answer concisely and list any steps clearly.")
        ]
    )

    def format_docs(docs: List[Document]) -> str:
        return "\n\n".join(
            f"(sop_id={d.metadata.get('sop_id')})\n{d.page_content}" for d in docs
        )

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

    chain = (
        {"question": RunnablePassthrough(), "context": retriever | format_docs}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# ----------------------------
# Utilities
# ----------------------------
def ensure_env_or_fail() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError("GOOGLE_API_KEY not set (put it in .env or env vars).")


def rebuild_persist_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Rebuilt persist directory at: {path}")


# ----------------------------
# Main
# ----------------------------
def main(
    jsonl_path: str = INPUT_JSONL,
    persist_dir: str = PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    demo_queries: bool = True,
    run_rag_demo: bool = False,
    rebuild: bool = False,
) -> None:
    ensure_env_or_fail()

    # Embeddings (Gemini)
    embedding = GoogleGenerativeAIEmbeddings(
        model=GEMINI_EMBED_MODEL,
        task_type="retrieval_document",  # better indexing quality for docs
    )

    # Load & convert
    rows = load_chunks(jsonl_path)
    docs, ids = rows_to_documents(rows)

    # Optionally rebuild the store for a clean run
    if rebuild:
        rebuild_persist_dir(persist_dir)

    # Vector store
    vs = Chroma(
        collection_name=collection_name,
        persist_directory=persist_dir,
        embedding_function=embedding,
    )
    upsert_chunks(vs, docs, ids)

    # Example boolean filters (mirrors your original intent)
    if demo_queries:
        example_query(
            vs,
            "Is there an outage in my area?",
            filter_={"$and": [{"sop_id": "BB_OUTAGE_001"}, {"tag_outage": True}]},
            k=3,
        )
        example_query(
            vs,
            "Refund my failed recharge",
            filter_={"$and": [{"sop_id": "PREPAID_RECHARGE_002"}, {"tag_refund": True}]},
            k=3,
        )

    # Optional RAG demo:
    if run_rag_demo:
        rag = build_rag_chain(vs)
        print("\n[RAG ANSWER]\n", rag.invoke("Refund my failed recharge"))


if __name__ == "__main__":
    # You can tweak these flags or wire up argparse if you prefer.
    main(
        jsonl_path=INPUT_JSONL,
        persist_dir=PERSIST_DIR,
        collection_name=COLLECTION_NAME,
        demo_queries=True,
        run_rag_demo=False,   # set True to see an LLM answer
        rebuild=False,        # set True to wipe and rebuild ./chroma_store
    )
