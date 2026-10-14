"""
AI-Powered Customer Support Assistant  (ONE code file, 100% free)
RAG (FAISS + local embeddings) + LangGraph routing + human escalation.

Install:  pip install -r requirements.txt

Uses OLLAMA (free, runs on your PC, NO API key needed).
One-time setup:
  1. Install Ollama from https://ollama.com
  2. In a terminal run:  ollama pull llama3.2
  3. Keep Ollama running, then run this file.
"""
import csv
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

# ===================== CONFIG =====================
OLLAMA_MODEL = "llama3.2"   # use "llama3.2:1b" if your PC has low RAM
# ==================================================

BASE = Path(__file__).parent
DOCS_DIR = BASE / "docs"   # put your .txt / .pdf policy files here
INDEX_DIR = BASE / "faiss_index"
ESCALATION_FILE = BASE / "escalations.csv"
CATEGORIES = ["Payment", "Delivery", "Returns", "Technical"]

# ---------------- RAG ----------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def load_chunks(paths):
    docs = []
    for p in paths:
        loader = PyPDFLoader(str(p)) if p.suffix.lower() == ".pdf" else TextLoader(str(p), encoding="utf-8")
        docs.extend(loader.load())
    return splitter.split_documents(docs)


def build_vectorstore(rebuild=False):
    if INDEX_DIR.exists() and not rebuild:
        return FAISS.load_local(str(INDEX_DIR), embeddings(), allow_dangerous_deserialization=True)
    files = [p for p in DOCS_DIR.glob("*") if p.suffix.lower() in {".txt", ".md", ".pdf"}]
    chunks = load_chunks(files)
    vs = FAISS.from_documents(chunks, embeddings())
    vs.save_local(str(INDEX_DIR))
    print(f"[RAG] Indexed {len(files)} files / {len(chunks)} chunks")
    return vs


def upload_document(vs, path_str):
    src = Path(path_str.strip().strip('"'))
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")
    if src.suffix.lower() not in {".txt", ".md", ".pdf"}:
        raise ValueError("Use .txt, .md or .pdf")
    dest = DOCS_DIR / src.name
    if src.resolve() != dest.resolve():
        shutil.copy(src, dest)
    chunks = load_chunks([dest])
    vs.add_documents(chunks)
    vs.save_local(str(INDEX_DIR))
    return len(chunks)


# ---------------- LLM ----------------
def get_llm():
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, temperature=0, format="json")


# ---------------- LangGraph ----------------
class State(TypedDict, total=False):
    question: str
    category: str
    context: List[Document]
    answer: str
    next_step: str
    can_answer: bool
    escalated: bool
    ticket_id: str


def keyword_category(q):
    q = q.lower()
    if any(w in q for w in ["return", "refund", "exchange", "replace"]):
        return "Returns"
    if any(w in q for w in ["deliver", "ship", "track", "courier", "arrive"]):
        return "Delivery"
    if any(w in q for w in ["pay", "card", "charge", "discount", "paypal"]):
        return "Payment"
    return "Technical"


def to_bool(v):
    return v if isinstance(v, bool) else str(v).strip().lower() in {"true", "yes", "1"}


# noinspection PyTypeChecker
def build_graph(vs):
    llm = get_llm()
    parser = JsonOutputParser()
    retriever = vs.as_retriever(search_kwargs={"k": 4})

    classifier = ChatPromptTemplate.from_messages([
        ("system", "Classify the e-commerce customer query into exactly one category: "
                   "Payment, Delivery, Returns, or Technical. Refunds/exchanges are Returns. "
                   "Login/app/website problems are Technical.\n"
                   "Respond ONLY with a JSON object with one key called category."),
        ("human", "{question}"),
    ]) | llm | parser

    generator = ChatPromptTemplate.from_messages([
        ("system", "You are a polite customer support assistant for ShopEase.\n"
                   "Answer ONLY using the policy context below. Never invent policies, numbers "
                   "or timeframes. If the context does not clearly cover the customer's "
                   "situation, set can_answer to false.\n\n"
                   "Respond ONLY with a JSON object with keys: can_answer (true or false), "
                   "answer (string), next_step (string, one clear action).\n\n"
                   "Query category: {category}\n\nPolicy context:\n{context}"),
        ("human", "{question}"),
    ]) | llm | parser

    # noinspection PyBroadException
    def classify(state: State):
        try:
            cat = str(classifier.invoke({"question": state["question"]})["category"]).title()
            if cat not in CATEGORIES:
                cat = keyword_category(state["question"])
        except Exception:
            cat = keyword_category(state["question"])
        return {"category": cat}

    def retrieve(state: State):
        return {"context": retriever.invoke(f"{state['category']}: {state['question']}")}

    # noinspection PyBroadException
    def generate(state: State):
        ctx = "\n\n---\n\n".join(f"[{Path(d.metadata.get('source', 'doc')).name}]\n{d.page_content}"
                                 for d in state["context"])
        try:
            r = generator.invoke({"question": state["question"],
                                  "category": state["category"], "context": ctx})
            return {"can_answer": to_bool(r.get("can_answer", False)),
                    "answer": str(r.get("answer", "")),
                    "next_step": str(r.get("next_step", "")), "escalated": False}
        except Exception:
            return {"can_answer": False, "answer": "", "next_step": "", "escalated": False}

    def escalate(state: State):
        tid = "TKT-" + uuid.uuid4().hex[:8].upper()
        new = not ESCALATION_FILE.exists()
        with open(ESCALATION_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["ticket_id", "timestamp", "category", "question"])
            w.writerow([tid, datetime.now().isoformat(timespec="seconds"),
                        state["category"], state["question"]])
        return {"escalated": True, "ticket_id": tid,
                "answer": "I couldn't find this in our policies, so I've passed your "
                          "question to a human support agent.",
                "next_step": f"Your ticket number is {tid}. An agent will contact you within 24 hours."}

    def route(state: State):
        return "answered" if state["can_answer"] and state["answer"] else "escalate"

    g = StateGraph(State)
    g.add_node("classify", classify)
    g.add_node("retrieve", retrieve)
    g.add_node("generate", generate)
    g.add_node("escalate", escalate)
    g.add_edge(START, "classify")
    g.add_edge("classify", "retrieve")
    g.add_edge("retrieve", "generate")
    g.add_conditional_edges("generate", route, {"answered": END, "escalate": "escalate"})
    g.add_edge("escalate", END)
    return g.compile()


# ---------------- Chat loop ----------------
# noinspection PyBroadException
def check_ollama():
    """Fail early with a clear message if Ollama is not ready."""
    try:
        get_llm().invoke("Reply with a JSON object with key ok and value true.")
    except Exception as e:
        raise SystemExit(
            "\nCannot reach Ollama or the model is missing.\n"
            "  1) Install Ollama: https://ollama.com and make sure it is running\n"
            f"  2) Run in a terminal:  ollama pull {OLLAMA_MODEL}\n"
            f"  3) Run this file again.\n\nDetails: {e}"
        )


def main():
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*")):
        raise SystemExit(f"No documents found. Put policy files in: {DOCS_DIR}")
    print("Checking Ollama...")
    check_ollama()
    print("Loading documents (first run downloads a small embedding model)...")
    vs = build_vectorstore()
    app = build_graph(vs)
    print("\nShopEase Support Assistant ready.")
    print("Commands: /upload <file path>   /quit\n")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text.lower() in {"/quit", "quit", "exit"}:
            break
        if text.lower().startswith("/upload"):
            try:
                print(f"Added ({upload_document(vs, text[7:])} chunks).\n")
            except Exception as e:
                print(f"Upload failed: {e}\n")
            continue
        res = app.invoke({"question": text})
        route_txt = "ESCALATED to human" if res.get("escalated") else "Answered by AI"
        print(f"\n[Category: {res['category']} | {route_txt}]")
        print(f"Assistant: {res['answer']}")
        print(f"Next step: {res['next_step']}\n")
    print("Goodbye!")


if __name__ == "__main__":
    main()