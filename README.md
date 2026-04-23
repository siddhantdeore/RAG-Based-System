# 🧠 RAG + LangGraph + HITL — Dynamic PDF Knowledge Assistant

A production-grade, agentic Retrieval-Augmented Generation (RAG) system built with:

- **LangGraph** — orchestrates multi-step decision flows (retrieve → grade → answer → fallback → escalate)
- **Streamlit** — interactive UI with live PDF upload and Human-in-the-Loop (HITL) approval
- **ChromaDB** — per-session, isolated vector stores
- **OpenAI / Groq** — pluggable LLM backend
- **LangChain** — document loading, splitting, embeddings, chains

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 Live PDF Upload | Upload any PDF; it's immediately indexed into your session's vector store |
| 🔒 Session Isolation | Each Streamlit session gets its own isolated Chroma collection |
| 🧭 LangGraph Agentic Flow | Decision graph: retrieve → grade relevance → generate → fallback/escalate |
| ✋ HITL Approval | Low-confidence answers pause for human review before being sent |
| 🔁 Clarification Loop | Agent asks clarifying questions when the query is ambiguous |
| 📊 Source Citations | Every answer cites the source document and page number |
| 💬 Conversation Memory | Multi-turn chat with context window |

---

## 🗂️ Project Structure

```
rag-langgraph-hitl/
├── app.py                  # Streamlit entry point
├── config.py               # Central configuration
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── ingest.py           # PDF loading, splitting, embedding
│   ├── retriever.py        # Per-session ChromaDB retriever
│   ├── graph.py            # LangGraph state machine
│   ├── nodes.py            # Individual graph node functions
│   ├── chains.py           # LangChain LCEL chains
│   ├── prompts.py          # All prompt templates
│   ├── state.py            # GraphState TypedDict
│   └── utils.py            # Helpers, logging
└── data/
    └── sessions/           # Runtime: per-session vector stores (gitignored)
```

---

## 🚀 Quick Start

```bash
# 1. Clone / enter project
cd rag-langgraph-hitl

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or GROQ_API_KEY

# 5. Run
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes (or Groq) | OpenAI API key |
| `GROQ_API_KEY` | Optional | Groq API key (faster/cheaper) |
| `LLM_PROVIDER` | No | `openai` (default) or `groq` |
| `LLM_MODEL` | No | e.g. `gpt-4o-mini`, `llama3-70b-8192` |
| `EMBEDDING_MODEL` | No | OpenAI embedding model name |
| `CHROMA_BASE_DIR` | No | Where session vector stores live |
| `CHUNK_SIZE` | No | Text chunk size (default 800) |
| `CHUNK_OVERLAP` | No | Chunk overlap (default 150) |
| `TOP_K` | No | Number of retrieved chunks (default 5) |
| `RELEVANCE_THRESHOLD` | No | Min relevance score 0–1 (default 0.25) |
| `HITL_CONFIDENCE_THRESHOLD` | No | Below this → HITL pause (default 0.55) |

---

## 🧭 LangGraph Flow

```
START
  │
  ▼
[classify_query]
  │
  ├─ ambiguous ──► [ask_clarification] ──► END (awaits user reply)
  │
  ▼
[retrieve_documents]
  │
  ▼
[grade_relevance]
  │
  ├─ no_docs / irrelevant ──► [fallback_response] ──► END
  │
  ▼
[generate_answer]
  │
  ├─ low_confidence ──► [hitl_pause] ──► (human approves/edits) ──► END
  │
  └─ high_confidence ──► END
```

---

## 🧪 Sample Questions

After uploading a PDF (e.g., a research paper or company policy):

1. *"Summarize the key findings in the document."*
2. *"What does the document say about data privacy?"*
3. *"List all recommendations mentioned."*
4. *"Explain [technical term] as described in the uploaded paper."*
5. *"What are the limitations discussed by the authors?"*
6. *"Compare section 2 and section 4."*

---

## 📝 Session & PDF Isolation

Each browser session gets a unique `session_id` (UUID). Uploaded PDFs are vectorized into a Chroma collection named `session_<id>`. Collections are stored under `data/sessions/<session_id>/`. When the user clears their session or closes the app, their data is logically isolated and can be purged independently. No data bleeds between sessions.

---

## 📦 Tech Stack

- `streamlit` — UI
- `langchain`, `langchain-community`, `langchain-openai` — RAG pipeline
- `langgraph` — agentic graph execution
- `chromadb` — local vector store
- `pypdf` — PDF parsing
- `openai` / `groq` — LLM providers
- `python-dotenv` — env management
