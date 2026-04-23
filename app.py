"""
app.py — Streamlit entry point for the RAG + LangGraph + HITL assistant.

Run with:
    streamlit run app.py

UI Layout:
  ┌─────────────────────────────────────────────────────────┐
  │  Sidebar: session info, PDF upload, indexed files list  │
  ├─────────────────────────────────────────────────────────┤
  │  Main: chat history, message input, HITL approval panel │
  └─────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import time
from typing import List

import streamlit as st

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAG + LangGraph Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Imports after page config ────────────────────────────────────────────────
import config
from src.graph import run_graph
from src.ingest import ingest_pdf, list_ingested_sources
from src.retriever import purge_session
from src.state import GraphState
from src.utils import get_logger, new_session_id

logger = get_logger(__name__)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.04);
    border-right: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(12px);
}

/* ── Header ── */
.main-header {
    text-align: center;
    padding: 1.5rem 0 0.5rem 0;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.main-header p {
    color: rgba(255,255,255,0.45);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Chat bubbles ── */
.chat-row {
    display: flex;
    margin: 0.6rem 0;
    gap: 0.75rem;
    animation: fadeSlideIn 0.3s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
.chat-row.user    { flex-direction: row-reverse; }
.chat-row.user    .bubble { background: linear-gradient(135deg, #7c3aed, #4f46e5); color: #fff; border-radius: 18px 18px 4px 18px; }
.chat-row.assistant .bubble { background: rgba(255,255,255,0.07); color: #e2e8f0; border-radius: 18px 18px 18px 4px; border: 1px solid rgba(255,255,255,0.1); }

.bubble {
    max-width: 72%;
    padding: 0.85rem 1.1rem;
    font-size: 0.93rem;
    line-height: 1.6;
    word-wrap: break-word;
}

.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.avatar.user      { background: linear-gradient(135deg, #7c3aed, #4f46e5); }
.avatar.assistant { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); }

/* ── Source citations ── */
.source-card {
    background: rgba(96,165,250,0.08);
    border: 1px solid rgba(96,165,250,0.25);
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    margin-top: 0.4rem;
    font-size: 0.78rem;
    color: #93c5fd;
}
.source-card strong { color: #60a5fa; }

/* ── HITL panel ── */
.hitl-panel {
    background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(234,88,12,0.08));
    border: 1px solid rgba(245,158,11,0.4);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin: 1rem 0;
    animation: pulseGlow 2s ease-in-out infinite;
}
@keyframes pulseGlow {
    0%,100% { box-shadow: 0 0 0px rgba(245,158,11,0.3); }
    50%      { box-shadow: 0 0 18px rgba(245,158,11,0.4); }
}
.hitl-panel h4 { color: #fbbf24; margin: 0 0 0.4rem 0; font-size: 1rem; }
.hitl-panel p  { color: rgba(255,255,255,0.65); font-size: 0.85rem; margin: 0 0 0.8rem 0; }

/* ── Sidebar cards ── */
.sidebar-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.83rem;
    color: rgba(255,255,255,0.7);
}
.sidebar-card .label { color: rgba(255,255,255,0.4); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; }

/* ── Confidence badge ── */
.conf-badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 99px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 0.4rem;
}
.conf-high   { background: rgba(52,211,153,0.15); color: #34d399; border: 1px solid rgba(52,211,153,0.3); }
.conf-medium { background: rgba(251,191,36,0.15); color: #fbbf24; border: 1px solid rgba(251,191,36,0.3); }
.conf-low    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Node trace ── */
.trace-pill {
    display: inline-block;
    background: rgba(167,139,250,0.1);
    border: 1px solid rgba(167,139,250,0.25);
    border-radius: 99px;
    padding: 0.12rem 0.5rem;
    font-size: 0.7rem;
    color: #a78bfa;
    margin: 0.15rem;
}

/* ── Streamlit overrides ── */
[data-testid="stChatInput"] > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
}
.stButton > button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stFileUploader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── Session State Initialisation ─────────────────────────────────────────────

def _init_session() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = new_session_id()
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []          # [{role, content, meta}]
    if "hitl_pending" not in st.session_state:
        st.session_state.hitl_pending = None        # GraphState awaiting review
    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []
    if "upload_status" not in st.session_state:
        st.session_state.upload_status = {}         # {filename: "ok"|"error"}


_init_session()
session_id: str = st.session_state.session_id


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 RAG Assistant")
    st.markdown("---")

    # Session Info
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="label">Session ID</div>
            <code style="color:#a78bfa;font-size:0.8rem">{session_id[:12]}…</code>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # LLM Info
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="label">LLM</div>
            <span>{config.LLM_PROVIDER.upper()} · {config.LLM_MODEL}</span>
        </div>
        <div class="sidebar-card">
            <div class="label">Embedding</div>
            <span>{config.EMBEDDING_MODEL}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("### 📄 Upload PDFs")

    uploaded_files = st.file_uploader(
        label="Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader",
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uf in uploaded_files:
            if uf.name not in st.session_state.upload_status:
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        n_chunks = ingest_pdf(
                            file_bytes=uf.read(),
                            filename=uf.name,
                            session_id=session_id,
                        )
                        st.session_state.upload_status[uf.name] = f"✅ {n_chunks} chunks"
                        st.session_state.indexed_files = list_ingested_sources(session_id)
                    except Exception as e:
                        st.session_state.upload_status[uf.name] = f"❌ {e}"
                        logger.exception("Ingest failed for %s", uf.name)

    # Show indexed files
    if st.session_state.indexed_files:
        st.markdown("### 🗂️ Indexed Documents")
        for fname in st.session_state.indexed_files:
            status = st.session_state.upload_status.get(fname, "✅")
            st.markdown(
                f'<div class="sidebar-card">📎 <strong>{fname}</strong>'
                f'<br><span style="color:#34d399;font-size:0.75rem">{status}</span></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.hitl_pending = None
            st.rerun()
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            purge_session(session_id)
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    st.markdown(
        "<br><div style='color:rgba(255,255,255,0.25);font-size:0.72rem;text-align:center'>"
        "RAG · LangGraph · HITL · ChromaDB</div>",
        unsafe_allow_html=True,
    )


# ─── Main Area Header ─────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="main-header">
        <h1>🧠 Dynamic PDF Knowledge Assistant</h1>
        <p>Upload PDFs → Ask questions → LangGraph orchestrates retrieval, grading, and Human-in-the-Loop review</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Warn if no docs uploaded
if not st.session_state.indexed_files:
    st.info(
        "👈  **Upload at least one PDF** in the sidebar to get started. "
        "Your documents are isolated to this session only.",
        icon="📄",
    )


# ─── Chat History Renderer ────────────────────────────────────────────────────

def _confidence_badge(conf: float) -> str:
    if conf >= 0.7:
        return f'<span class="conf-badge conf-high">⬤ {conf:.0%}</span>'
    elif conf >= 0.45:
        return f'<span class="conf-badge conf-medium">⬤ {conf:.0%}</span>'
    else:
        return f'<span class="conf-badge conf-low">⬤ {conf:.0%}</span>'


def _render_chat() -> None:
    for msg in st.session_state.chat_history:
        role = msg["role"]
        content = msg["content"]
        meta = msg.get("meta", {})

        if role == "user":
            st.markdown(
                f"""
                <div class="chat-row user">
                    <div class="avatar user">👤</div>
                    <div class="bubble">{content}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            conf = meta.get("confidence", None)
            badge = _confidence_badge(conf) if conf is not None else ""
            sources = meta.get("sources", [])
            trace = meta.get("node_trace", [])

            sources_html = ""
            for s in sources:
                sources_html += (
                    f'<div class="source-card">'
                    f'<strong>📎 {s["file"]}</strong> · Page {s["page"]}<br>'
                    f'<em>{s["snippet"]}</em></div>'
                )

            trace_html = ""
            if trace:
                pills = "".join(f'<span class="trace-pill">{n}</span>' for n in trace)
                trace_html = f'<div style="margin-top:0.5rem">{pills}</div>'

            st.markdown(
                f"""
                <div class="chat-row assistant">
                    <div class="avatar assistant">🤖</div>
                    <div class="bubble">
                        {content}{badge}
                        {sources_html}
                        {trace_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


_render_chat()


# ─── HITL Approval Panel ──────────────────────────────────────────────────────

if st.session_state.hitl_pending is not None:
    pending: GraphState = st.session_state.hitl_pending

    st.markdown(
        f"""
        <div class="hitl-panel">
            <h4>✋ Human Review Required</h4>
            <p>
                The assistant generated an answer with low confidence
                <strong>({pending.get('confidence', 0):.0%})</strong>.
                Please review, edit if needed, then approve or reject.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    edited = st.text_area(
        "📝 Review / Edit the Answer",
        value=pending.get("answer", ""),
        height=180,
        key="hitl_edit_area",
    )

    col_a, col_r = st.columns([1, 1])
    with col_a:
        if st.button("✅ Approve & Send", use_container_width=True, type="primary"):
            final_answer = edited.strip() or pending.get("answer", "")
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": f"*(Human-reviewed)*\n\n{final_answer}",
                    "meta": {
                        "confidence": pending.get("confidence", 0),
                        "sources": pending.get("sources", []),
                        "node_trace": pending.get("node_trace", []) + ["hitl_approved"],
                    },
                }
            )
            st.session_state.hitl_pending = None
            st.rerun()

    with col_r:
        if st.button("❌ Reject & Discard", use_container_width=True):
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": (
                        "⚠️ The generated answer was reviewed and discarded. "
                        "Please rephrase your question or upload more relevant documents."
                    ),
                    "meta": {"node_trace": pending.get("node_trace", []) + ["hitl_rejected"]},
                }
            )
            st.session_state.hitl_pending = None
            st.rerun()


# ─── Chat Input ───────────────────────────────────────────────────────────────

user_input = st.chat_input(
    placeholder="Ask anything about your uploaded documents…",
    disabled=(st.session_state.hitl_pending is not None),
)

if user_input and user_input.strip():
    query = user_input.strip()

    # Add user message to history immediately
    st.session_state.chat_history.append({"role": "user", "content": query})

    # Build LangChain-format chat history (exclude current message)
    lc_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_history[:-1]
    ]

    # Build initial graph state
    initial_state: GraphState = {
        "session_id": session_id,
        "user_query": query,
        "chat_history": lc_history,
        "node_trace": [],
    }

    with st.spinner("🔍 Thinking…"):
        try:
            config.validate()
            result = run_graph(initial_state)
        except EnvironmentError as env_err:
            result = {
                "answer": f"⚠️ Configuration error: {env_err}",
                "sources": [],
                "confidence": 0.0,
                "needs_hitl": False,
                "node_trace": ["error"],
            }
        except Exception as ex:
            logger.exception("Graph execution error")
            result = {
                "answer": f"⚠️ Unexpected error: {ex}",
                "sources": [],
                "confidence": 0.0,
                "needs_hitl": False,
                "node_trace": ["error"],
            }

    if result.get("needs_hitl"):
        # Park the state — render HITL panel on next rerun
        st.session_state.hitl_pending = result
    else:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.get("answer", "No answer generated."),
                "meta": {
                    "confidence": result.get("confidence"),
                    "sources": result.get("sources", []),
                    "node_trace": result.get("node_trace", []),
                },
            }
        )

    st.rerun()
