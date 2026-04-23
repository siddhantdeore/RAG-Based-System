"""
config.py — Central configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _getint(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def _getfloat(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


# ─── LLM ─────────────────────────────────────────────────────────────────────
LLM_PROVIDER: str = _get("LLM_PROVIDER", "gemini")
LLM_MODEL: str = _get("LLM_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY: str = _get("GEMINI_API_KEY")

# ─── Embeddings ───────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = _get("EMBEDDING_MODEL", "models/embedding-001")

# ─── Vector Store ─────────────────────────────────────────────────────────────
CHROMA_BASE_DIR: Path = Path(_get("CHROMA_BASE_DIR", "data/sessions"))

# ─── Retrieval ────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = _getint("CHUNK_SIZE", 800)
CHUNK_OVERLAP: int = _getint("CHUNK_OVERLAP", 150)
TOP_K: int = _getint("TOP_K", 5)
RELEVANCE_THRESHOLD: float = _getfloat("RELEVANCE_THRESHOLD", 0.25)

# ─── HITL ─────────────────────────────────────────────────────────────────────
HITL_CONFIDENCE_THRESHOLD: float = _getfloat("HITL_CONFIDENCE_THRESHOLD", 0.55)


# ─── Validation ───────────────────────────────────────────────────────────────
def validate() -> None:
    if LLM_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set. "
            "Add it to your .env file. Get it free at https://aistudio.google.com/app/apikeys"
        )