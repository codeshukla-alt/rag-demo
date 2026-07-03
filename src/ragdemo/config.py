"""Central configuration, loaded from environment / .env.

Provider selection is "auto" by default: prefer OpenAI if a key is present,
then Gemini, then a local (offline) embedding model.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


OPENAI_API_KEY = _get("OPENAI_API_KEY")
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")

OPENAI_CHAT_MODEL = _get("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = _get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
GEMINI_CHAT_MODEL = _get("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
GEMINI_EMBED_MODEL = _get("GEMINI_EMBED_MODEL", "gemini-embedding-001")
LOCAL_EMBED_MODEL = _get("LOCAL_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

CHUNK_SIZE = int(_get("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(_get("CHUNK_OVERLAP", "120"))
TOP_K = int(_get("TOP_K", "4"))
INDEX_DIR = _get("INDEX_DIR", "index")

_LLM_PROVIDER = _get("LLM_PROVIDER", "auto").lower()
_EMBEDDING_PROVIDER = _get("EMBEDDING_PROVIDER", "auto").lower()


def resolve_llm_provider() -> str:
    """Return the concrete LLM provider name: 'openai' or 'gemini'."""
    if _LLM_PROVIDER in ("openai", "gemini"):
        return _LLM_PROVIDER
    if OPENAI_API_KEY:
        return "openai"
    if GOOGLE_API_KEY:
        return "gemini"
    raise RuntimeError(
        "No LLM provider configured. Set OPENAI_API_KEY or GOOGLE_API_KEY in .env."
    )


def resolve_embedding_provider() -> str:
    """Return the concrete embedding provider: 'openai', 'gemini' or 'local'."""
    if _EMBEDDING_PROVIDER in ("openai", "gemini", "local"):
        return _EMBEDDING_PROVIDER
    if OPENAI_API_KEY:
        return "openai"
    if GOOGLE_API_KEY:
        return "gemini"
    return "local"
