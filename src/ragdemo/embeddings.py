"""Embedding providers with a common interface.

Provider libraries are imported lazily so the package imports cleanly even when
only one provider is installed/configured.
"""
from __future__ import annotations

from typing import List, Protocol

from . import config


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text."""
        ...


class OpenAIEmbedder:
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_EMBED_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in resp.data]


class GeminiEmbedder:
    def __init__(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=config.GOOGLE_API_KEY)
        self._genai = genai
        self._model = config.GEMINI_EMBED_MODEL

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            res = self._genai.embed_content(model=self._model, content=text)
            vectors.append(res["embedding"])
        return vectors


class LocalEmbedder:
    """Offline embeddings via sentence-transformers (no API key required)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.LOCAL_EMBED_MODEL)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, normalize_embeddings=False).tolist()


def get_embedder(provider: str | None = None) -> Embedder:
    """Factory that returns the configured embedder."""
    provider = provider or config.resolve_embedding_provider()
    if provider == "openai":
        return OpenAIEmbedder()
    if provider == "gemini":
        return GeminiEmbedder()
    if provider == "local":
        return LocalEmbedder()
    raise ValueError(f"Unknown embedding provider: {provider!r}")
