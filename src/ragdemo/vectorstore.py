"""A minimal in-memory vector store using cosine similarity.

The interface is intentionally small (add / search / save / load) so it can be
swapped for FAISS, pgvector, or Pinecone without touching the RAG pipeline.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class SearchResult:
    text: str
    source: str
    chunk_id: int
    score: float


class VectorStore:
    def __init__(self) -> None:
        self._vectors: np.ndarray | None = None  # shape (n, d), L2-normalized
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._chunk_ids: List[int] = []

    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return mat / norms

    def add(
        self,
        vectors: List[List[float]],
        texts: List[str],
        sources: List[str],
        chunk_ids: List[int],
    ) -> None:
        mat = self._normalize(np.asarray(vectors, dtype=np.float32))
        self._vectors = mat if self._vectors is None else np.vstack([self._vectors, mat])
        self._texts.extend(texts)
        self._sources.extend(sources)
        self._chunk_ids.extend(chunk_ids)

    def search(self, query_vector: List[float], k: int = 4) -> List[SearchResult]:
        if self._vectors is None or len(self._texts) == 0:
            raise RuntimeError("Vector store is empty. Build the index first.")
        q = self._normalize(np.asarray([query_vector], dtype=np.float32))[0]
        scores = self._vectors @ q  # cosine similarity (both normalized)
        top = np.argsort(scores)[::-1][:k]
        return [
            SearchResult(
                text=self._texts[i],
                source=self._sources[i],
                chunk_id=self._chunk_ids[i],
                score=float(scores[i]),
            )
            for i in top
        ]

    def save(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        np.savez_compressed(os.path.join(index_dir, "vectors.npz"), vectors=self._vectors)
        meta = {
            "texts": self._texts,
            "sources": self._sources,
            "chunk_ids": self._chunk_ids,
        }
        with open(os.path.join(index_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str) -> "VectorStore":
        store = cls()
        data = np.load(os.path.join(index_dir, "vectors.npz"))
        store._vectors = data["vectors"]
        with open(os.path.join(index_dir, "meta.json"), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        store._texts = meta["texts"]
        store._sources = meta["sources"]
        store._chunk_ids = meta["chunk_ids"]
        return store

    def __len__(self) -> int:
        return len(self._texts)
