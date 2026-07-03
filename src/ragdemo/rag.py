"""The RAG pipeline: build an index, retrieve context, and generate cited answers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from . import config
from .embeddings import get_embedder
from .ingest import build_chunks
from .llm import get_llm
from .vectorstore import SearchResult, VectorStore

SYSTEM_PROMPT = (
    "You are a precise assistant. Answer the user's question using ONLY the numbered context "
    "passages provided. Cite the passages you use inline like [1], [2]. If the context does not "
    "contain the answer, say you don't have enough information. Do not invent facts."
)


@dataclass
class Answer:
    question: str
    answer: str
    sources: List[SearchResult]


def _build_prompt(question: str, contexts: List[SearchResult]) -> str:
    blocks = []
    for i, c in enumerate(contexts, start=1):
        blocks.append(f"[{i}] (source: {c.source})\n{c.text}")
    context_str = "\n\n".join(blocks)
    return (
        f"Context passages:\n{context_str}\n\n"
        f"Question: {question}\n\n"
        f"Answer (cite passages as [n]):"
    )


class RAGPipeline:
    def __init__(self, store: VectorStore) -> None:
        self._store = store
        self._embedder = get_embedder()
        self._llm = get_llm()

    # ---- Index building -------------------------------------------------
    @classmethod
    def build_index(cls, docs_dir: str, index_dir: str | None = None) -> VectorStore:
        index_dir = index_dir or config.INDEX_DIR
        chunks = build_chunks(docs_dir)
        embedder = get_embedder()
        vectors = embedder.embed([c.text for c in chunks])
        store = VectorStore()
        store.add(
            vectors=vectors,
            texts=[c.text for c in chunks],
            sources=[c.source for c in chunks],
            chunk_ids=[c.chunk_id for c in chunks],
        )
        store.save(index_dir)
        return store

    @classmethod
    def from_index(cls, index_dir: str | None = None) -> "RAGPipeline":
        index_dir = index_dir or config.INDEX_DIR
        return cls(VectorStore.load(index_dir))

    # ---- Retrieval + generation ----------------------------------------
    def retrieve(self, question: str, k: int | None = None) -> List[SearchResult]:
        k = k or config.TOP_K
        q_vec = self._embedder.embed([question])[0]
        return self._store.search(q_vec, k=k)

    def answer(self, question: str, k: int | None = None) -> Answer:
        contexts = self.retrieve(question, k=k)
        prompt = _build_prompt(question, contexts)
        text = self._llm.generate(prompt, system=SYSTEM_PROMPT)
        return Answer(question=question, answer=text, sources=contexts)

    def answer_stream(self, question: str, k: int | None = None) -> Iterator[str]:
        contexts = self.retrieve(question, k=k)
        prompt = _build_prompt(question, contexts)
        yield from self._llm.stream(prompt, system=SYSTEM_PROMPT)
