"""Document ingestion: load text/markdown files and split them into overlapping chunks."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from . import config


@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int


def load_documents(docs_dir: str) -> List[tuple[str, str]]:
    """Return a list of (filename, text) for supported files in a directory."""
    supported = (".md", ".txt", ".markdown")
    docs: List[tuple[str, str]] = []
    for root, _dirs, files in os.walk(docs_dir):
        for fname in sorted(files):
            if fname.lower().endswith(supported):
                path = os.path.join(root, fname)
                with open(path, "r", encoding="utf-8") as fh:
                    docs.append((fname, fh.read()))
    if not docs:
        raise FileNotFoundError(f"No .md/.txt documents found under {docs_dir!r}")
    return docs


def chunk_text(
    text: str,
    source: str,
    size: int | None = None,
    overlap: int | None = None,
) -> List[Chunk]:
    """Split text into fixed-size character chunks with overlap.

    Overlap keeps sentences that straddle a boundary intact for retrieval.
    """
    size = size or config.CHUNK_SIZE
    overlap = overlap or config.CHUNK_OVERLAP
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")

    cleaned = " ".join(text.split())  # normalize whitespace
    chunks: List[Chunk] = []
    start = 0
    idx = 0
    step = size - overlap
    while start < len(cleaned):
        piece = cleaned[start : start + size]
        chunks.append(Chunk(text=piece, source=source, chunk_id=idx))
        idx += 1
        start += step
    return chunks


def build_chunks(docs_dir: str) -> List[Chunk]:
    """Load all documents in a directory and return their combined chunks."""
    all_chunks: List[Chunk] = []
    for fname, text in load_documents(docs_dir):
        all_chunks.extend(chunk_text(text, source=fname))
    # Re-number chunk ids globally so citations are unique across the corpus.
    for global_id, chunk in enumerate(all_chunks):
        chunk.chunk_id = global_id
    return all_chunks
