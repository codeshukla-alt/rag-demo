"""Tests that run WITHOUT any API key (pure logic: chunking + vector search)."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragdemo.ingest import chunk_text  # noqa: E402
from ragdemo.vectorstore import VectorStore  # noqa: E402


def test_chunking_overlap():
    text = "word " * 500  # ~2500 chars
    chunks = chunk_text(text, source="t.md", size=800, overlap=120)
    assert len(chunks) >= 3
    # chunk ids are sequential
    assert [c.chunk_id for c in chunks] == list(range(len(chunks)))
    # each chunk respects the size bound
    assert all(len(c.text) <= 800 for c in chunks)


def test_vector_store_cosine_search():
    store = VectorStore()
    store.add(
        vectors=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        texts=["east", "north", "east-ish"],
        sources=["a", "b", "c"],
        chunk_ids=[0, 1, 2],
    )
    results = store.search([1.0, 0.0], k=2)
    # closest to the query [1,0] should be "east", then "east-ish"
    assert results[0].text == "east"
    assert results[1].text == "east-ish"
    assert results[0].score >= results[1].score


def test_save_and_load(tmp_path):
    store = VectorStore()
    store.add([[1.0, 0.0]], ["only"], ["s"], [0])
    store.save(str(tmp_path))
    loaded = VectorStore.load(str(tmp_path))
    assert len(loaded) == 1
    assert loaded.search([1.0, 0.0], k=1)[0].text == "only"
