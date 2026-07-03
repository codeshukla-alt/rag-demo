"""CLI: build the vector index from a documents directory."""
from __future__ import annotations

import argparse
import os
import sys

# Make `ragdemo` importable when run from the repo root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragdemo import config  # noqa: E402
from ragdemo.rag import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG vector index.")
    parser.add_argument("--docs", default="data/sample_docs", help="Directory of .md/.txt docs.")
    parser.add_argument("--index", default=config.INDEX_DIR, help="Output index directory.")
    args = parser.parse_args()

    print(f"Embedding provider: {config.resolve_embedding_provider()}")
    print(f"Building index from {args.docs!r} -> {args.index!r} ...")
    store = RAGPipeline.build_index(args.docs, index_dir=args.index)
    print(f"Done. Indexed {len(store)} chunks.")


if __name__ == "__main__":
    main()
