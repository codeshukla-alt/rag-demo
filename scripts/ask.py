"""CLI: ask a question against the built index."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragdemo import config  # noqa: E402
from ragdemo.rag import RAGPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the RAG pipeline a question.")
    parser.add_argument("question", help="Your question (in quotes).")
    parser.add_argument("--k", type=int, default=config.TOP_K, help="Top-k passages.")
    parser.add_argument("--stream", action="store_true", help="Stream the answer.")
    args = parser.parse_args()

    rag = RAGPipeline.from_index()

    if args.stream:
        for token in rag.answer_stream(args.question, k=args.k):
            print(token, end="", flush=True)
        print()
        return

    result = rag.answer(args.question, k=args.k)
    print("\nAnswer:\n" + result.answer)
    print("\nSources:")
    for i, s in enumerate(result.sources, start=1):
        print(f"  [{i}] {s.source} (chunk {s.chunk_id}, score {s.score:.3f})")


if __name__ == "__main__":
    main()
