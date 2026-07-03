"""LLM-as-judge evaluation harness for the RAG pipeline.

Runs a golden Q&A set and scores each answer for:
  - faithfulness  : is every claim grounded in the retrieved context? (1-5)
  - relevance     : does the answer address the question?             (1-5)
  - retrieval_hit : did retrieval surface any expected keyword?       (0/1)

This mirrors the production-eval approach behind dataengprep.tech.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import List

from .llm import get_llm
from .rag import RAGPipeline

JUDGE_SYSTEM = (
    "You are a strict evaluation judge for a question-answering system. "
    "You return ONLY compact JSON, no prose."
)

JUDGE_TEMPLATE = """Evaluate the ANSWER to the QUESTION given the CONTEXT and a REFERENCE answer.

Score two criteria from 1 (poor) to 5 (excellent):
- faithfulness: every claim in the ANSWER is supported by the CONTEXT (no hallucination).
- relevance: the ANSWER directly addresses the QUESTION and aligns with the REFERENCE.

Return ONLY JSON of the form:
{{"faithfulness": <int>, "relevance": <int>, "reason": "<one short sentence>"}}

QUESTION:
{question}

CONTEXT:
{context}

REFERENCE:
{reference}

ANSWER:
{answer}
"""


@dataclass
class CaseResult:
    question: str
    faithfulness: int
    relevance: int
    retrieval_hit: int
    reason: str
    answer: str


def _parse_json(text: str) -> dict:
    """Best-effort extraction of the first JSON object in a string."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"faithfulness": 0, "relevance": 0, "reason": "unparseable judge output"}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"faithfulness": 0, "relevance": 0, "reason": "invalid judge JSON"}


def evaluate(golden_path: str, k: int | None = None) -> List[CaseResult]:
    with open(golden_path, "r", encoding="utf-8") as fh:
        cases = json.load(fh)

    rag = RAGPipeline.from_index()
    judge = get_llm()
    results: List[CaseResult] = []

    for case in cases:
        question = case["question"]
        reference = case.get("reference", "")
        keywords = [k.lower() for k in case.get("expected_keywords", [])]

        result = rag.answer(question, k=k)
        context = "\n\n".join(f"[{i+1}] {s.text}" for i, s in enumerate(result.sources))

        # Retrieval hit: did any expected keyword appear in retrieved context?
        ctx_lower = context.lower()
        retrieval_hit = int(any(kw in ctx_lower for kw in keywords)) if keywords else 1

        verdict = _parse_json(
            judge.generate(
                JUDGE_TEMPLATE.format(
                    question=question,
                    context=context,
                    reference=reference,
                    answer=result.answer,
                ),
                system=JUDGE_SYSTEM,
            )
        )
        results.append(
            CaseResult(
                question=question,
                faithfulness=int(verdict.get("faithfulness", 0)),
                relevance=int(verdict.get("relevance", 0)),
                retrieval_hit=retrieval_hit,
                reason=str(verdict.get("reason", "")),
                answer=result.answer,
            )
        )
    return results


def _summary(results: List[CaseResult]) -> dict:
    n = len(results) or 1
    return {
        "cases": len(results),
        "avg_faithfulness": round(sum(r.faithfulness for r in results) / n, 2),
        "avg_relevance": round(sum(r.relevance for r in results) / n, 2),
        "retrieval_hit_rate": round(sum(r.retrieval_hit for r in results) / n, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline (LLM-as-judge).")
    parser.add_argument("--golden", default="eval/golden_qa.json", help="Path to golden Q&A JSON.")
    parser.add_argument("--k", type=int, default=None, help="Top-k passages to retrieve.")
    parser.add_argument("--json", action="store_true", help="Print full per-case JSON.")
    args = parser.parse_args()

    results = evaluate(args.golden, k=args.k)

    print("\n=== Per-case scores ===")
    for r in results:
        print(
            f"- faith={r.faithfulness} rel={r.relevance} hit={r.retrieval_hit} "
            f"| {r.question}\n    {r.reason}"
        )

    print("\n=== Summary ===")
    summary = _summary(results)
    for key, val in summary.items():
        print(f"{key}: {val}")

    if args.json:
        print("\n=== Raw ===")
        print(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
