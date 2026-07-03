# Evaluating LLM Applications

Shipping an LLM feature without evaluation is risky: outputs are non-deterministic and quality drifts
as prompts, models, and data change. Evaluation makes quality measurable and regressions catchable.

## Offline vs online evaluation
- Offline evaluation runs a fixed "golden" dataset of questions with reference answers before release.
- Online evaluation measures real traffic in production (thumbs up/down, task success, latency, cost).

## LLM-as-judge
LLM-as-judge uses a strong language model to score another model's output against criteria or a
reference answer. Two common patterns are:
- Reference-based grading: compare the answer to a known-correct reference and score correctness.
- Rubric-based grading: score the answer on named criteria such as faithfulness and relevance.
Its limitations include position bias, verbosity bias, and self-preference; mitigations include
randomizing option order, using rubrics, and calibrating against human labels.

## Key metrics for RAG
- Faithfulness (groundedness): is every claim in the answer supported by the retrieved context?
- Answer relevance: does the answer actually address the user's question?
- Context precision and recall: did retrieval return the right passages, and only those?
- Retrieval hit rate: was at least one correct passage retrieved in the top-k results?

## Golden datasets
A golden dataset is a curated set of representative questions with reference answers and, ideally, the
expected source passages. Running it on every change turns quality into a regression test. Tools such
as Ragas and Promptfoo help automate these metrics, but a small custom LLM-as-judge harness is often
enough to catch the big regressions.
