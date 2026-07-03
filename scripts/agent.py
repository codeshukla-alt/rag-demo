"""CLI: run the tool-calling agent against a question."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragdemo.agent import Agent  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReAct tool-calling agent.")
    parser.add_argument("question", help="Your question (in quotes).")
    parser.add_argument("--max-steps", type=int, default=5, help="Max reasoning steps.")
    args = parser.parse_args()

    result = Agent().run(args.question, max_steps=args.max_steps)

    print("\n=== Agent trace ===")
    for i, step in enumerate(result.steps, start=1):
        print(f"\nStep {i}:")
        if step.thought:
            print(f"  thought: {step.thought}")
        print(f"  action : {step.tool}({step.tool_input})")
        if step.observation:
            print(f"  observation: {step.observation[:300]}")

    print("\n=== Final answer ===")
    print(result.answer)


if __name__ == "__main__":
    main()
