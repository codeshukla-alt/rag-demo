"""CLI: run the LLM-as-judge evaluation over a golden Q&A set."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ragdemo.evaluate import main  # noqa: E402


if __name__ == "__main__":
    main()
