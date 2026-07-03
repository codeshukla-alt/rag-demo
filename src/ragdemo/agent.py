"""A minimal, provider-agnostic ReAct-style agent with tool calling.

The agent reasons step by step and emits a JSON *action* on each turn. The
runtime executes the requested tool, feeds the observation back into the
model's scratchpad, and loops until the model calls `finish`.

This demonstrates agentic behaviour - planning, tool selection, and multi-step
reasoning - without depending on any single provider's native function-calling
API, so it works identically across OpenAI and Gemini via the shared `LLM`
interface. The one tool provided here searches the RAG knowledge base, so the
agent decides *when* to retrieve versus answer directly.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from .llm import get_llm, LLM
from .rag import RAGPipeline
from .vectorstore import SearchResult

MAX_STEPS = 5

SYSTEM_PROMPT = (
    "You are a reasoning agent that solves a user's question using tools.\n"
    "On every turn you MUST reply with ONLY a single JSON object, no prose, of the form:\n"
    '{"thought": "<your reasoning>", "tool": "<tool_name>", "tool_input": "<string>"}\n\n'
    "Available tools:\n"
    "- search_knowledge_base: look up relevant passages. tool_input = the search query.\n"
    "- finish: return the final answer to the user. tool_input = the final answer.\n\n"
    "Rules:\n"
    "- Use search_knowledge_base when the question needs facts from the knowledge base.\n"
    "- You may search more than once to refine or gather more context.\n"
    "- For greetings or trivial chit-chat that need no facts, call finish directly.\n"
    "- When you call finish, ground your answer in the observed passages and cite them "
    "inline like [1], [2]. If the passages lack the answer, say you don't have enough "
    "information. Never invent facts."
)


@dataclass
class AgentStep:
    thought: str
    tool: str
    tool_input: str
    observation: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    sources: List[SearchResult] = field(default_factory=list)


def _parse_action(text: str) -> dict:
    """Best-effort extraction of the first JSON object from the model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {"thought": "", "tool": "finish", "tool_input": text.strip()}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"thought": "", "tool": "finish", "tool_input": text.strip()}


class Agent:
    """A small ReAct loop over a tool registry backed by the RAG pipeline."""

    def __init__(self, pipeline: RAGPipeline | None = None, llm: LLM | None = None) -> None:
        self._rag = pipeline or RAGPipeline.from_index()
        self._llm = llm or get_llm()
        self._collected: List[SearchResult] = []
        self.tools: Dict[str, Callable[[str], str]] = {
            "search_knowledge_base": self._tool_search,
        }

    # ---- Tools ----------------------------------------------------------
    def _tool_search(self, query: str) -> str:
        results = self._rag.retrieve(query)
        self._collected.extend(results)
        if not results:
            return "No passages found."
        return "\n".join(
            f"[{i + 1}] (source: {r.source}, score: {r.score:.3f}) {r.text[:300]}"
            for i, r in enumerate(results)
        )

    # ---- Loop -----------------------------------------------------------
    def run(self, question: str, max_steps: int = MAX_STEPS) -> AgentResult:
        scratchpad = ""
        steps: List[AgentStep] = []

        for _ in range(max_steps):
            prompt = (
                f"Question: {question}\n\n"
                f"Scratchpad (previous steps):\n{scratchpad or '(empty)'}\n\n"
                "Respond with the next JSON action."
            )
            raw = self._llm.generate(prompt, system=SYSTEM_PROMPT)
            action = _parse_action(raw)
            tool = str(action.get("tool", "finish"))
            tool_input = str(action.get("tool_input", ""))
            thought = str(action.get("thought", ""))

            if tool == "finish":
                steps.append(AgentStep(thought=thought, tool=tool, tool_input=tool_input))
                return AgentResult(
                    question=question,
                    answer=tool_input,
                    steps=steps,
                    sources=self._collected,
                )

            tool_fn = self.tools.get(tool)
            observation = (
                tool_fn(tool_input) if tool_fn else f"Unknown tool: {tool!r}."
            )
            steps.append(
                AgentStep(thought=thought, tool=tool, tool_input=tool_input, observation=observation)
            )
            scratchpad += (
                f"\nThought: {thought}\nAction: {tool}({tool_input})\nObservation: {observation}\n"
            )

        # Fell through without finishing: make a best-effort final answer.
        final = self._llm.generate(
            f"Question: {question}\n\nNotes:\n{scratchpad}\n\n"
            "Give the best final answer grounded in the notes, citing passages as [n].",
            system=SYSTEM_PROMPT,
        )
        return AgentResult(
            question=question, answer=final, steps=steps, sources=self._collected
        )
