"""LLM providers with a common interface (generate + streaming).

Provider libraries are imported lazily.
"""
from __future__ import annotations

from typing import Iterator, Protocol

from . import config


class LLM(Protocol):
    def generate(self, prompt: str, system: str = "") -> str: ...
    def stream(self, prompt: str, system: str = "") -> Iterator[str]: ...


class OpenAILLM:
    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._model = config.OPENAI_CHAT_MODEL

    def _messages(self, prompt: str, system: str):
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        return msgs

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(prompt, system),
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=self._messages(prompt, system),
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class GeminiLLM:
    def __init__(self) -> None:
        import google.generativeai as genai

        genai.configure(api_key=config.GOOGLE_API_KEY)
        self._genai = genai
        self._model_name = config.GEMINI_CHAT_MODEL

    def _model(self, system: str):
        return self._genai.GenerativeModel(
            self._model_name,
            system_instruction=system or None,
            generation_config={"temperature": 0.2},
        )

    def generate(self, prompt: str, system: str = "") -> str:
        resp = self._model(system).generate_content(prompt)
        return (resp.text or "").strip()

    def stream(self, prompt: str, system: str = "") -> Iterator[str]:
        for chunk in self._model(system).generate_content(prompt, stream=True):
            if getattr(chunk, "text", None):
                yield chunk.text


def get_llm(provider: str | None = None) -> LLM:
    """Factory that returns the configured LLM."""
    provider = provider or config.resolve_llm_provider()
    if provider == "openai":
        return OpenAILLM()
    if provider == "gemini":
        return GeminiLLM()
    raise ValueError(f"Unknown LLM provider: {provider!r}")
