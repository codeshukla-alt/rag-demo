"""FastAPI service exposing the RAG pipeline (JSON + streaming SSE)."""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import config
from .agent import Agent
from .rag import RAGPipeline
from .vectorstore import VectorStore

app = FastAPI(title="RAG Demo", version="0.1.0")

_pipeline: RAGPipeline | None = None


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline.from_index()
        except FileNotFoundError as exc:  # index not built yet
            raise HTTPException(
                status_code=503,
                detail="Index not found. Run: python scripts/build_index.py --docs data/sample_docs",
            ) from exc
    return _pipeline


class AskRequest(BaseModel):
    question: str
    k: int | None = None


class Source(BaseModel):
    source: str
    chunk_id: int
    score: float
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_built": VectorStore is not None,
        "llm_provider": config.resolve_llm_provider(),
        "embedding_provider": config.resolve_embedding_provider(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = _get_pipeline().answer(req.question, k=req.k)
    return AskResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            Source(
                source=s.source,
                chunk_id=s.chunk_id,
                score=round(s.score, 4),
                preview=s.text[:200],
            )
            for s in result.sources
        ],
    )


class AgentStepModel(BaseModel):
    thought: str
    tool: str
    tool_input: str
    observation: str


class AgentResponse(BaseModel):
    question: str
    answer: str
    steps: list[AgentStepModel]
    sources: list[Source]


@app.post("/agent", response_model=AgentResponse)
def agent(req: AskRequest) -> AgentResponse:
    """Run the ReAct tool-calling agent, which decides when to search the KB."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    _get_pipeline()  # ensure the index exists (raises 503 otherwise)
    result = Agent().run(req.question)
    return AgentResponse(
        question=result.question,
        answer=result.answer,
        steps=[
            AgentStepModel(
                thought=s.thought,
                tool=s.tool,
                tool_input=s.tool_input,
                observation=s.observation,
            )
            for s in result.steps
        ],
        sources=[
            Source(
                source=s.source,
                chunk_id=s.chunk_id,
                score=round(s.score, 4),
                preview=s.text[:200],
            )
            for s in result.sources
        ],
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest) -> StreamingResponse:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    pipeline = _get_pipeline()

    def event_gen():
        for token in pipeline.answer_stream(req.question, k=req.k):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
