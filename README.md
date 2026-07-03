# RAG Demo - Production-Style Retrieval-Augmented Generation with Built-in Evaluation

A compact, **production-shaped** RAG service in Python: document ingestion → chunking → embeddings →
vector search → grounded answer generation **with citations** → and an **LLM-as-judge evaluation
harness** (faithfulness, answer relevance, retrieval hit-rate).

Built to demonstrate how I ship *and measure* LLM applications - not just prototype them.

> Author: Aditya Kumar · [dataengprep.tech](https://dataengprep.tech) · [github.com/codeshukla-alt](https://github.com/codeshukla-alt)

---

## Why this project
Most RAG demos stop at "it returns an answer." This one also answers the question every serious team
asks: **"How do you know the answer is any good?"** - via an evaluation module that scores answers for
groundedness and relevance using an LLM-as-judge (the same technique behind my product, dataengprep.tech).

## Features
- **Pluggable providers:** OpenAI *or* Google Gemini for both embeddings and generation (auto-detected
  from your API keys); optional fully-offline embeddings via `sentence-transformers`.
- **Clean pipeline:** ingestion, character-overlap chunking, cosine-similarity vector store
  (swappable with FAISS / pgvector / Pinecone), top-k retrieval, cited generation.
- **Grounded answers with citations** - the model must cite `[1], [2]` from retrieved context.
- **Evaluation harness (the differentiator):** runs a golden Q&A set and scores
  **faithfulness**, **answer relevance**, and **retrieval hit** with an LLM judge.
- **Agentic tool-calling loop:** a provider-agnostic **ReAct-style agent** that reasons in
  steps and decides *when* to call the `search_knowledge_base` tool vs. answer directly - with a
  full, inspectable action/observation trace.
- **Two interfaces:** a **FastAPI** service (JSON + **streaming SSE** + `/agent`) and a **CLI**.
- **Tests** that run without any API key.

## Architecture
```
docs ─► ingest (chunk) ─► embed ─► VectorStore (cosine) 
                                        │
question ─► embed ─► retrieve top-k ────┘─► build cited prompt ─► LLM ─► answer + sources
                                                                     │
                                              eval/golden_qa.json ─► LLM-as-judge ─► scores

agent: [reason ─► search_knowledge_base ↺ ─► finish] ─► cited answer + action trace
```

## Quickstart
```bash
# 1) Create a virtualenv and install
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# 2) Configure a provider (pick ONE)
cp .env.example .env
#   then set GOOGLE_API_KEY=...   (or OPENAI_API_KEY=...)

# 3) Build the index from sample docs
python scripts/build_index.py --docs data/sample_docs

# 4) Ask a question (CLI)
python scripts/ask.py "What is RAG and why use it?"

# 5) Run the tool-calling agent (prints its reasoning + tool trace)
python scripts/agent.py "What limitations does LLM-as-judge have?"

# 6) Run the API
uvicorn ragdemo.api:app --reload --app-dir src
#   POST http://localhost:8000/ask         {"question": "..."}
#   POST http://localhost:8000/ask/stream  (Server-Sent Events)
#   POST http://localhost:8000/agent       (agent trace + answer)

# 7) Evaluate answer quality (LLM-as-judge)
python scripts/evaluate.py --golden eval/golden_qa.json
```

## Configuration (`.env`)
| Var | Default | Notes |
| --- | --- | --- |
| `LLM_PROVIDER` / `EMBEDDING_PROVIDER` | `auto` | `auto` picks OpenAI → Gemini → local |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` | – | set at least one |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | characters |
| `TOP_K` | `4` | retrieved chunks |
| `INDEX_DIR` | `index` | where the vector index is saved |

## Design decisions (talking points)
- **Cosine-similarity numpy store** by default so the repo runs anywhere with zero native deps; the
  `VectorStore` interface is deliberately small so you can drop in FAISS/pgvector/Pinecone.
- **Citations enforced in the prompt** to reduce hallucination and make answers auditable.
- **Evaluation is first-class**, not an afterthought - because in production, an un-measured LLM
  feature is an un-owned one.

## Roadmap
- Hybrid retrieval (BM25 + vector) and re-ranking
- Multi-tool agent (calculator, web search) + tool-choice evaluation
- Streaming token eval + regression gate in CI
- Swap-in FAISS backend + pgvector adapter

## License
MIT
