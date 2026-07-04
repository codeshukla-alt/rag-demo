# Learn This Project (Beginner's Guide)

This is your personal study guide for the `rag-demo` project. Read it slowly. By the end you'll
be able to explain every part in your own words and answer interviewer questions confidently.

> **Golden rule for interviews:** Never claim to be an expert. Say *"I built this to learn
> production RAG, evaluation, and agents hands-on."* Honest builders impress interviewers.

---

## 1. The big picture (in one paragraph)

This project is an **AI question-answering system that answers from your own documents, with
proof it's telling the truth.** You give it some documents. It reads them, remembers them in a
smart way, and when you ask a question it (a) finds the most relevant parts, (b) writes an answer
that **cites** those parts, (c) can act as an **agent** that decides when to look things up, and
(d) has an **evaluator** that automatically scores how good the answers are.

**Analogy:** It's an **open-book exam taker**. Instead of answering from memory (which leads to
made-up "facts"), it opens the book, finds the right page, and answers with a page reference.

---

## 2. The 7 core concepts (learn these first)

| # | Concept | Plain-English meaning |
|---|---------|-----------------------|
| 1 | **RAG** (Retrieval-Augmented Generation) | Look up relevant text first, THEN let the AI answer using it. Open-book, not from memory. |
| 2 | **Embedding** | Turning text into a list of numbers that captures *meaning*. Similar meaning → similar numbers. |
| 3 | **Chunking** | Splitting documents into small pieces so we fetch just the relevant paragraph, not a whole file. |
| 4 | **Vector store** | A database of those number-lists that can quickly find the closest matches. |
| 5 | **Cosine similarity** | A math way to measure "how close in meaning" two number-lists are (1.0 = identical, 0 = unrelated). |
| 6 | **LLM-as-judge** | Using a second AI call to *grade* an answer (is it faithful? is it relevant?). |
| 7 | **Agent (ReAct)** | An AI that loops: **think → use a tool → look at the result → repeat → finish.** It decides *when* to use tools. |

If you can explain these 7 rows, you understand 80% of the project.

---

## 3. The folder map (what each file does, one line each)

```
rag-demo/
├─ data/sample_docs/        # the "book" - example documents the AI answers from
├─ eval/golden_qa.json      # test questions + known-correct answers (for grading)
├─ src/ragdemo/
│   ├─ config.py            # reads settings from .env (API keys, model names, sizes)
│   ├─ ingest.py            # loads documents and splits them into chunks
│   ├─ embeddings.py        # turns text into number-vectors (OpenAI / Gemini / offline)
│   ├─ vectorstore.py       # stores vectors and finds the closest matches (cosine)
│   ├─ rag.py               # the main pipeline: retrieve + generate cited answer
│   ├─ llm.py               # talks to the AI model (OpenAI / Gemini) to write text
│   ├─ agent.py             # the tool-calling agent (think → search → finish)
│   ├─ evaluate.py          # LLM-as-judge: scores answers on a golden set
│   └─ api.py               # FastAPI web service exposing /ask, /agent, /ask/stream
├─ scripts/                 # command-line entry points (build_index, ask, agent, evaluate)
└─ tests/                   # tests that run without any API key
```

---

## 4. File-by-file walkthrough (plain English)

### `config.py` - the settings
Reads your `.env` file (API keys, which model to use, chunk size, how many results to fetch).
The clever bit: `resolve_llm_provider()` auto-picks a provider — if you have an OpenAI key it uses
OpenAI, else Gemini, else a fully-offline model. **Say:** *"Configuration is centralized and
provider-agnostic, so switching from OpenAI to Gemini is a one-line change in .env."*

### `ingest.py` - reading and splitting the documents
- `load_documents()` reads every `.md`/`.txt` file in a folder.
- `chunk_text()` splits text into fixed-size pieces (default 800 characters) with a 120-character
  **overlap**. Overlap matters: if an important sentence sits on the boundary between two chunks,
  the overlap keeps it whole in at least one chunk so retrieval doesn't miss it.
- **Say:** *"I chunk with overlap so sentences that straddle a boundary aren't lost."*

### `embeddings.py` - turning text into numbers
Three interchangeable providers (OpenAI, Gemini, local sentence-transformers), all with the same
`embed(texts)` method. This is the "meaning → numbers" step.
**Say:** *"The embedder is behind a common interface, so the pipeline doesn't care which provider
produced the vectors."*

### `vectorstore.py` - the search engine
- Stores all the chunk vectors in a NumPy array.
- `_normalize()` scales every vector to length 1, which makes **cosine similarity** just a dot
  product (fast).
- `search()` compares your question's vector against all chunk vectors and returns the top-k
  closest ones with a score.
- `save()`/`load()` write the index to disk so you don't re-embed every time.
- **Say:** *"I used a simple NumPy cosine store so it runs anywhere with zero native dependencies.
  The interface is tiny (add/search/save/load) on purpose, so I can swap in FAISS or pgvector
  without changing the pipeline."*

### `rag.py` - the heart (retrieve + generate)
- `build_index()` = documents → chunks → embeddings → save to vector store.
- `retrieve()` = embed the question → find top-k closest chunks.
- `answer()` = retrieve, build a prompt that includes the chunks, ask the LLM to answer **and cite
  `[1] [2]`**, return the answer + sources.
- The `SYSTEM_PROMPT` is important: it tells the model *"answer ONLY from the context, cite it, and
  if the answer isn't there, say you don't know."* This is how we fight hallucination.
- **Say:** *"The system prompt forces grounding and citations; if the context lacks the answer, the
  model must admit it rather than invent."*

### `llm.py` - talking to the AI
Wraps OpenAI and Gemini behind one interface with two methods: `generate()` (full answer) and
`stream()` (token-by-token). Temperature is low (0.2) so answers are focused, not random.
**Say:** *"One LLM interface, two providers, plus streaming support."*

### `agent.py` - the agentic part (your "Agentic AI" proof)
A **ReAct-style** loop. Each turn the model returns a small JSON action:
```json
{"thought": "...", "tool": "search_knowledge_base", "tool_input": "..."}
```
- If the tool is `search_knowledge_base`, the agent runs a retrieval and feeds the results back.
- If the tool is `finish`, the loop ends with the final cited answer.
- It runs up to `MAX_STEPS` times, so it can search more than once to gather enough context.
- **Why it matters:** unlike the fixed pipeline (which always retrieves), the agent **decides when**
  to use a tool — that's the core idea of agentic AI.
- **Say:** *"It's a provider-agnostic ReAct loop — the model reasons, picks a tool, observes the
  result, and repeats until it can finish. I kept it model-agnostic instead of using one vendor's
  native function-calling so it works across OpenAI and Gemini."*

### `evaluate.py` - the differentiator (LLM-as-judge)
- Reads `eval/golden_qa.json` (questions + reference answers + expected keywords).
- For each question: gets the RAG answer, then asks a **judge LLM** to score **faithfulness**
  (no hallucination) and **relevance** (answers the question) from 1-5, plus a **retrieval hit**
  (did retrieval surface an expected keyword?).
- Prints per-case scores and averages.
- **Say:** *"Most demos stop at 'it returns an answer.' This measures answer quality automatically.
  In production it becomes a regression gate — if a change lowers the scores, I catch it before
  shipping. This mirrors the LLM-as-judge approach behind my product, dataengprep.tech."*

### `api.py` - the web service
FastAPI app exposing:
- `POST /ask` → JSON answer + sources
- `POST /ask/stream` → streams the answer token-by-token (Server-Sent Events)
- `POST /agent` → runs the agent and returns its step-by-step trace + answer
- `GET /health` → status + which providers are active
FastAPI auto-generates the interactive docs at `/docs`.
**Say:** *"The same core logic is exposed as a REST API with auto-generated Swagger docs — that's
how a real front-end or another service would consume it."*

---

## 5. How the pieces connect (the flows)

**Building the index (once):**
```
documents → ingest (chunk) → embeddings → vector store → saved to disk
```

**Answering a question (/ask):**
```
question → embed → vector store finds top-k chunks → build cited prompt → LLM → answer + sources
```

**The agent (/agent):**
```
question → [ think → search_knowledge_base → observe ] (repeat) → finish → cited answer + trace
```

**Evaluation:**
```
golden questions → run RAG → judge LLM scores each → averages (faithfulness / relevance / hit-rate)
```

---

## 6. Run it yourself (do this until it feels familiar)

```powershell
cd "c:\Agentic AI\rag-demo"
.venv\Scripts\activate

# 1) Ask a question (see retrieval + citations)
python scripts/ask.py "What is RAG and why use it?"

# 2) Run the agent (watch it think and use a tool)
python scripts/agent.py "What limitations does LLM-as-judge have?"

# 3) Grade answer quality (your winning moment)
python scripts/evaluate.py --golden eval/golden_qa.json

# 4) Run the API, then open http://localhost:8000/docs
uvicorn ragdemo.api:app --reload --app-dir src

# 5) Run the tests (no API key needed)
pytest -q
```
For each, ask yourself: *which of the 7 concepts is this showing?*

---

## 7. Interviewer Q&A (practice out loud)

- **"What is RAG?"** → *"Retrieval-Augmented Generation. Instead of the model answering from memory,
  I retrieve relevant document chunks and give them to the model as context, so answers are grounded
  and can cite sources."*
- **"What's an embedding?"** → *"A numeric vector representing meaning. Similar text gets similar
  vectors, so I can find relevant chunks by measuring vector closeness."*
- **"How do you prevent hallucination?"** → *"Two layers: the prompt forces the model to answer only
  from retrieved context and cite it, and the evaluation harness measures faithfulness so I can
  detect drift."*
- **"Why chunk with overlap?"** → *"So a sentence split across a boundary stays intact in at least
  one chunk and isn't lost during retrieval."*
- **"How would you scale to millions of docs?"** → *"Swap my NumPy cosine store for FAISS or
  pgvector — the vector-store interface is small on purpose, so the pipeline doesn't change."*
- **"Why an agent instead of a fixed pipeline?"** → *"A fixed pipeline always retrieves. The agent
  decides when retrieval is needed, can search multiple times to refine, and could use more tools
  like a calculator or web search. It's more flexible for real tasks."*
- **"Is LLM-as-judge reliable?"** → *"It has known biases — position, verbosity, self-preference.
  I mitigate with a strict rubric and reference answers, and for high-stakes cases I'd add human
  spot-checks and multiple judges."* (This answer signals senior-level awareness.)
- **"What would you add next?"** → *"Hybrid retrieval (keyword + vector) with re-ranking, more agent
  tools, and a CI gate that fails the build if eval scores drop."*

---

## 8. Your pitches (memorize)

**30-second version:**
> *"I built a production-style RAG system in Python. It answers questions from a knowledge base
> with citations, has an agent that decides when to use tools, and — most importantly — a built-in
> evaluation harness that scores answer quality automatically. Most demos stop at 'it gives an
> answer.' Mine proves how good the answer is."*

**2-minute version:** Walk through the flow — *"documents get chunked and embedded into a vector
store; a question is embedded and matched to the closest chunks; the model answers using only those
chunks and cites them; an agent can decide when to search; and an LLM-as-judge scores faithfulness
and relevance on a golden set so quality is measurable."*

---

## 9. Honesty guardrails

- ✅ Say: *"I built this to learn production RAG, evaluation, and agents."*
- ✅ Say: *"I used Gemini for embeddings and generation; the design is provider-agnostic."*
- ❌ Don't claim you trained a model, used CUDA/C++, or ran this at massive scale.
- ❌ Don't claim numbers you can't back up.
- If asked something you don't know: *"I haven't gone deep on that yet, but here's how I'd approach
  it..."* — curiosity beats bluffing every time.

---

## 10. Mini-glossary

- **LLM** - Large Language Model (e.g., Gemini, GPT), the AI that writes text.
- **Token** - a piece of a word; models read/write in tokens.
- **Top-k** - the k most similar chunks we retrieve (default 4).
- **Prompt** - the text we send to the model (instructions + context + question).
- **SSE (Server-Sent Events)** - a way to stream the answer to a browser word-by-word.
- **CI** - Continuous Integration; automated checks that run on every code change.

Study this, run the commands a few times, and the project becomes genuinely yours. Good luck!
