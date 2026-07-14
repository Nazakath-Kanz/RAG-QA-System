# Hybrid RAG QA System with Agentic Self-Correction

A retrieval-augmented generation system that answers natural-language questions about uploaded PDF/text documents. Built as an independent portfolio project to explore production-grade RAG engineering: hybrid retrieval, table-aware document parsing, an agentic self-correction layer, automated evaluation and full observability. It's not a single-pass "embed and ask" tutorial project.

Every claim in this README is backed by something actually run and verified during development. Where something didn't fully work, that's documented too, along with why.

---

## Table of Contents

- [What This Is](#what-this-is)
- [Architecture](#architecture)
- [Key Engineering Decisions](#key-engineering-decisions)
- [Tech Stack](#tech-stack)
- [Evaluation](#evaluation)
- [Known Limitations](#known-limitations)
- [Observability](#observability)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the System](#running-the-system)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)

---

## What This Is

Upload a PDF (or `.txt` file) — a resume, a certificate, a marksheet, any document mixing prose and tables — and ask natural-language questions about its contents. The system:

1. Extracts text **and** table structure from the PDF. Most naive extractors flatten tables into unreadable, positionally-dependent text; this one detects tables and converts each row into an explicit, unambiguous sentence.
2. Chunks content in a way that respects line and paragraph boundaries, so a fact never gets split mid-sentence across chunks.
3. Retrieves relevant chunks using **hybrid search** — BM25 keyword matching fused with dense vector search, then reranked with a cross-encoder.
4. Generates an answer strictly grounded in retrieved context, refusing to answer when context doesn't support one.
5. Optionally routes the query through an **agentic layer** (LangGraph) that detects a failed retrieval and retries once with a rephrased query before giving up.

Two endpoints expose this: `/ask` (single-pass) and `/ask_agentic` (self-correcting). A Streamlit frontend toggles between them live.

---

## Architecture

```
                    ┌─────────────────────┐
                    │   PDF / TXT Upload   │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  Table-Aware Extraction│
                    │  (pdfplumber)          │
                    │  - Detects tables       │
                    │  - Converts rows to      │
                    │    explicit labeled       │
                    │    sentences                │
                    │  - Crops table regions        │
                    │    from plain-text pass to     │
                    │    avoid duplicate indexing      │
                    └──────────┬────────────────────────┘
                               │
                    ┌──────────▼────────────────────┐
                    │  Parent-Child Chunking          │
                    │  (LangChain                     │
                    │   RecursiveCharacterTextSplitter)│
                    │  Line/paragraph-aware, not        │
                    │  naive character slicing            │
                    └──────────┬─────────────────────────┘
                               │
              ┌────────────────▼────────────────────┐
              │   Dual Indexing on Upload             │
              │   ├── Dense embeddings → Qdrant        │
              │   │   (all-MiniLM-L6-v2, 384-dim)        │
              │   └── Sparse corpus → BM25 (pickled)       │
              └────────────────┬────────────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │     User Query         │
                     └──────────┬─────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   Hybrid Retrieval                    │
              │   LangChain EnsembleRetriever          │
              │   (BM25Retriever + Qdrant dense,         │
              │    weighted 50/50)                        │
              └─────────────────┬─────────────────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │   FlashRank Cross-Encoder Reranking    │
              │   - Always keeps top-1 result             │
              │     regardless of raw score                 │
              │   - Additional results gated by a             │
              │     relevance threshold                          │
              └─────────────────┬───────────────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │  Gemini Generation    │
                     │  (google-genai SDK,    │
                     │   temperature=0.0,       │
                     │   strict context-bound     │
                     │   system prompt)              │
                     └──────────┬──────────────────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Answer + Sources    │
                     └───────────────────────┘

    Alternative path — /ask_agentic (LangGraph):

    START → retrieve → generate → check_confidence
                                       │
                        ┌──────────────┼───────────────┐
                    confident      low confidence    max retries
                        │           (retry < 1)          hit
                       END          rewrite_query          END
                                       │
                                       └──→ back to retrieve
```

---

## Key Engineering Decisions

Every choice below was made to solve a real, observed problem during development — not added speculatively.

### Why hybrid retrieval (BM25 + dense), not dense-only

Dense embedding search is good at semantic closeness but unreliable on exact literal terms. During testing, `"what is the roll no?"` succeeded while `"what is the roll number?"` failed on identical underlying data — pure embedding-space luck. BM25 catches exact tokens regardless of phrasing distance, giving dense search a reliable fallback. Fused via LangChain's `EnsembleRetriever`.

### Why table-aware extraction, not flat text extraction

Naive PDF extraction (`pypdf`) flattens a table into loose text lines — the relationship between a column header and a row's value is only implied by position, never stated. Questions like *"what are the theory marks for Science?"* failed because no sentence anywhere actually said that. `pdfplumber`-based extraction detects table regions, converts each row into an explicit sentence (`"SUBJECT NAME: SCIENCE. THEORY: 070."`), and crops the table's bounding box out of the plain-text pass to avoid double-indexing the same data.

### Why the reranker always keeps its top-1 result

The initial design filtered reranked results with an absolute score threshold (`score > 0.05`). This discarded genuinely correct answers to narrow, single-fact queries, because a cross-encoder's confidence score isn't comparable across queries of different specificity. The current policy always accepts the reranker's #1 result and only applies the threshold to additional supporting context.

### Why LangGraph, not a linear LangChain chain

The agentic layer targets a specific, observed failure: phrasing sensitivity in retrieval. `/ask_agentic` detects a fallback response and retries once with a rewritten query. This requires runtime-conditional branching (retrieve → generate → confidence check → conditional retry), which is what LangGraph is designed for, unlike a linear chain that can't branch on state.

### Why Ragas for evaluation

Manual testing doesn't scale and isn't reproducible. Ragas provides an LLM-judged, quantifiable measure of whether generated answers are grounded in retrieved context — and, as documented below, it actually caught a real generation bug that manual spot-checking had missed.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (async), Uvicorn |
| Vector DB | Qdrant (Dockerized) |
| Dense embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Sparse retrieval | BM25 (`rank_bm25`, via `langchain_community.BM25Retriever`) |
| Retrieval fusion | LangChain `EnsembleRetriever` |
| Reranking | FlashRank (`ms-marco-MiniLM-L-12-v2` cross-encoder) |
| PDF/table extraction | `pdfplumber` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Generation | Google Gemini (`gemini-2.5-flash`) via `google-genai` SDK |
| Agentic orchestration | LangGraph |
| Evaluation | Ragas (Faithfulness metric, Gemini-judged) |
| Observability | LangSmith |
| Frontend | Streamlit |

---

## Evaluation

Evaluated using [Ragas](https://github.com/explodinggym/ragas)'s **Faithfulness** metric — whether every claim in a generated answer is actually supported by retrieved context, judged by an independent LLM rather than developer self-assessment. Run against 8 questions covering direct lookups, table-derived facts, and a deliberate out-of-scope refusal case, using the real end-to-end pipeline (no mocking).

| # | Question | Result | Faithfulness |
|---|---|---|---|
| 1 | Subject code for science | Correct | 1.0 |
| 2 | Roll number | Correct | 1.0 |
| 3 | Total marks for Malayalam | Correct | — (judge call rate-limited) |
| 4 | Date of birth | Correct | 1.0 |
| 5 | Grade in mathematics | **Incorrect — see below** | 0.0 |
| 6 | Kanz's mother's name | Correct | — (judge call rate-limited) |
| 7 | School attended | Correct | 1.0 |
| 8 | Aadhaar number (not in document) | Correctly refused | 0.0 (metric artifact, see below) |

**Reading these results honestly, not just averaging them:**

- **6 of 8 answers were correct**, with a Faithfulness score of 1.0 wherever judging completed.
- **2 judge calls (#3, #6) failed to complete** due to Gemini free-tier per-minute rate limiting during the evaluation run itself — a real constraint of running LLM-judged eval on a free API tier, documented rather than hidden. The generated answers for both were independently confirmed correct by inspection.
- **Question 5 surfaced a genuine bug**: the correct data (`SUBJECT NAME: MATHEMATICS STANDARD. THEORY: 044. ... POSITIONAL GRADE: B2.`) was present in the retrieved context, but the model answered "no relevant context found" anyway — a generation-layer failure, not a retrieval failure. This is exactly the kind of issue an automated eval harness is meant to catch that manual spot-checking can miss, and it's flagged as an open item below rather than glossed over.
- **Question 8's low score is a metric characteristic, not a system failure.** Refusing to answer (correctly, since Aadhaar numbers aren't in this document) makes no factual claims for Faithfulness to verify, so the metric scores it near zero by construction. The system behavior itself — refusing rather than hallucinating — was correct.

---

## Known Limitations

Documenting these honestly is itself part of the engineering practice this project demonstrates.

- **Generation can occasionally miss data present in retrieved context** (see Question 5 above). Root cause not yet fully isolated — candidate factors include context density (5 rows in one chunk) and load-related noise during the evaluation run itself. Flagged for further investigation.
- **The agentic retry can't resolve semantic/world-knowledge gaps.** The rewrite node paraphrases a failed query (e.g. "roll no" ↔ "roll number") but can't substitute specific facts it doesn't know — `"how did Kanz do in his mother tongue subject?"` fails, because resolving "mother tongue" to "Malayalam" requires external knowledge the rewrite prompt isn't given. Verified and documented as a genuine boundary between query rewriting and true entity-linking.
- **Single-document scope** — each upload purges the entire index rather than supporting a multi-document knowledge base with per-document filtering.
- **Free-tier API quota** (20 requests/day, 5/minute on Gemini's free tier) limits real-time usage and evaluation scale, as seen directly in the results above.
- **No automated test suite** — verification was done via manual and scripted smoke tests, not `pytest`-based unit/integration tests.
- **Ragas's dependency chain requires a local import stub** (see Troubleshooting) due to a removed class in a transitive dependency. This does not persist across a fresh virtual environment and must be reapplied.

---

## Observability

Every request through `/ask` or `/ask_agentic` is traced end-to-end via LangSmith:

- Full LangGraph node execution (retrieve → generate → confidence check → optional rewrite)
- Nested spans for hybrid retrieval internals (BM25Retriever, Qdrant dense retriever, EnsembleRetriever fusion), individually timed
- Generation calls, including latency and failure states — rate-limit errors surface as failed spans with full error detail, not silent failures

Observed component latencies in testing: BM25 ~0.04s, dense retrieval ~0.06s, full hybrid fusion ~0.26s, generation ~0.45s, full agentic round-trip ~0.81s.

---

## Project Structure

```
RAG System/
├── src/
│   ├── main.py                 # FastAPI app: /upload, /ask, /ask_agentic
│   ├── config.py                # Environment/settings management
│   ├── database.py               # Qdrant client
│   ├── agent/
│   │   └── graph.py                # LangGraph agentic layer
│   ├── generation/
│   │   └── llm.py                    # Gemini generation (google-genai SDK)
│   ├── ingestion/
│   │   ├── pdf_parsing.py              # Table-aware PDF extraction
│   │   └── pipeline.py                   # Chunking + embedding + indexing
│   ├── models/
│   │   └── schemas.py                      # Pydantic request/response models
│   └── retrieval/
│       └── search.py                         # Hybrid retrieval + reranking
├── app.py                        # Streamlit frontend
├── eval_ragas.py                   # Ragas evaluation harness (8-question golden set)
├── debug_retrieval.py                # Retrieval-only test script (no LLM calls, no cost)
├── create_stub.py                      # One-time setup: patches missing Vertex AI import
├── docker-compose.yml                    # Qdrant container definition
├── requirements.txt
└── .env                                     # API keys (not committed)
```

---

## Setup

### 1. Prerequisites

- Python 3.11+ (developed and tested on 3.14)
- Docker Desktop (for Qdrant)
- A Google Gemini API key — [Google AI Studio](https://aistudio.google.com/)
- (Optional but recommended) A LangSmith API key for tracing — [smith.langchain.com](https://smith.langchain.com/)
- **Windows users:** Microsoft C++ Build Tools may be required for some dependencies. See Troubleshooting if you hit a compiler error during install.

### 2. Clone and set up the environment

```bash
git clone <your-repo-url>
cd "RAG System"

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install` fails on `scikit-network` with a compiler error, see [Troubleshooting](#troubleshooting) — it's a known optional dependency issue, not a problem with this project's code.

### 3. Patch the Ragas/Vertex AI import stub

Ragas's dependency chain imports `ChatVertexAI` from `langchain_community`, which has been removed from recent versions of that package. Since this project doesn't use Vertex AI, a lightweight stub satisfies the unused import:

```bash
python create_stub.py
```

**This must be rerun any time the virtual environment is recreated from scratch** — the stub is written into `venv/`, which is git-ignored and not tracked. Forgetting this step is the single most likely setup issue for a fresh clone; see Troubleshooting.

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optional — enables LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=rag-qa-system
```

**Never commit this file.** Confirm `.env` is listed in `.gitignore` before your first commit.

### 5. Start Qdrant

```bash
docker-compose up -d
```

Verify it's running:
```bash
docker ps
```
You should see a container named `rag_vector_db`. Confirm visually at `http://localhost:6333/dashboard`.

---

## Running the System

**Terminal 1 — backend:**
```bash
uvicorn src.main:app --reload --port 8000
```
Wait for `Application startup complete` with no errors.

**Terminal 2 — frontend:**
```bash
streamlit run app.py --server.port 8501
```

Open `http://localhost:8501`, upload a PDF, click "Process & Index Data," then ask questions. Use the "Agentic Mode" toggle to switch between `/ask` and `/ask_agentic`.

Alternatively, interact directly via the auto-generated API docs at `http://127.0.0.1:8000/docs` — useful for testing endpoints without the frontend.

**To run retrieval tests without using any API quota:**
```bash
python debug_retrieval.py
```
This exercises the full retrieval + reranking pipeline against whatever document is currently indexed, without calling Gemini at all — the fastest way to confirm ingestion and retrieval are working.

**To run the evaluation harness** (uses Gemini quota — see note in Evaluation section):
```bash
python eval_ragas.py
```

---

## API Reference

### `POST /upload`
Ingests a `.pdf` or `.txt` file. **Purges the existing vector index** and rebuilds it from the new document — uploading a new file replaces, not adds to, the knowledge base.

### `POST /ask`
Single-pass RAG: hybrid retrieval → reranking → grounded generation.
```json
{ "question": "what is the roll number?" }
```

### `POST /ask_agentic`
Agentic RAG via LangGraph: retrieval → generation → confidence check → (if low-confidence) query rewrite → retry, once.
```json
{ "question": "what is the roll number?" }
```

Both return:
```json
{
  "query": "...",
  "answer": "...",
  "context_used": ["..."]
}
```

---

## Troubleshooting

Every error below was actually encountered during development of this project. If you hit something not listed here, check the terminal output carefully — the actual Python traceback is almost always more informative than the last line alone.

### `httpcore.ConnectError: All connection attempts failed` on startup

**Cause:** Qdrant isn't running. This is the most common startup error.
**Fix:**
```bash
docker-compose up -d
docker ps   # confirm rag_vector_db appears
```
Then restart `uvicorn`.

### `ModuleNotFoundError: No module named 'langchain.retrievers'` (or similar LangChain import errors)

**Cause:** LangChain 1.x moved several classes (including `EnsembleRetriever`) out of the base package into `langchain-classic`.
**Fix:**
```bash
pip install langchain-classic
```
Import as `from langchain_classic.retrievers import EnsembleRetriever`. `BM25Retriever` remains in `langchain_community.retrievers`.

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

**Cause:** Ragas's dependency chain imports `ChatVertexAI`, which has been removed from current `langchain_community`. This project doesn't use Vertex AI, so this is a dead import that only needs to be satisfied, not actually used.
**Fix:** Run `python create_stub.py` (see Setup step 3). If the error persists, confirm the stub file exists at `venv/Lib/site-packages/langchain_community/chat_models/vertexai.py`.

### `error: Microsoft Visual C++ 14.0 or greater is required` during `pip install`

**Cause:** A dependency (commonly `scikit-network`, an optional Ragas dependency not required for this project's features) needs to compile from source, and no C++ compiler is available on Windows by default.
**Fix (recommended for a one-time cost):** Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the "Desktop development with C++" workload.
**Fix (faster workaround):** Install the failing package with `--no-deps` to skip the problematic dependency, then manually install only the sub-dependencies actually needed (see `requirements.txt` for the resolved set used in this project).

### `429 RESOURCE_EXHAUSTED` errors from Gemini

**Cause:** Free-tier quota exhausted — either the per-minute limit (5 requests/minute) or the daily limit (20 requests/day).
**Fix:** Wait. The per-minute limit clears within about a minute. The daily limit resets at midnight **Pacific Time**, not local time — for IST users, this lands around 12:30 PM IST, not midnight IST. Check real-time usage at `https://ai.dev/rate-limit` (note: the "Rate Limit" page shows a 28-day peak, not live remaining quota — the "API Usage" dashboard's daily request chart is more reliable for checking today's actual usage).
**Mitigation:** `eval_ragas.py` is designed with a capped `RunConfig` (`max_retries=2`) so a quota exhaustion fails within about a minute rather than retrying indefinitely.

### `SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes` when running a script with a Windows file path

**Cause:** Python interprets `\U` in a plain string as the start of a unicode escape sequence.
**Fix:** Use a raw string (prefix with `r`) or forward slashes:
```python
path = r"C:\Users\name\file.pdf"
# or
path = "C:/Users/name/file.pdf"
```

### Retrieval works inconsistently for near-identical phrasing (e.g. "roll no" vs "roll number")

**Cause:** This was a real bug during development, caused by an absolute score threshold in the reranking step discarding valid low-scoring results.
**Fix:** Already resolved in `src/retrieval/search.py` — the reranker's top-1 result is always kept regardless of raw score. If you see this behavior after modifying that file, check that the top-1 guard logic wasn't accidentally removed.

### Stale/duplicate data appears in retrieval results after re-uploading a document

**Cause:** `/upload` must actively call `qdrant_client.delete_collection()` before reindexing. If this call is missing or fails silently, old chunks persist alongside new ones.
**Fix:** Confirm `src/main.py`'s `/upload` route logs `"Existing collection dropped."` on every upload. If not, verify the `delete_collection` call is present and not wrapped in a `try/except` that's swallowing a real error.

---

## Roadmap

- [ ] Investigate and fix the Question 5 generation failure (correct context retrieved, incorrect refusal generated)
- [ ] Expand the Ragas evaluation set beyond 8 questions and add a second metric (e.g. Context Recall) once quota allows sustained testing
- [ ] Multi-document support with metadata filtering, instead of full-index purge on each upload
- [ ] Automated `pytest` test suite for retrieval and ingestion logic
- [ ] Rate-limit-aware retry/backoff handling in the generation layer for production resilience
- [ ] Migrate Ragas metric imports from the deprecated `ragas.metrics` path to `ragas.metrics.collections` ahead of the v1.0 removal
