# Production-Grade Agentic RAG Platform

A state-of-the-art, enterprise-ready **Agentic Retrieval-Augmented Generation (RAG)** platform featuring a **3-Tier Memory Architecture**, **ReAct Agentic Orchestrator**, **Resilient Multi-Provider LLM Gateway**, **Sub-Process Latency Instrumentation**, and a **Real-Time Database Inspector Dashboard** — served as a unified FastAPI application and embedded Gradio UI.

---

## 🏛️ System Architecture

```
                                  ┌──────────────────────────┐
                                  │       User / Client      │
                                  │   Web UI / Gradio / API  │
                                  └─────────────┬────────────┘
                                                │ User Query
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       FastAPI Server                                         │
│   Endpoints: /query  /query/stream  /ingest  /api/db/tables  /api/db/table/{name}  /dashboard│
└───────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                                │
                                                ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 Agentic ReAct Orchestrator                                   │
│  • Input Guardrails (Llama Prompt Guard 2 injection classifier)                              │
│  • Self-RAG Context Verification & Reflection                                                │
│  • Automated Web Search Fallback (Tavily Tool)                                               │
└──────────────────────┬────────────────────────┬────────────────────────┬─────────────────────┘
                       │                        │                        │
                       ▼                        ▼                        ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│    3-Tier Memory Engine     │ │   Resilient LLM Gateway     │ │   Latency Instrumentation   │
│  • Working Memory (Turn)    │ │  • Primary: gpt-oss-120b    │ │  • TTFT & Step Spans        │
│  • Semantic Memory (Facts)  │ │  • Fallback: qwen3.8-27b    │ │  • Microsecond Latency      │
│  • Episodic (Traces/Logs)   │ │  • Rate-Limit Failover      │ │  • Token & Cost (estimated) │
│  • Procedural (Plan DAGs)   │ │  • Token & Cost Tracker     │ │                             │
└─────────────────────────────┘ └───────────────┬─────────────┘ └─────────────────────────────┘
                                                │
                                                ▼
                                   ┌────────────────────────┐
                                   │    Hybrid Retrieval    │
                                   └───────┬────────┬───────┘
                                           │        │
                                    ┌──────┘        └──────┐
                                    ▼                      ▼
                             ┌────────────┐         ┌────────────┐
                             │Dense Search│         │Sparse Search│
                             │  BGE-M3    │         │    BM25    │
                             │  Qdrant    │         │  Keyword   │
                             └──────┬─────┘         └─────┬──────┘
                                    │                     │
                                    └──────────┬──────────┘
                                               ▼
                                     Reciprocal Rank Fusion
                                               │
                                               ▼
                                ┌──────────────────────────────┐
                                │   Cross-Encoder Reranker     │
                                │   ms-marco-TinyBERT-L-2-v2   │
                                └──────────────┬───────────────┘
                                               │
                                               ▼
                                     Top Relevant Chunks
                                               │
                                               ▼
                                ┌──────────────────────────────┐
                                │   Context Builder & Prompts  │
                                └──────────────┬───────────────┘
                                               │
                                               ▼
                                ┌──────────────────────────────┐
                                │  Resilient LLM Generation    │
                                │    (Streaming Output)        │
                                └──────────────┬───────────────┘
                                               │
                                               ▼
                                    User Streamed Response
```

---

## ⚡ Key Highlights & Core Features

### 1. 🧠 3-Tier Enterprise Memory Architecture
* **Short-Term Working Memory**: Preserves active conversational context, turn histories, and scratchpad execution states per session.
* **Long-Term Semantic Memory**: Extracts and stores subject-predicate-object knowledge triples and user facts in persistent relational tables.
* **Episodic Trace Memory**: Captures complete execution trajectories, step plans, tool calls, final responses, and token costs for auditability.
* **Procedural Memory**: Learns and indexes successful workflow strategies and execution DAGs for automated query planning.

### 2. 🤖 Agentic ReAct Orchestrator & Self-RAG
* **Dynamic Planning & Reasoning**: Decomposes complex queries into actionable step plans.
* **Self-RAG Reflection**: Evaluates whether retrieved document contexts contain sufficient evidence to answer the query.
* **Web Search Fallback**: Automatically invokes external search tools (e.g. Tavily API) if internal vector knowledge is insufficient.
* **Failure Recovery**: Self-corrects and reroutes queries on low confidence or execution failures.

### 3. 🛡️ Resilient Multi-Provider LLM Gateway
* **Automated Failover**: Primary model (`openai/gpt-oss-120b`) falling back through `qwen/qwen3.8-27b` and `openai/gpt-oss-20b` on error or rate limit.
* **Rate Limit Handling**: Metadata generation retries on HTTP 429 with backoff, honouring the provider's own retry hint.
* **Token & Cost Estimates**: Reports prompt tokens, completion tokens and USD cost per request. See the note under Telemetry — these are estimates, not billed figures.

### 4. ⏱️ Granular Latency Instrumentation
* **Sub-Process Millisecond Timings**: Records per-stage latency for Guardrails, Memory Retrieval, Query Rewrite, Dense Vector Search, Sparse BM25 Search, Cross-Encoder Reranking, Self-RAG Check, Tool Execution, LLM Generation and Memory Consolidation, returned in every `/query` response.
* **Time-to-First-Token (TTFT)**: Measured directly on the streaming path (`/query/stream`, Gradio chat).

> **What these numbers are and are not.** [`observability/tracer.py`](observability/tracer.py)
> is a hand-rolled span logger — it is named after OpenTelemetry but does not use
> the OTel SDK, and has no exporter or collector; spans are logged and dropped.
> Token counts come from `len(text) // 4`, not the provider's usage response, so
> every derived USD figure is an approximation. On the non-streaming `/query`
> path, `ttft_ms` is `generation_ms * 0.35` — a constant, not a measurement.
> Wall-clock stage timings are real.

### 5. ✂️ Semantic Chunking with Overlap

Documents are split by **structure first, meaning second**:

1. Code blocks and tables are lifted out whole and never split.
2. Headings divide the document into sections; each section keeps its heading.
3. Within a section, sentences are embedded and boundaries are placed where consecutive sentences are **least similar** — the topic shifts.
4. Each chunk carries **15–20% of the previous chunk's sentences** as overlap, so a fact sitting on a boundary stays retrievable from both sides. Overlap never crosses a heading, table or code boundary.

Headings are embedded with the chunk body, indexed by BM25, **and** scored by the reranker, so a section titled `TTS (Target: 40ms)` is findable even when its body never repeats those words.

Sizing: `CHUNK_SIZE` caps the *new* content per chunk; overlap is added on top, so a finished chunk reaches ~1.2× that. Sentences are indivisible, so the realised overlap lands as close to the target as sentence boundaries allow.

> **Ingestion is dominated by LLM metadata generation** — one call per chunk. On a
> 15-PDF / 484-chunk corpus: ~7 min total, of which ~72% is enrichment and ~16%
> embedding. Set `GENERATE_METADATA=false` for bulk loads (~100s for the same
> corpus). Full enrichment is floored by your account's tokens-per-minute limit.

### 6. 🔒 Injection Guardrails

Input checking is two-stage:

1. A short regex list of unambiguous jailbreak phrasings — free, and the offline fallback.
2. **`meta-llama/llama-prompt-guard-2-86m`**, a model trained for injection detection, returning a probability scored against `PROMPT_GUARD_THRESHOLD`.

Measured on 16 realistic questions about the indexed corpus plus 5 attacks: **0 false positives, 5/5 attacks caught**, with benign queries scoring ~0.001 and attacks ~0.9995. Costs ~380ms per query; set `USE_PROMPT_GUARD=false` to fall back to regex alone.

The check **fails open** — an unreachable classifier degrades to the regex result rather than blocking all traffic.

### 7. 📊 Real-Time Database Inspector & Architecture Dashboard
* **Interactive Component Topology**: Visualizes query execution stepping across nodes in real-time.
* **Live DB Inspector**: Allows browsing and querying all database tables (`semantic_facts`, `episodic_traces`, `procedural_strategies`, `working_sessions`) directly inside the UI without external SQL clients.
* **Embedded Web UI Tab**: Accessible via **Tab 3 ("⚡ System Dashboard & Architecture")** at `http://localhost:8000/`.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **LLM Gateway** | Groq API (`openai/gpt-oss-120b` → `qwen/qwen3.8-27b` → `openai/gpt-oss-20b`) with automatic failover |
| **Embedding Model** | `BAAI/bge-m3` (GPU-accelerated XPU/CUDA/CPU auto-detection) |
| **Reranker** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` (4× faster than MiniLM) |
| **Chunking** | Semantic — sentence embeddings, breakpoints at meaning shifts, 15–20% overlap |
| **Vector Store** | Qdrant, embedded (persistent multi-vector local storage, single-process) |
| **Sparse Search** | BM25 (`rank-bm25` keyword retrieval, rebuilt in memory at startup) |
| **Fusion & Ranking** | Reciprocal Rank Fusion (RRF, `k=60`) |
| **Guardrails** | `meta-llama/llama-prompt-guard-2-86m` + narrow regex pre-filter |
| **Orchestration** | ReAct Agentic Orchestrator + LangGraph |
| **Memory Database** | SQLite via SQLAlchemy ORM (path hardcoded; Postgres not wired) |
| **Telemetry** | In-process span logger with per-stage millisecond timings |
| **API & Web UI** | FastAPI + Gradio 3-Tab Interface (Chat, Add Knowledge, System Dashboard) |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

```bash
# Clone repository
git clone https://github.com/abntazim-1/Agentic-RAG-Platform.git
cd Agentic-RAG-Platform

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

> **Note:** document loaders are imported at runtime but are not all pinned in
> `requirements.txt` yet. For PDF and DOCX ingestion also install:
> ```bash
> pip install pypdf python-docx pdfminer.six
> ```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # Optional: real web-search fallback

# Models — verify against GET https://api.groq.com/openai/v1/models before changing.
# Groq retires models; llama-3.3-70b-versatile and llama-3.1-8b-instant now 404.
LLM_MODEL=openai/gpt-oss-120b
METADATA_MODEL=qwen/qwen3.8-27b          # metadata generation + query rewriting

# Retrieval
USE_RERANKER=true
TOP_K_RETRIEVAL=15
TOP_K_RERANK=5

# Chunking (semantic)
CHUNK_SIZE=1000                          # max characters of new content per chunk
CHUNK_OVERLAP_RATIO=0.175                # honoured as a 15-20% band

# Guardrails
USE_PROMPT_GUARD=true                    # Llama Prompt Guard 2; ~380ms per query

# Storage
QDRANT_PATH=qdrant_db
```

> **Every key must match a field name in [`config.py`](config.py), uppercased.**
> `Config.extra = "ignore"` means a misspelled or invented key is silently
> dropped rather than raising — so a typo looks like it worked. There is no
> `FALLBACK_LLM_MODEL` (the fallback is `METADATA_MODEL`) and no
> `VECTOR_STORE_PATH` (it is `QDRANT_PATH`). `DATABASE_URL` is ignored entirely;
> [`db/database.py`](db/database.py) hardcodes `sqlite:///<repo>/metrics.db`.

### 3. Launch the Application

```bash
.venv/Scripts/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

> Startup takes ~20s: BGE-M3 and the cross-encoder both load and warm up, and
> the vector store rehydrates every chunk from Qdrant so BM25 can be rebuilt in
> memory. Qdrant is embedded and holds an **exclusive file lock** on
> `qdrant_db/`, so only one process may run at a time — stop the server before
> running `test_pipeline.py` or `stress_test.py`.

Open your browser at **[http://localhost:8000/](http://localhost:8000/)**:
* **💬 Chat Tab**: Multi-turn streaming chat with source citations and timings footer.
* **📁 Add Knowledge Tab**: Drag-and-drop file ingestion (PDF, DOCX, TXT, MD, CSV, JSON) + Live table of uploaded files and deletion tools.
* **⚡ System Dashboard & Architecture Tab**: Real-time topology visualizer, latency waterfall card, and live Database Inspector.

---

## 🔌 API Endpoints

### Core RAG & Ingestion Endpoints

* **`GET /health`**
  ```bash
  curl http://localhost:8000/health
  # Returns: status, chunks_in_memory, vectors_in_qdrant, active_sessions, indexed_sources
  ```

* **`POST /ingest`**
  ```bash
  curl -X POST http://localhost:8000/ingest \
    -H "Content-Type: application/json" \
    -d '{"text": "Document content...", "source": "guide.pdf"}'
  ```

* **`POST /query`**
  ```bash
  curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"query": "What are the project details?", "session_id": "sess-001"}'
  ```

* **`POST /query/stream`** (Server-Sent Events streaming)
  ```bash
  curl -X POST http://localhost:8000/query/stream \
    -H "Content-Type: application/json" \
    -d '{"query": "Explain the architecture", "session_id": "sess-001"}'
  ```

* **`DELETE /source/{source_name}`**
  ```bash
  curl -X DELETE http://localhost:8000/source/guide.pdf
  ```

### Real-Time Database Inspector Endpoints

* **`GET /api/db/tables`**: List all database tables and record counts.
* **`GET /api/db/table/{table_name}?limit=50`**: View live records from any memory table (`semantic_facts`, `episodic_traces`, `procedural_strategies`, `working_sessions`).
* **`GET /api/db/raw_query?query_str=...`**: Execute read-only `SELECT` queries against the metrics database.

> ⚠️ These inspector routes are **unauthenticated**, as is every other endpoint.
> `raw_query` will return any row the caller asks for. Do not expose this service
> to a network you do not control until authentication is in place.

---

## 📁 Repository Structure

```
Agentic-RAG-Platform/
├── agent/
│   ├── orchestrator.py      # ReAct agentic orchestrator loop & Self-RAG check (serves all traffic)
│   ├── graph.py             # LangGraph rewrite→retrieve→generate pipeline (backs /evaluate only)
│   └── multi_agent.py       # Unreferenced; kept for reference
├── ingestion/
│   ├── chunker.py           # Semantic chunker — structure + meaning boundaries + overlap
│   ├── metadata.py          # LLM metadata enrichment (summary, keywords, questions)
│   └── loaders.py           # File-type loaders
├── evaluation/
│   └── evaluator.py         # Embedding-based faithfulness & answer relevancy
├── api/
│   ├── app.py               # FastAPI application + SSE streaming + Gradio 3-tab UI
│   └── db_router.py         # Database inspector REST API endpoints
├── db/
│   ├── database.py          # SQLAlchemy engine & session factory
│   └── models.py            # ORM models for 3-tier memory & query metrics
├── gateway/
│   └── llm_gateway.py       # Resilient multi-provider LLM router with fallback & cost tracking
├── memory/
│   ├── three_tier_memory.py # 3-Tier Memory Manager (Working, Semantic, Episodic, Procedural)
│   └── conversation.py      # Short-term thread-safe session memory
├── observability/
│   └── tracer.py            # In-process span logger & per-stage latency collector
├── dashboard/
│   └── index.html           # Production interactive system topology & DB inspector dashboard
├── retrieval/
│   ├── embedder.py          # BGE-M3 GPU embedder + LRU cache
│   ├── vector_store.py      # Persistent Qdrant vector database store
│   ├── bm25_store.py        # Sparse BM25 keyword store
│   ├── hybrid_retriever.py  # Parallel dense+sparse retrieval with RRF
│   └── query_rewriter.py    # Multi-turn query rewriter
├── reranking/
│   └── reranker.py          # TinyBERT cross-encoder reranker
├── guardrails/
│   └── guards.py            # Prompt Guard 2 classifier + regex pre-filter
├── eval-dashboard/          # Separate Next.js app reading /api/eval/metrics
├── docs/                    # Architectural specs, reports, and benchmarks (gitignored)
├── config.py                # Pydantic system settings — the source of truth for env var names
├── models.py                # Pydantic data schemas
├── client.py                # CLI interactive terminal chat client
├── mini_rag.py              # Original single-file prototype; the modular code is a port of it
├── test_pipeline.py         # Pipeline smoke script (writes to the real vector store)
├── stress_test.py           # Load & latency harness against a running server
├── CLAUDE.md                # Repo guidance: architecture notes and known gotchas
├── requirements.txt         # Dependencies manifest
└── README.md                # Project documentation
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

