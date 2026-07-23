# Production-Grade Agentic RAG Platform

A state-of-the-art, enterprise-ready **Agentic Retrieval-Augmented Generation (RAG)** platform featuring a **3-Tier Memory Architecture**, **ReAct Agentic Orchestrator**, **Resilient Multi-Provider LLM Gateway**, **OpenTelemetry Sub-Process Latency Waterfall**, and a **Real-Time Database Inspector Dashboard** — served as a unified FastAPI application and embedded Gradio UI.

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
│  • Input Guardrails (Security & Injection Filter)                                            │
│  • Self-RAG Context Verification & Reflection                                                │
│  • Automated Web Search Fallback (Tavily Tool)                                               │
└──────────────────────┬────────────────────────┬────────────────────────┬─────────────────────┘
                       │                        │                        │
                       ▼                        ▼                        ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐ ┌─────────────────────────────┐
│    3-Tier Memory Engine     │ │   Resilient LLM Gateway     │ │  OpenTelemetry Telemetry    │
│  • Working Memory (Turn)    │ │  • Primary: Llama 3.3 70B   │ │  • TTFT & Step Spans        │
│  • Semantic Memory (Facts)  │ │  • Fallback: Llama 3.1 8B   │ │  • Microsecond Latency      │
│  • Episodic (Traces/Logs)   │ │  • Rate-Limit Failover      │ │  • Token & Cost ($) Metrics │
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
* **Automated Failover**: Primary model (`llama-3.3-70b-versatile`) with seamless fallback to lightweight models (`llama-3.1-8b-instant`).
* **Rate Limit Circuit Breaker**: Prevents downtime during provider rate limits or API throttles.
* **Token & Cost Telemetry**: Calculates exact prompt tokens, completion tokens, and real-time USD cost (`$`) for every request.

### 4. ⏱️ OpenTelemetry Granular Latency Suite
* **Time-to-First-Token (TTFT)**: Tracks time to first token emission for real-time user UX monitoring.
* **Sub-Process Millisecond Waterfall**: Measures exact latencies for Guardrails, Memory Retrieval, Query Rewrite, Dense Vector Search, Sparse BM25 Search, Cross-Encoder Reranking, Self-RAG Check, Tool Execution, LLM Generation, and Memory Consolidation.

### 5. 📊 Real-Time Database Inspector & Architecture Dashboard
* **Interactive Component Topology**: Visualizes query execution stepping across nodes in real-time.
* **Live DB Inspector**: Allows browsing and querying all database tables (`semantic_facts`, `episodic_traces`, `procedural_strategies`, `working_sessions`) directly inside the UI without external SQL clients.
* **Embedded Web UI Tab**: Accessible via **Tab 3 ("⚡ System Dashboard & Architecture")** at `http://localhost:8000/`.

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **LLM Gateway** | Groq API (`llama-3.3-70b-versatile` & `llama-3.1-8b-instant`) with automatic failover |
| **Embedding Model** | `BAAI/bge-m3` (GPU-accelerated XPU/CUDA/CPU auto-detection) |
| **Reranker** | `cross-encoder/ms-marco-TinyBERT-L-2-v2` (4× faster than MiniLM) |
| **Vector Store** | Qdrant (persistent multi-vector local storage) |
| **Sparse Search** | BM25 (`rank-bm25` keyword retrieval) |
| **Fusion & Ranking** | Reciprocal Rank Fusion (RRF, `k=60`) |
| **Orchestration** | ReAct Agentic Orchestrator + LangGraph |
| **Memory Database** | SQLite / PostgreSQL via SQLAlchemy ORM & SQLModel |
| **Telemetry** | OpenTelemetry microsecond span tracer |
| **API & Web UI** | FastAPI + Gradio 3-Tab Interface (Chat, Add Knowledge, System Dashboard) |

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

```bash
# Clone repository
git clone https://github.com/abntazim-1/Production-Grade-Advanced-Rag.git
cd Production-Grade-Advanced-Rag

# Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
LLM_MODEL=llama-3.3-70b-versatile
FALLBACK_LLM_MODEL=llama-3.1-8b-instant
DATABASE_URL=sqlite:///./metrics.db
USE_RERANKER=true
VECTOR_STORE_PATH=./qdrant_db
```

### 3. Launch the Application

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **[http://localhost:8000/](http://localhost:8000/)**:
* **💬 Chat Tab**: Multi-turn streaming chat with source citations and timings footer.
* **📁 Add Knowledge Tab**: Drag-and-drop file ingestion (PDF, DOCX, TXT, MD, CSV, JSON) + Live table of uploaded files and deletion tools.
* **⚡ System Dashboard & Architecture Tab**: Real-time topology visualizer, OpenTelemetry latency waterfall card, and live Database Inspector.

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
* **`POST /api/db/raw_query`**: Execute safe read-only SQL queries against the metrics database.

---

## 📁 Repository Structure

```
Production-Grade-Advanced-Rag/
├── agent/
│   └── orchestrator.py      # ReAct agentic orchestrator loop & Self-RAG check
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
│   └── tracer.py            # OpenTelemetry microsecond span tracer & latency collector
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
│   └── guards.py            # Security & injection guardrails
├── docs/                    # Architectural specs, reports, and benchmarks
├── config.py                # Pydantic system settings
├── models.py                # Pydantic data schemas
├── client.py                # CLI interactive terminal chat client
├── test_pipeline.py         # End-to-end integration test suite
├── requirements.txt         # Dependencies manifest
└── README.md                # Project documentation
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

