# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

A single-process agentic RAG service: FastAPI + an embedded Gradio UI, backed by Qdrant
(dense) + BM25 (sparse), a cross-encoder reranker, a ReAct-style orchestrator, and a
SQLite-backed multi-tier memory store. `README.md` covers the feature set and endpoints;
this file covers what is not obvious from reading it.

## Commands

Run the server (serves the API, the Gradio UI at `/`, and the dashboard at `/dashboard`):

```bash
.venv/Scripts/python -m uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Pipeline smoke test (chunk to embed to BM25 to hybrid retrieve to rerank; no Groq key needed):

```bash
.venv/Scripts/python test_pipeline.py
```

Load/latency harness against a running server:

```bash
.venv/Scripts/python stress_test.py
```

Interactive CLI client against a running server:

```bash
.venv/Scripts/python client.py --url http://localhost:8000
```

Next.js eval dashboard (separate app, expects the API on 127.0.0.1:8000):

```bash
cd eval-dashboard && npm run dev
```

There is no pytest suite and no linter configured on the Python side. `test_pipeline.py` and
`stress_test.py` are scripts, not test cases.

## Hard constraints

**Qdrant holds an exclusive file lock on `qdrant_db/`.** `VectorStore` uses embedded
`QdrantClient(path=...)`, so only one process may touch it at a time. Running
`test_pipeline.py` or `stress_test.py` while uvicorn is up fails with a
storage-already-accessed error. Stop the server first.

**Startup is slow and stateful.** [api/app.py](api/app.py) builds every singleton at import
time — BGE-M3 and the cross-encoder both load and warm up, and
`VectorStore._restore_from_qdrant()` rehydrates `all_chunks` from Qdrant so BM25 can be
rebuilt in memory. Expect tens of seconds before the first request; `--reload` re-pays this
on every edit.

**Env vars must match `Settings` field names in [config.py](config.py), uppercased.**
`Config.extra = "ignore"` means a misspelled key is silently dropped rather than erroring.
The sample `.env` in `README.md` is partly wrong on this point: there is no
`FALLBACK_LLM_MODEL` (the fallback model is `METADATA_MODEL`), no `VECTOR_STORE_PATH` (it is
`QDRANT_PATH`), and `DATABASE_URL` is ignored entirely — [db/database.py](db/database.py)
hardcodes `sqlite:///<repo>/metrics.db`.

## Architecture

[mini_rag.py](mini_rag.py) (~1100 lines, root) is the original single-file prototype.
Everything under `retrieval/`, `ingestion/`, `reranking/`, `agent/graph.py`, and `api/app.py`
is a port of it, and their docstrings say so ("matches mini_rag.py exactly"). Nothing imports
it. Treat it as the reference implementation when behavior is ambiguous, but change the
modular code, not `mini_rag.py`.

**Two query paths coexist.** `build_simple_graph()` in [agent/graph.py](agent/graph.py)
produces the LangGraph `rewrite → retrieve → generate` pipeline plus helpers (`run`,
`generate_stream`, `cached_retrieve_and_rerank`).
[ProductionAgenticOrchestrator](agent/orchestrator.py) is the newer ReAct loop: guardrails →
memory read → rewrite → hybrid retrieve → Self-RAG check → optional Tavily tool → generate →
memory consolidation. **The orchestrator is what actually serves traffic** — `/query`,
`/query/stream`, and the Gradio chat all call it. The graph's `run` survives only to back
`/evaluate`.

`run` and `run_stream` now share `_retrieval_loop()` — a generator that yields status
events and returns `(hits, queries_tried, used_web)`. `run_stream` does `yield from` to
surface progress; `run` drains it and reads `StopIteration.value`. Retrieval, reflection,
retry and tool logic therefore exist once. The steps around it (guardrails, memory,
rewrite, generation, consolidation) are still duplicated between the two.

[agent/multi_agent.py](agent/multi_agent.py) is unreferenced dead code (see the bug note
below).

## Gotchas that will bite

**`HybridRetriever.retrieve()` returns `(list[RetrievedChunk], rewritten_query)`** despite a
return annotation of `list[RetrievedChunk]`. Callers must unpack.
[multi_agent.py:37](agent/multi_agent.py:37) does not, so that file is broken as written.

**Two independent, uncoordinated LRU caches wrap retrieve+rerank.** `build_cached_retriever()`
is called once in [graph.py:107](agent/graph.py:107) (bound to `cached_retrieve_and_rerank` in
`api/app.py`) and again in [orchestrator.py:78](agent/orchestrator.py:78)
(`self.cached_retriever`). The ingest and delete paths in `api/app.py` only call
`cached_retrieve_and_rerank.cache_clear()` — the orchestrator's cache is never invalidated, so
a repeated query can return pre-ingest results until restart. If you touch ingestion, clear
both.

**Evidence accumulates across retry attempts.** `_retrieval_loop` keeps a pool keyed by
chunk id with the best score seen, so a chunk found on attempt 1 is not lost when a later
reformulation retrieves worse ones. The judge scores the accumulated pool, since that is
what generation receives.

**Telemetry numbers are estimates, not measurements.**
[observability/tracer.py](observability/tracer.py) is a hand-rolled span logger named after
OpenTelemetry, not the OTel SDK — no exporter, no collector; spans log and drop.
`_estimate_tokens()` in [gateway/llm_gateway.py](gateway/llm_gateway.py) is `len(text) // 4`,
so token counts and every USD figure derived from them are approximations rather than Groq's
reported usage. TTFT in `run()` is literally `generation_ms * 0.35`
([orchestrator.py:250](agent/orchestrator.py:250)).

**Sufficiency is judged by the model, not a threshold.** `_judge_sufficiency()` asks
`metadata_model` whether the retrieved context answers the question and returns a reason.
`self_rag_score_threshold` (0.15) survives only as the fallback when the judge is disabled
or unreachable — and it still compares a raw cross-encoder logit against the RRF score when
`use_reranker` is off, two scales that are not comparable. With no `TAVILY_API_KEY`,
`_execute_web_search_tool` now returns `[]`; it used to fabricate a "simulated" snippet
that then *replaced* the real chunks, turning answerable questions into "I do not have
enough information". Web results are appended to the document chunks, never substituted.

**Ingestion calls the LLM once per chunk.** With `generate_metadata=True`, `MetadataEnricher`
runs over every chunk through `metadata_model` on `metadata_workers` threads. It dominates
ingestion cost; set it false for bulk loads.

**`QueryMetric` rows are written only from the Gradio `chat_fn`**, not from `/query` or
`/query/stream`. The Next.js eval dashboard reads `/api/eval/metrics`, so API-only traffic
leaves it empty.

**`README.md` documents `POST /api/db/raw_query`; the route is `GET`** with a `query_str`
query parameter ([api/db_router.py](api/db_router.py)).

**[client.py](client.py) reads `s["rerank_score"]` from source dicts, but the orchestrator
emits `"score"`** — so the CLI always prints `score: 0.000`.

**Guardrails are regex substring matches** ([guardrails/guards.py](guardrails/guards.py)) and
the out-of-scope list includes broad medical and security words (`pain`, `treatment`, `hack`,
`vulnerability`). Ordinary questions containing them are rejected outright. `OutputGuard` is
defined and instantiated but never actually called in the orchestrator.

## Data and persistence

- `qdrant_db/` — vectors plus chunk payloads. Gitignored. This is the source of truth for
  chunks; both `all_chunks` and the BM25 index are rebuilt from it at startup.
- `metrics.db` — SQLite holding `query_metrics`, `eval_runs`, and the memory tables
  (`working_sessions`, `semantic_facts`, `episodic_traces`, `procedural_strategies`,
  `document_chunks`). Schema in [db/models.py](db/models.py). `init_db()` only runs
  `create_all` — there are no migrations, so a changed column means deleting the file or
  migrating by hand.
- `docs/` is gitignored, so anything written there stays local.
- Sources are deduplicated by name: re-ingesting an existing `source` returns 409. Delete it
  first via `DELETE /source/{name}`.

## Conventions

- Commits follow Conventional Commits with a scope: `feat(agent):`, `docs(readme):`,
  `chore(config):`.
- Settings live in `config.py` as `Settings` fields with an inline comment giving units and
  rationale. Read them via `get_settings()` (LRU-cached), never `os.getenv`.
- Ported modules carry a header docstring naming the `mini_rag.py` function they mirror.
  Preserve that note when editing them.
- Device selection for both the embedder and the reranker goes through
  `_best_device()` in [retrieval/embedder.py](retrieval/embedder.py) — Intel XPU, then CUDA,
  then CPU.
- `eval-dashboard/` is a separate Next.js 16 / React 19 app with its own
  [CLAUDE.md](eval-dashboard/CLAUDE.md); follow that file when working inside it.
