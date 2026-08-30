"""
Production-Grade Agentic RAG Orchestrator — Implements a modular ReAct planning loop
with 3-tier memory integration, hybrid retrieval, cross-encoder reranking, Self-RAG reflection,
tool execution (web search fallback), resilient LLM gateway routing, and granular latency telemetry.
"""
import json
import time
import uuid
import logging
from typing import Dict, Any, List, Optional, TypedDict, Generator
from langchain_core.messages import HumanMessage, SystemMessage

from config import get_settings
from models import RetrievedChunk, Chunk
from retrieval.hybrid_retriever import HybridRetriever, build_cached_retriever
from reranking.reranker import Reranker, ContextCompressor
from memory.three_tier_memory import ThreeTierMemoryEngine
from gateway.llm_gateway import ResilientLLMGateway
from guardrails.guards import InputGuard, OutputGuard
from observability.tracer import tracer

logger = logging.getLogger("AgentOrchestrator")
settings = get_settings()

_SYSTEM_PROMPT = (
    "You are an enterprise AI assistant equipped with hybrid retrieval tools and multi-tier memory. "
    "Answer the user's question using ONLY the provided context below. "
    "Be concise, highly accurate, and factual. "
    "Always cite your sources using the provided markers (e.g., [1], [2]). "
    "If the answer is not contained in the context, explicitly state "
    "'I do not have enough information in the context to answer this.' Do not guess."
)


class OrchestratorResponse:
    def __init__(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        trace_id: str,
        timings: Dict[str, float],
        telemetry: Dict[str, Any],
        rewritten_query: str = ""
    ):
        self.answer = answer
        self.sources = sources
        self.trace_id = trace_id
        self.timings = timings
        self.telemetry = telemetry
        self.rewritten_query = rewritten_query

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "trace_id": self.trace_id,
            "timings": self.timings,
            "telemetry": self.telemetry,
            "rewritten_query": self.rewritten_query
        }


class ProductionAgenticOrchestrator:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        compressor: ContextCompressor,
        memory_engine: ThreeTierMemoryEngine,
        llm_gateway: ResilientLLMGateway
    ):
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor
        self.memory = memory_engine
        self.llm_gateway = llm_gateway
        self.input_guard = InputGuard()
        self.output_guard = OutputGuard()
        self.cached_retriever = build_cached_retriever(retriever, reranker)

    def _build_context(self, hits: List[RetrievedChunk]) -> str:
        parts, total = [], 0
        for i, rc in enumerate(hits):
            snippet = rc.chunk.content.strip()
            entry = f"[{i+1}] {snippet}"
            if total + len(entry) > settings.max_context_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    def _execute_web_search_tool(self, query: str) -> List[Dict[str, Any]]:
        """Invokes the Tavily Web Search API if configured.

        Returns [] when no key is set. It previously returned a fabricated
        "simulated" snippet, which was worse than returning nothing: the caller
        replaced the real retrieved chunks with that one invented sentence, so a
        question the documents could answer came back as "I do not have enough
        information in the context".
        """
        tavily_key = settings.tavily_api_key
        if not tavily_key:
            logger.info("Tavily API key not configured — skipping web search tool.")
            return []

        logger.info(f"Invoking Tool: Real Tavily Web Search API for query '{query}'")
        try:
            import httpx
            payload = {
                "api_key": tavily_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 3
            }
            resp = httpx.post("https://api.tavily.com/search", json=payload, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                formatted = []
                for r in results:
                    formatted.append({
                        "source": f"Web Search: {r.get('title', 'Tavily Result')}",
                        "content": r.get("content", "")
                    })
                return formatted if formatted else [{"source": "Tavily Search", "content": "No results found."}]
            else:
                logger.error(f"Tavily API returned status {resp.status_code}: {resp.text}")
        except Exception as e:
            logger.error(f"Error during Tavily API execution: {e}")

        # A failed lookup returns nothing rather than an invented snippet.
        return []

    # ── Reflection ────────────────────────────────────────────────────────────

    def _judge_sufficiency(self, question: str, context: str) -> tuple[bool, str]:
        """Ask the fast model whether the retrieved context can answer the
        question. Returns (sufficient, reason).

        This replaces `top_score < 0.15`, which compared against a raw
        cross-encoder logit — and against the RRF score when the reranker was
        off, a different scale entirely, so toggling the reranker silently
        changed how often the fallback fired.
        """
        if not context.strip():
            return False, "no context was retrieved"

        prompt = (
            "You are checking whether a passage set can answer a question.\n"
            f"Question: {question}\n\n"
            f"Context:\n{context[:4000]}\n\n"
            "Reply with ONLY a JSON object:\n"
            '{"sufficient": true|false, "reason": "<8 words on what is missing>"}'
        )
        try:
            res = self.llm_gateway.invoke(
                [HumanMessage(content=prompt)], max_tokens=200,
                model=settings.metadata_model,
            ).content
            start, end = res.find("{"), res.rfind("}") + 1
            if start != -1 and end != 0:
                data = json.loads(res[start:end])
                return bool(data.get("sufficient")), str(data.get("reason", ""))[:120]
        except Exception as e:
            logger.warning(f"Sufficiency judge failed ({e}); falling back to score threshold.")
        return None, "judge unavailable"

    def _reformulate(self, original: str, tried: List[str], reason: str) -> str:
        """Rewrite the query for another retrieval attempt, told what was missing."""
        prompt = (
            "A document search did not find enough to answer this question.\n"
            f"Question: {original}\n"
            f"Queries already tried: {'; '.join(tried)}\n"
            f"What was missing: {reason}\n\n"
            "Write ONE different search query likely to surface the missing "
            "information. Use different wording and likely synonyms or section "
            "titles. Output ONLY the query."
        )
        try:
            out = self.llm_gateway.invoke(
                [HumanMessage(content=prompt)], max_tokens=120,
                model=settings.metadata_model,
            ).content.strip().strip('"')
            return out or original
        except Exception as e:
            logger.warning(f"Reformulation failed ({e}); reusing original query.")
            return original

    # ── The agent loop ────────────────────────────────────────────────────────

    def _retrieval_loop(self, query: str, timings: Dict[str, float],
                        plan_steps: List[str], tool_calls: List[Dict[str, Any]]):
        """Retrieve, judge, and retry with a reformulated query until the context
        is sufficient or the attempt budget runs out; then optionally call the
        web tool.

        This is a generator so the streaming and non-streaming paths share one
        implementation: `run_stream` does `yield from` to surface progress, and
        `run` drains it silently. Returns (hits, queries_tried, used_web).
        """
        attempts     = max(1, settings.max_retrieval_attempts)
        current      = query
        tried        = []
        # Evidence ACCUMULATES across attempts, keyed by chunk id and keeping the
        # best score seen. Replacing `hits` each round meant a good chunk found on
        # attempt 1 was discarded if a later reformulation retrieved worse ones —
        # observed in testing, where a query answered on attempt 1 went on to
        # retry and ended up carrying the weaker set into generation.
        pool: Dict[str, RetrievedChunk] = {}
        hits         = []
        sufficient   = False
        reason       = ""
        t_loop       = time.time()

        for attempt in range(attempts):
            if attempt > 0:
                yield {"type": "status", "step": "reformulate",
                       "msg": f"🔁 Context insufficient ({reason}). Rephrasing and searching again..."}
                current = self._reformulate(query, tried, reason)
                plan_steps.append(f"query_reformulated_{attempt}")

            tried.append(current)
            yield {"type": "status", "step": "retrieval",
                   "msg": f"📡 Searching document database (attempt {attempt + 1}/{attempts})..."}

            for rc in self.cached_retriever(current):
                prior = pool.get(rc.chunk.id)
                if prior is None or rc.rerank_score > prior.rerank_score:
                    pool[rc.chunk.id] = rc
            hits = sorted(pool.values(), key=lambda x: -x.rerank_score)[:settings.top_k_rerank]
            plan_steps.append(f"retrieval_attempt_{attempt + 1}")

            yield {"type": "status", "step": "self_rag",
                   "msg": "🤖 Evaluating whether the context answers the question..."}
            t_judge = time.time()
            verdict = None
            if settings.use_llm_sufficiency_judge:
                # Judge the accumulated evidence, not just this round's, since
                # that is what generation will actually see.
                verdict, reason = self._judge_sufficiency(query, self._build_context(hits))

            if verdict is None:
                # Judge disabled or unreachable — fall back to the score threshold.
                top = hits[0].rerank_score if hits and settings.use_reranker else (
                      hits[0].rrf_score if hits else 0.0)
                verdict = top >= settings.self_rag_score_threshold
                reason  = f"top score {top:.2f} below threshold"

            # Cumulative reflection cost across every attempt in this loop.
            timings["self_rag_ms"] = round(
                timings.get("self_rag_ms", 0.0) + (time.time() - t_judge) * 1000, 2
            )

            if verdict:
                sufficient = True
                plan_steps.append("context_verified")
                break

        timings["retrieval_loop_ms"] = round((time.time() - t_loop) * 1000, 2)
        timings["retrieval_attempts"] = len(tried)

        # ── Tool: web search, only after retries are exhausted ────────────────
        used_web = False
        if not sufficient:
            t_tool = time.time()
            yield {"type": "status", "step": "web_search",
                   "msg": "🌐 Documents insufficient. Trying web search..."}
            web = self._execute_web_search_tool(query)
            timings["tool_execution_ms"] = round((time.time() - t_tool) * 1000, 2)
            if web:
                used_web = True
                tool_calls.append({"tool": "tavily_web_search", "query": query,
                                   "results_count": len(web)})
                plan_steps.append("tool_web_search_executed")
                # Web results are APPENDED to the document chunks, not swapped in.
                # Replacing them meant a weak-but-correct local answer was thrown
                # away in favour of whatever the tool returned.
                for i, r in enumerate(web):
                    hits.append(RetrievedChunk(
                        chunk=Chunk(id=f"web-{i}", content=r["content"],
                                    source=r["source"], heading="Web Search Result"),
                        rrf_score=1.0, rerank_score=1.0,
                    ))
            else:
                plan_steps.append("web_search_unavailable")
                yield {"type": "status", "step": "web_search",
                       "msg": "⚠️ No web search configured — answering from documents alone."}
        else:
            timings.setdefault("tool_execution_ms", 0.0)

        return hits, tried, used_web


    def run(self, query: str, session_id: str = "default_session", user_id: str = "default_user") -> OrchestratorResponse:
        """Complete step-by-step execution trajectory with granular latency tracking."""
        t0 = time.time()
        trace_id = f"tr-{uuid.uuid4().hex[:8]}-exec"
        root_span = tracer.start_span("orchestrator.run", trace_id=trace_id)
        
        timings = {}
        plan_steps = []
        tool_calls = []

        # ── Step 1: Input Security Guardrails Latency ─────────────────────────
        t_guard = time.time()
        span_guard = tracer.start_span("security_guardrails", trace_id=trace_id, parent_id=root_span.span_id)
        g_res = self.input_guard.check(query)
        timings["guardrails_ms"] = round((time.time() - t_guard) * 1000, 2)
        span_guard.finish()

        if not g_res.passed:
            answer = f"I cannot process this request. Security policy violation: {g_res.reason}"
            total_latency = round((time.time() - t0) * 1000, 2)
            timings["total_latency_ms"] = total_latency
            root_span.finish()

            self.memory.consolidate_turn(
                session_id=session_id,
                trace_id=trace_id,
                user_query=query,
                assistant_response=answer,
                plan_steps=["input_guard", "rejected"],
                tool_calls=[],
                latency_ms=int(total_latency),
                prompt_tokens=10,
                completion_tokens=20,
                cost_usd=0.0,
                status="failed"
            )
            return OrchestratorResponse(
                answer=answer,
                sources=[],
                trace_id=trace_id,
                timings=timings,
                telemetry={"prompt_tokens": 10, "completion_tokens": 20, "cost_usd": 0.0, "status": "blocked"}
            )

        plan_steps.append("input_guard_passed")

        # ── Step 2: Fetch 3-Tier Memory Context Latency ────────────────────────
        t_mem = time.time()
        span_mem = tracer.start_span("memory_read", trace_id=trace_id, parent_id=root_span.span_id)
        working_history = self.memory.get_working_history(session_id, last_n=6)
        user_facts = self.memory.get_semantic_facts(user_id)
        timings["memory_read_ms"] = round((time.time() - t_mem) * 1000, 2)
        span_mem.finish()
        plan_steps.append("memory_loaded")

        # ── Step 3: Query Rewriting & Intent Planning Latency ──────────────────
        t_rewrite = time.time()
        span_rw = tracer.start_span("query_rewriting", trace_id=trace_id, parent_id=root_span.span_id)
        if working_history.strip():
            rewritten_query = self.retriever.query_rewriter.rewrite(query, working_history)
        else:
            rewritten_query = query
        timings["rewrite_ms"] = round((time.time() - t_rewrite) * 1000, 2)
        span_rw.finish()
        plan_steps.append("query_rewritten")

        # ── Step 4+5: Retrieve → judge → reformulate → retry → optional tool ───
        # Shared with run_stream() via the same generator, so the two paths
        # cannot drift apart. Status events are drained and discarded here.
        t_ret = time.time()
        span_ret = tracer.start_span("retrieval_loop", trace_id=trace_id, parent_id=root_span.span_id)

        loop = self._retrieval_loop(rewritten_query, timings, plan_steps, tool_calls)
        try:
            while True:
                next(loop)
        except StopIteration as stop:
            hits, queries_tried, used_web = stop.value

        timings["retrieval_total_ms"] = round((time.time() - t_ret) * 1000, 2)
        span_ret.finish()

        # ── Step 6: Context Assembly & Resilient LLM Generation Latency ──────
        t_gen = time.time()
        span_gen = tracer.start_span("llm_generation", trace_id=trace_id, parent_id=root_span.span_id)
        context_str = self._build_context(hits)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {rewritten_query}\n\nAnswer:"
        
        # Estimate Time To First Token (TTFT) ~ 35% of total LLM time on Groq
        llm_res = self.llm_gateway.invoke([
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ])
        gen_latency_ms = round((time.time() - t_gen) * 1000, 2)
        ttft_ms = round(gen_latency_ms * 0.35, 2)
        
        timings["ttft_ms"] = ttft_ms
        timings["generation_ms"] = gen_latency_ms
        span_gen.finish()
        plan_steps.append("generation_complete")

        # ── Step 7: Formatted Output & Memory Consolidation ────────────────────
        formatted_sources = [
            {
                "id": i + 1,
                "chunk_id": rc.chunk.id,
                "source": rc.chunk.source,
                "heading": rc.chunk.heading,
                "score": round(rc.rerank_score if settings.use_reranker else rc.rrf_score, 3),
                "content": rc.chunk.content
            }
            for i, rc in enumerate(hits)
        ]

        # Memory Consolidation Latency
        t_cons = time.time()
        total_latency = round((time.time() - t0) * 1000, 2)
        timings["total_latency_ms"] = total_latency

        self.memory.consolidate_turn(
            session_id=session_id,
            trace_id=trace_id,
            user_query=query,
            assistant_response=llm_res.content,
            plan_steps=plan_steps,
            tool_calls=tool_calls,
            latency_ms=int(total_latency),
            prompt_tokens=llm_res.prompt_tokens,
            completion_tokens=llm_res.completion_tokens,
            cost_usd=llm_res.cost_usd,
            status="recovered" if llm_res.fallback_triggered else "success"
        )
        timings["memory_consolidation_ms"] = round((time.time() - t_cons) * 1000, 2)
        root_span.finish()

        telemetry = {
            "prompt_tokens": llm_res.prompt_tokens,
            "completion_tokens": llm_res.completion_tokens,
            "cost_usd": llm_res.cost_usd,
            "model_used": llm_res.model_used,
            "fallback_triggered": llm_res.fallback_triggered,
            "status": "success"
        }

        return OrchestratorResponse(
            answer=llm_res.content,
            sources=formatted_sources,
            trace_id=trace_id,
            timings=timings,
            telemetry=telemetry,
            rewritten_query=rewritten_query
        )

    def run_stream(self, query: str, session_id: str = "default_session", user_id: str = "default_user") -> Generator[Dict[str, Any], None, None]:
        """Complete streaming execution trajectory with intermediate reasoning step updates."""
        t0 = time.time()
        trace_id = f"tr-{uuid.uuid4().hex[:8]}-stream"
        root_span = tracer.start_span("orchestrator.run_stream", trace_id=trace_id)
        
        timings = {}
        plan_steps = []
        tool_calls = []

        # ── Step 1: Input Security Guardrails Latency ─────────────────────────
        yield {"type": "status", "step": "guardrails", "msg": "🛡️ Checking security guardrails..."}
        t_guard = time.time()
        span_guard = tracer.start_span("security_guardrails", trace_id=trace_id, parent_id=root_span.span_id)
        g_res = self.input_guard.check(query)
        timings["guardrails_ms"] = round((time.time() - t_guard) * 1000, 2)
        span_guard.finish()

        if not g_res.passed:
            answer = f"I cannot process this request. Security policy violation: {g_res.reason}"
            total_latency = round((time.time() - t0) * 1000, 2)
            timings["total_latency_ms"] = total_latency
            root_span.finish()

            self.memory.consolidate_turn(
                session_id=session_id,
                trace_id=trace_id,
                user_query=query,
                assistant_response=answer,
                plan_steps=["input_guard", "rejected"],
                tool_calls=[],
                latency_ms=int(total_latency),
                prompt_tokens=10,
                completion_tokens=20,
                cost_usd=0.0,
                status="failed"
            )
            yield {
                "type": "error",
                "error": "Guardrail rejection",
                "content": answer,
                "timings": timings,
                "telemetry": {"prompt_tokens": 10, "completion_tokens": 20, "cost_usd": 0.0, "status": "blocked"}
            }
            return

        plan_steps.append("input_guard_passed")

        # ── Step 2: Fetch 3-Tier Memory Context Latency ────────────────────────
        yield {"type": "status", "step": "memory", "msg": "🧠 Retrieving working memory history & facts..."}
        t_mem = time.time()
        span_mem = tracer.start_span("memory_read", trace_id=trace_id, parent_id=root_span.span_id)
        working_history = self.memory.get_working_history(session_id, last_n=6)
        user_facts = self.memory.get_semantic_facts(user_id)
        timings["memory_read_ms"] = round((time.time() - t_mem) * 1000, 2)
        span_mem.finish()
        plan_steps.append("memory_loaded")

        # ── Step 3: Query Rewriting & Intent Planning Latency ──────────────────
        yield {"type": "status", "step": "rewrite", "msg": "🔍 Expanding and rewriting query context..."}
        t_rewrite = time.time()
        span_rw = tracer.start_span("query_rewriting", trace_id=trace_id, parent_id=root_span.span_id)
        if working_history.strip():
            rewritten_query = self.retriever.query_rewriter.rewrite(query, working_history)
        else:
            rewritten_query = query
        timings["rewrite_ms"] = round((time.time() - t_rewrite) * 1000, 2)
        span_rw.finish()
        plan_steps.append("query_rewritten")

        # ── Step 4+5: Retrieve → judge → reformulate → retry → optional tool ───
        # Same generator as run(); `yield from` surfaces each attempt to the UI.
        t_ret = time.time()
        span_ret = tracer.start_span("retrieval_loop", trace_id=trace_id, parent_id=root_span.span_id)

        hits, queries_tried, used_web_search = yield from self._retrieval_loop(
            rewritten_query, timings, plan_steps, tool_calls
        )

        timings["retrieval_total_ms"] = round((time.time() - t_ret) * 1000, 2)
        span_ret.finish()

        # Assemble Context
        # `hits` already carries any web results, appended by the retrieval loop
        # rather than substituted for the document chunks.
        context_hits = hits
        context_str = self._build_context(context_hits)
        user_prompt = f"Context:\n{context_str}\n\nQuestion: {rewritten_query}\n\nAnswer:"

        # ── Step 6: Context Assembly & Resilient LLM Generation Latency ──────
        yield {"type": "status", "step": "generation", "msg": "✍️ Synthesizing response and streaming tokens..."}
        t_gen = time.time()
        span_gen = tracer.start_span("llm_generation", trace_id=trace_id, parent_id=root_span.span_id)
        
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]
        
        ttft_ms = None
        full_response_parts = []
        
        # Invoke stream via llm_gateway
        for token in self.llm_gateway.stream(messages):
            if ttft_ms is None:
                ttft_ms = round((time.time() - t_gen) * 1000, 2)
            full_response_parts.append(token)
            yield {"type": "token", "token": token}
            
        gen_latency_ms = round((time.time() - t_gen) * 1000, 2)
        if ttft_ms is None:
            ttft_ms = gen_latency_ms
            
        timings["ttft_ms"] = ttft_ms
        timings["generation_ms"] = gen_latency_ms
        span_gen.finish()
        plan_steps.append("generation_complete")

        # ── Step 7: Formatted Output & Memory Consolidation ────────────────────
        answer = "".join(full_response_parts).strip()
        
        formatted_sources = [
            {
                "id": i + 1,
                "chunk_id": rc.chunk.id,
                "source": rc.chunk.source,
                "heading": rc.chunk.heading,
                "score": round(rc.rerank_score if settings.use_reranker else rc.rrf_score, 3),
                "content": rc.chunk.content
            }
            for i, rc in enumerate(context_hits)
        ]

        # Memory Consolidation Latency
        t_cons = time.time()
        total_latency = round((time.time() - t0) * 1000, 2)
        timings["total_latency_ms"] = total_latency

        # Estimate token usage
        prompt_tokens = sum(self.llm_gateway._estimate_tokens(m.content) for m in messages)
        completion_tokens = self.llm_gateway._estimate_tokens(answer)
        cost_usd = self.llm_gateway._calculate_cost(self.llm_gateway.primary_model, prompt_tokens, completion_tokens)

        self.memory.consolidate_turn(
            session_id=session_id,
            trace_id=trace_id,
            user_query=query,
            assistant_response=answer,
            plan_steps=plan_steps,
            tool_calls=tool_calls,
            latency_ms=int(total_latency),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            status="success"
        )
        timings["memory_consolidation_ms"] = round((time.time() - t_cons) * 1000, 2)
        root_span.finish()

        telemetry = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cost_usd": cost_usd,
            "model_used": self.llm_gateway.primary_model,
            "fallback_triggered": False,
            "status": "success"
        }

        yield {
            "type": "done",
            "answer": answer,
            "sources": formatted_sources,
            "trace_id": trace_id,
            "timings": timings,
            "telemetry": telemetry,
            "rewritten_query": rewritten_query
        }

