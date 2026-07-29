"""
Production-Grade Agentic RAG Orchestrator — Implements a modular ReAct planning loop
with 3-tier memory integration, hybrid retrieval, cross-encoder reranking, Self-RAG reflection,
tool execution (web search fallback), resilient LLM gateway routing, and granular latency telemetry.
"""
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
        """Invokes external Tavily Web Search API if configured, otherwise falls back to simulation."""
        tavily_key = settings.tavily_api_key
        if not tavily_key:
            logger.info(f"Tavily API key not configured. Using simulated Web Search fallback for query '{query}'")
            return [
                {
                    "source": "Web Search (Tavily API - Simulated)",
                    "content": f"Simulated web search snippet for: {query}. Enterprise RAG environments utilize real-time search APIs for dynamic fallback queries."
                }
            ]

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

        # Fallback to simulated result if the API call failed
        return [
            {
                "source": "Web Search (Tavily API Failover)",
                "content": f"Tavily web search lookup failed. Using simulated fallback snippet for query: {query}"
            }
        ]


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

        # ── Step 4: Hybrid RAG Retrieval (Dense + Sparse) & Rerank Latency ──────
        t_ret = time.time()
        span_ret = tracer.start_span("hybrid_retrieval", trace_id=trace_id, parent_id=root_span.span_id)
        
        # Track dense vs sparse search latencies
        t_dense_start = time.time()
        q_vec = self.retriever.embedder.embed_query(rewritten_query)
        dense_results = self.retriever.vector_store.search(q_vec, settings.top_k_retrieval)
        timings["dense_search_ms"] = round((time.time() - t_dense_start) * 1000, 2)

        t_sparse_start = time.time()
        sparse_results = self.retriever.bm25_store.search(rewritten_query, settings.top_k_retrieval)
        timings["sparse_search_ms"] = round((time.time() - t_sparse_start) * 1000, 2)

        hits = list(self.cached_retriever(rewritten_query))
        timings["retrieval_total_ms"] = round((time.time() - t_ret) * 1000, 2)
        span_ret.finish()
        plan_steps.append("hybrid_retrieval_complete")

        # ── Step 5: Self-RAG Relevance Evaluation & Tool Fallback Latency ────
        t_eval = time.time()
        span_eval = tracer.start_span("self_rag_eval", trace_id=trace_id, parent_id=root_span.span_id)
        top_score = hits[0].rerank_score if hits and settings.use_reranker else (hits[0].rrf_score if hits else 0.0)
        
        tool_latency_ms = 0.0
        if top_score < 0.15:
            t_tool = time.time()
            tool_res = self._execute_web_search_tool(rewritten_query)
            tool_calls.append({"tool": "tavily_web_search", "query": rewritten_query, "results_count": len(tool_res)})
            tool_latency_ms = round((time.time() - t_tool) * 1000, 2)
            plan_steps.append("tool_web_search_executed")
        else:
            plan_steps.append("self_rag_verified")
            
        timings["self_rag_ms"] = round((time.time() - t_eval) * 1000, 2)
        timings["tool_execution_ms"] = tool_latency_ms
        span_eval.finish()

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

        # ── Step 4: Hybrid RAG Retrieval (Dense + Sparse) & Rerank Latency ──────
        yield {"type": "status", "step": "retrieval", "msg": "📡 Searching document database (Dense + Sparse RRF)..."}
        t_ret = time.time()
        span_ret = tracer.start_span("hybrid_retrieval", trace_id=trace_id, parent_id=root_span.span_id)
        
        # Track dense vs sparse search latencies
        t_dense_start = time.time()
        q_vec = self.retriever.embedder.embed_query(rewritten_query)
        dense_results = self.retriever.vector_store.search(q_vec, settings.top_k_retrieval)
        timings["dense_search_ms"] = round((time.time() - t_dense_start) * 1000, 2)

        t_sparse_start = time.time()
        sparse_results = self.retriever.bm25_store.search(rewritten_query, settings.top_k_retrieval)
        timings["sparse_search_ms"] = round((time.time() - t_sparse_start) * 1000, 2)

        hits = list(self.cached_retriever(rewritten_query))
        timings["retrieval_total_ms"] = round((time.time() - t_ret) * 1000, 2)
        span_ret.finish()
        plan_steps.append("hybrid_retrieval_complete")

        # ── Step 5: Self-RAG Relevance Evaluation & Tool Fallback Latency ────
        yield {"type": "status", "step": "self_rag", "msg": "🤖 Evaluating context relevance (Self-RAG)..."}
        t_eval = time.time()
        span_eval = tracer.start_span("self_rag_eval", trace_id=trace_id, parent_id=root_span.span_id)
        top_score = hits[0].rerank_score if hits and settings.use_reranker else (hits[0].rrf_score if hits else 0.0)
        
        tool_latency_ms = 0.0
        used_web_search = False
        web_hits = []
        if top_score < 0.15:
            yield {"type": "status", "step": "web_search", "msg": "🌐 Low context confidence. Invoking Tavily Web Search fallback..."}
            t_tool = time.time()
            tool_res = self._execute_web_search_tool(rewritten_query)
            used_web_search = True
            tool_calls.append({"tool": "tavily_web_search", "query": rewritten_query, "results_count": len(tool_res)})
            tool_latency_ms = round((time.time() - t_tool) * 1000, 2)
            plan_steps.append("tool_web_search_executed")
            
            # Map tool_res to RetrievedChunks to construct context easily
            for r_idx, r in enumerate(tool_res):
                web_hits.append(
                    RetrievedChunk(
                        chunk=Chunk(
                            id=f"web-{r_idx}",
                            content=r["content"],
                            source=r["source"],
                            heading="Web Search Result"
                        ),
                        rrf_score=1.0,
                        rerank_score=1.0
                    )
                )
        else:
            plan_steps.append("self_rag_verified")
            
        timings["self_rag_ms"] = round((time.time() - t_eval) * 1000, 2)
        timings["tool_execution_ms"] = tool_latency_ms
        span_eval.finish()

        # Assemble Context
        context_hits = web_hits if used_web_search else hits
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

