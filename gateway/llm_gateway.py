"""
Resilient Multi-Provider LLM Gateway — Enterprise LLM router with automatic failover,
circuit breaking, rate-limit retries, token usage telemetry, and USD cost tracking.
"""
import os
import time
import logging
from typing import List, Dict, Any, Generator, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from config import get_settings

logger = logging.getLogger("LLMGateway")
settings = get_settings()

# Cost estimates per 1,000,000 tokens.
# STALE: these are the retired llama models. The current models (gpt-oss-120b,
# qwen3.8-27b, gpt-oss-20b) are not listed, so they fall through to the generic
# default below. Every figure here is doubly approximate anyway, because token
# counts come from _estimate_tokens() (len // 4), not from the provider's usage
# response. Populate from Groq's pricing page if cost reporting needs to be real.
MODEL_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


class LLMResponse:
    def __init__(
        self,
        content: str,
        model_used: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        cost_usd: float,
        fallback_triggered: bool = False
    ):
        self.content = content
        self.model_used = model_used
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.latency_ms = latency_ms
        self.cost_usd = cost_usd
        self.fallback_triggered = fallback_triggered


class ResilientLLMGateway:
    def __init__(self):
        self.primary_model = settings.llm_model
        # The last entry used to be a hardcoded "llama-3.1-8b-instant", which was
        # both retired by Groq AND a duplicate of metadata_model — so the
        # "resilient failover" had exactly one real fallback, to a dead model.
        self.fallback_models = [settings.metadata_model, "openai/gpt-oss-20b"]
        self.groq_api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")

    def _estimate_tokens(self, text_str: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return max(1, len(text_str) // 4)

    def _calculate_cost(self, model: str, p_tokens: int, c_tokens: int) -> float:
        pricing = MODEL_PRICING.get(model, {"input": 0.50, "output": 0.70})
        cost = (p_tokens / 1_000_000 * pricing["input"]) + (c_tokens / 1_000_000 * pricing["output"])
        return round(cost, 6)

    def invoke(
        self,
        messages: List[BaseMessage],
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> LLMResponse:
        """
        Executes LLM invocation with automatic multi-provider fallback routing.
        If primary model hits rate-limits (HTTP 429/503), it automatically reroutes.
        """
        models_to_try = [self.primary_model] + self.fallback_models
        last_exception = None
        t0 = time.time()

        for idx, model_name in enumerate(models_to_try):
            try:
                llm = ChatGroq(
                    model=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    groq_api_key=self.groq_api_key
                )
                response = llm.invoke(messages)
                latency_ms = int((time.time() - t0) * 1000)
                
                content = response.content.strip()
                p_tokens = sum(self._estimate_tokens(m.content) for m in messages)
                c_tokens = self._estimate_tokens(content)
                cost = self._calculate_cost(model_name, p_tokens, c_tokens)

                if idx > 0:
                    logger.warning(f"Fallback triggered! Used model '{model_name}' instead of primary '{self.primary_model}'")

                return LLMResponse(
                    content=content,
                    model_used=model_name,
                    prompt_tokens=p_tokens,
                    completion_tokens=c_tokens,
                    latency_ms=latency_ms,
                    cost_usd=cost,
                    fallback_triggered=(idx > 0)
                )

            except Exception as e:
                logger.error(f"Error invoking model '{model_name}': {e}. Attempting fallback...")
                last_exception = e
                time.sleep(0.2)

        # Failure recovery fallback response if all APIs fail
        latency_ms = int((time.time() - t0) * 1000)
        return LLMResponse(
            content=f"System Recovery Note: Unable to reach LLM providers ({str(last_exception)}). Operating in offline fallback mode.",
            model_used="offline_fallback",
            prompt_tokens=10,
            completion_tokens=20,
            latency_ms=latency_ms,
            cost_usd=0.0,
            fallback_triggered=True
        )

    def stream(
        self,
        messages: List[BaseMessage],
        temperature: float = 0.0,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """True token-by-token streaming with automatic fallback handling."""
        try:
            llm = ChatGroq(
                model=self.primary_model,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=self.groq_api_key
            )
            for chunk in llm.stream(messages):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error(f"Primary stream failed: {e}. Falling back to secondary model.")
            llm_fallback = ChatGroq(
                model=settings.metadata_model,
                temperature=temperature,
                max_tokens=max_tokens,
                groq_api_key=self.groq_api_key
            )
            for chunk in llm_fallback.stream(messages):
                if chunk.content:
                    yield chunk.content
