"""
Input and output guardrails.

Input guarding is a two-stage check:

  1. A short regex list of unambiguous jailbreak phrasings. Free, instant, and
     it also serves as the offline fallback when the classifier is unreachable.
  2. Meta's Llama Prompt Guard 2 (86M), a model trained specifically to score
     prompt-injection and jailbreak attempts. Costs ~390ms per query.

Why the rewrite: the previous version was regex-only and carried an
OUT_OF_SCOPE_PATTERNS topic blocklist (medical / security vocabulary). Measured
on 16 realistic questions about the indexed documents, it refused 44% of them —
"how does the architecture handle a security vulnerability", "what treatment
does the OCR pipeline apply", and, on a call-centre project, "can the system act
as a call center agent". The blocklist also protected nothing: this system
answers only from the user's own documents, and when a topic is absent the
retriever returns nothing and the model already says it lacks the context.

Prompt Guard 2 scored 0/10 false positives and 5/5 attacks on the same set,
with roughly a thousandfold gap between the two groups (benign ~0.001,
attacks ~0.9995), so the 0.5 threshold sits in open space.

Output guards:
  - Hallucination heuristic on unsupported numbers.
  NOTE: OutputGuard is constructed by the orchestrator but never invoked on the
  generation path; only agent/graph.py wires it in.
"""
import logging
import os
import re
from dataclasses import dataclass

import httpx

from config import get_settings

logger = logging.getLogger("Guardrails")
settings = get_settings()


@dataclass
class GuardResult:
    passed: bool
    reason: str = ""
    score: float | None = None      # classifier probability, when one ran


# Unambiguous jailbreak phrasings only. This list is deliberately narrow — the
# classifier is what does the real work. Anything broad enough to appear in a
# genuine question about the documents does not belong here.
INJECTION_PATTERNS = [
    r"ignore (the )?(previous|all|above|prior) (instructions|prompts?|rules)",
    r"disregard (the )?(previous|all|above|prior)",
    r"forget (everything|all previous|your instructions)",
    r"(reveal|print|repeat|output) (your|the) (system )?(prompt|instructions)",
    r"\bjailbreak\b",
    r"\bDAN mode\b",
]


class InputGuard:
    def __init__(
        self,
        max_query_length: int = 1000,
        use_prompt_guard: bool | None = None,
    ):
        self.max_len = max_query_length
        self.injection_re = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)
        self.use_prompt_guard = (
            settings.use_prompt_guard if use_prompt_guard is None else use_prompt_guard
        )
        self.api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")

    # ── Classifier ────────────────────────────────────────────────────────────

    def _classify(self, query: str) -> float | None:
        """Return P(injection) in [0, 1], or None when the classifier could not
        be reached. Prompt Guard 2 returns the probability as its message
        content, e.g. "0.9995748400688171"."""
        if not self.use_prompt_guard or not self.api_key:
            return None
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": settings.prompt_guard_model,
                    "messages": [{"role": "user", "content": query}],
                },
                timeout=settings.prompt_guard_timeout_s,
            )
            if resp.status_code != 200:
                logger.warning(
                    f"Prompt Guard returned {resp.status_code}; falling back to regex only."
                )
                return None
            return float(resp.json()["choices"][0]["message"]["content"].strip())
        except Exception as e:
            logger.warning(f"Prompt Guard unavailable ({e}); falling back to regex only.")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def check(self, query: str) -> GuardResult:
        if len(query) > self.max_len:
            return GuardResult(False, f"Query too long ({len(query)} chars)")

        # Cheap pass first: an obvious attempt is rejected without an API call,
        # and this is also the only line of defence if the classifier is down.
        if self.injection_re.search(query):
            return GuardResult(False, "Prompt injection pattern detected")

        score = self._classify(query)
        if score is not None and score >= settings.prompt_guard_threshold:
            return GuardResult(
                False,
                f"Prompt injection detected (confidence {score:.3f})",
                score=score,
            )

        # Fails open: an unreachable classifier degrades to the regex result
        # above rather than blocking every question during an outage.
        return GuardResult(True, score=score)


class OutputGuard:
    def check(self, answer: str, context_chunks: list[str]) -> GuardResult:
        """
        Basic hallucination check: verify the answer doesn't introduce
        numbers not found in any context chunk.
        """
        answer_numbers = set(re.findall(r"\b\d+\.?\d*\b", answer))
        context_text = " ".join(context_chunks)
        context_numbers = set(re.findall(r"\b\d+\.?\d*\b", context_text))

        unsupported_numbers = answer_numbers - context_numbers
        if len(unsupported_numbers) > 3:
            return GuardResult(
                False,
                f"Answer contains unsupported numbers: {unsupported_numbers}"
            )

        return GuardResult(True)
