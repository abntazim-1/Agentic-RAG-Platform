"""
Metadata enrichment — port from mini_rag.py.

Key changes:
  - Uses metadata_model (llama-3.1-8b-instant, fast/cheap) not the main LLM
  - Parallel batch enrichment via ThreadPoolExecutor (matches mini_rag.py's
    process_document() ThreadPoolExecutor approach)
  - JSON-based output format (matching mini_rag.py's generate_metadata_for_chunk)
"""
import json
import time
import os
import re
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from config import get_settings
from models import Chunk

settings = get_settings()


class MetadataEnricher:
    def __init__(self):
        # Fast metadata model — matches mini_rag.py's metadata_llm
        # Pass the key from settings rather than relying on the ambient
        # GROQ_API_KEY env var — otherwise this only constructs successfully
        # when something called load_dotenv() first, which is true inside
        # api/app.py but not in scripts or tests.
        self.llm = ChatGroq(
            model=settings.metadata_model,
            temperature=0,
            max_tokens=settings.metadata_max_tokens,
            groq_api_key=settings.groq_api_key or os.getenv("GROQ_API_KEY", ""),
        )

    def _invoke_with_retry(self, prompt: str) -> str:
        """Call the metadata model, waiting out rate limits instead of dropping
        the chunk.

        Without this, a burst of chunks exceeds the account's tokens-per-minute
        allowance and the excess calls return HTTP 429. The old code treated
        that as a permanent failure, so on a 484-chunk corpus 257 chunks ended
        up with no summary, keywords, or questions — silently, and the API still
        reported the ingest as a success.
        """
        last: Exception | None = None
        for attempt in range(settings.metadata_max_retries):
            try:
                return self.llm.invoke([HumanMessage(content=prompt)]).content
            except Exception as e:
                last = e
                text = str(e)
                if "429" not in text and "rate limit" not in text.lower():
                    raise           # a real error — do not burn retries on it
                # Groq states the wait in the message ("try again in 6.5s");
                # honour it when present, otherwise back off exponentially.
                m = re.search(r"try again in ([\d.]+)s", text)
                delay = float(m.group(1)) + 0.5 if m else min(2 ** attempt, 30)
                time.sleep(delay)
        raise last if last else RuntimeError("metadata invoke failed")

    def enrich(self, chunk: Chunk) -> Chunk:
        """
        Adds summary, keywords, and synthetic questions to a chunk.
        Uses JSON output format from mini_rag.py's generate_metadata_for_chunk().
        """
        if len(chunk.content) < 30:
            return chunk

        prompt = (
            f"Analyze this text chunk (Heading: {chunk.heading or 'none'}):\n"
            f"{chunk.content}\n\n"
            "Return ONLY a JSON object with this exact structure:\n"
            '{"summary": "A 1-sentence summary", '
            '"keywords": ["keyword1", "keyword2"], '
            '"questions": ["Question 1?", "Question 2?"]}'
        )
        try:
            res = self._invoke_with_retry(prompt)
            start = res.find("{")
            end   = res.rfind("}") + 1
            if start != -1 and end != 0:
                data = json.loads(res[start:end])
                chunk.summary   = data.get("summary", "")
                chunk.keywords  = data.get("keywords", [])
                chunk.questions = data.get("questions", [])
        except Exception as e:
            # Headings routinely contain characters the Windows cp1252 console
            # cannot encode. Printing them raw made the error handler itself
            # raise UnicodeEncodeError, which propagated out of the thread pool
            # and killed the whole ingestion run over one bad chunk.
            msg = f"[MetadataEnricher] Failed for chunk '{chunk.heading}': {e}"
            print(msg.encode("ascii", "replace").decode("ascii"))
        return chunk

    def enrich_batch(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Parallel enrichment using ThreadPoolExecutor.
        Matches mini_rag.py's process_document() ThreadPoolExecutor approach.
        Respects settings.metadata_workers and settings.generate_metadata.
        """
        if not settings.generate_metadata:
            print(
                f"[MetadataEnricher] Skipping LLM metadata — fast ingestion mode. "
                f"({len(chunks)} chunks, set GENERATE_METADATA=True to enable enrichment)"
            )
            return chunks

        print(f"[MetadataEnricher] Generating metadata for {len(chunks)} chunks "
              f"using {settings.metadata_workers} workers...")
        with ThreadPoolExecutor(max_workers=settings.metadata_workers) as executor:
            enriched = list(executor.map(self.enrich, chunks))
        return enriched
