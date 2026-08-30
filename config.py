from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # ── API Keys ───────────────────────────────────────────────────────────────
    groq_api_key: str = ""
    tavily_api_key: str = ""

    # ── Models ─────────────────────────────────────────────────────────────────
    embedding_model: str  = "BAAI/bge-m3"
    # TinyBERT is 4x faster than MiniLM-L-6 with minimal quality loss
    reranker_model: str   = "cross-encoder/ms-marco-TinyBERT-L-2-v2"
    # NOTE: llama-3.3-70b-versatile and llama-3.1-8b-instant were retired by
    # Groq and now 404. Verify against GET /v1/models before changing these.
    llm_model: str        = "openai/gpt-oss-120b"  # main generation model
    metadata_model: str   = "qwen/qwen3.8-27b"     # fast model for metadata + query rewrite.
                                                   # Chosen over gpt-oss-20b, which spends its
                                                   # token budget on reasoning and returns
                                                   # truncated, unparseable JSON.

    # ── Agent loop ─────────────────────────────────────────────────────────────
    max_retrieval_attempts: int = 3     # reformulate-and-retry rounds before the
                                        # web tool is considered. 1 = old one-shot
                                        # behaviour.
    use_llm_sufficiency_judge: bool = True   # ask the fast model whether the
                                             # retrieved context answers the
                                             # question. False falls back to the
                                             # raw reranker-score threshold below.
    self_rag_score_threshold: float = 0.15   # only used when the judge is off or
                                             # unreachable. NOTE: this compares
                                             # against a raw cross-encoder logit
                                             # (range roughly ±11), not a 0-1
                                             # probability, and against the RRF
                                             # score when the reranker is off —
                                             # two scales that are not comparable.

    # ── Guardrails ─────────────────────────────────────────────────────────────
    use_prompt_guard: bool = True   # Meta Llama Prompt Guard 2 for injection
                                    # detection. Adds ~390ms per query; set false
                                    # to fall back to the narrow regex list alone.
    prompt_guard_model: str = "meta-llama/llama-prompt-guard-2-86m"
    prompt_guard_threshold: float = 0.5   # measured: benign ~0.001, attacks ~0.9995
    prompt_guard_timeout_s: float = 5.0

    # ── Retrieval ──────────────────────────────────────────────────────────────
    top_k_retrieval: int  = 15     # candidates passed to reranker
    top_k_rerank: int     = 5      # final chunks sent to LLM
    rrf_k: int            = 60     # Reciprocal Rank Fusion constant
    use_reranker: bool    = True
    cache_size: int       = 1024   # LRU cache entries for retrieve+rerank

    # ── Chunking (semantic) ────────────────────────────────────────────────────
    chunk_size: int       = 1000   # max characters of NEW content per chunk
                                   # (overlap is added on top, so a finished
                                   #  chunk reaches ~1.2x this)
    chunk_overlap_ratio: float = 0.175  # tail of the previous chunk carried into
                                        # the next, as a fraction of chunk content.
                                        # Honoured as a 15-20% band.
    min_chunk_chars: int  = 250    # below this a chunk is merged into a neighbour
    semantic_breakpoint_percentile: int = 95   # split where sentence-to-sentence
                                               # distance is in the top 5%
    semantic_buffer_size: int = 1  # neighbours blended into each sentence before
                                   # embedding, to damp single-sentence noise

    # ── Ingestion ──────────────────────────────────────────────────────────────
    generate_metadata: bool = True
    metadata_workers: int   = 2    # parallel LLM threads. Kept low because the
                                   # binding limit is tokens-per-minute, not
                                   # concurrency — more workers just cause 429s.
    embed_batch_size: int   = 256  # BGE-M3 batch size (tuned for 8 GB VRAM)

    # ── Generation ─────────────────────────────────────────────────────────────
    max_context_chars: int  = 12000
    max_tokens: int         = 512
    metadata_max_tokens: int = 512   # headroom so a long JSON reply is never
                                     # truncated into unparseable garbage
    metadata_max_retries: int = 5    # rate-limit retries per chunk

    # ── Memory ─────────────────────────────────────────────────────────────────
    memory_maxlen: int     = 50    # max turns per session before eviction

    # ── Vector Store (Qdrant) ──────────────────────────────────────────────────
    qdrant_path: str       = "qdrant_db"
    qdrant_collection: str = "chunks"

    # ── Legacy FAISS paths (kept for reference, not used) ─────────────────────
    faiss_index_path: str  = "./storage/faiss.index"
    bm25_index_path: str   = "./storage/bm25.pkl"

    class Config:
        env_file = ".env"
        extra = "ignore"   # silently ignore unknown .env keys (e.g. legacy supabase_* fields)


@lru_cache
def get_settings() -> Settings:
    return Settings()
