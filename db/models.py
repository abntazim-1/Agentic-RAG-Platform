from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from datetime import datetime
import uuid

Base = declarative_base()

class QueryMetric(Base):
    __tablename__ = "query_metrics"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    query = Column(String)
    rewritten_query = Column(String, nullable=True)
    answer = Column(String)
    
    # Timings (seconds)
    rewrite_time = Column(Float, nullable=True)
    retrieve_time = Column(Float, nullable=True)
    ttft = Column(Float, nullable=True)  # Time to first token
    generation_time = Column(Float, nullable=True)
    total_time = Column(Float, nullable=True)
    
    # RAGAS scores
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    
    # Extra data (sources retrieved, etc)
    contexts = Column(JSON, nullable=True)
    
    # Optional foreign key to a benchmark run
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=True)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String) # e.g. "Run with ChunkSize=512, Reranker=True"
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    # Aggregated metrics for the run
    avg_faithfulness = Column(Float, nullable=True)
    avg_answer_relevancy = Column(Float, nullable=True)
    avg_latency = Column(Float, nullable=True)
    error_rate = Column(Float, nullable=True)
    
    # Configuration used for this run
    configuration = Column(JSON, nullable=True)
    
    queries = relationship("QueryMetric", backref="run")


# ─── 3-Tier Memory & System Schemas ─────────────────────────────────────────

class ProceduralStrategy(Base):
    __tablename__ = "procedural_strategies"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    query_category = Column(String, index=True, nullable=False)
    description = Column(String, nullable=False)
    workflow_dag = Column(JSON, nullable=False)  # JSON representation of execution DAG
    success_rate = Column(Float, default=1.0)
    avg_latency_ms = Column(Integer, default=0)
    times_executed = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class EpisodicTrace(Base):
    __tablename__ = "episodic_traces"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = Column(String, index=True, nullable=False)
    trace_id = Column(String, unique=True, index=True, nullable=False)
    user_query = Column(String, nullable=False)
    plan_steps = Column(JSON, nullable=False)
    tool_calls = Column(JSON, default=list)
    final_response = Column(String, nullable=False)
    latency_ms = Column(Integer, nullable=False)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    cost_usd = Column(Float, default=0.0)
    status = Column(String, nullable=False, default="success") # 'success', 'recovered', 'failed'
    created_at = Column(DateTime, default=datetime.utcnow)


class SemanticFact(Base):
    __tablename__ = "semantic_facts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, default="default_user")
    subject = Column(String, nullable=False)
    predicate = Column(String, nullable=False)
    object = Column(String, nullable=False)
    confidence = Column(Float, default=1.0)
    source_trace_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_recalled_at = Column(DateTime, default=datetime.utcnow)


class WorkingSession(Base):
    __tablename__ = "working_sessions"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, index=True, default="default_user")
    turns = Column(JSON, default=list)
    scratchpad = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    chunk_id = Column(String, index=True, nullable=False)
    source = Column(String, index=True, nullable=False)
    heading = Column(String, nullable=True)
    content = Column(String, nullable=False)
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)

