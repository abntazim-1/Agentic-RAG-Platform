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
