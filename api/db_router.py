"""
Real-Time Database Inspector Router — Provides endpoints to query live database tables,
inspect memory records (Semantic, Episodic, Procedural, Working), and view vector metadata.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Dict, Any, List
import json

from db.database import get_db, engine
from db.models import (
    ProceduralStrategy,
    EpisodicTrace,
    SemanticFact,
    WorkingSession,
    DocumentChunkRecord,
    QueryMetric,
)

router = APIRouter(prefix="/api/db", tags=["Real-Time Database Inspector"])

VALID_TABLES = {
    "procedural_strategies": ProceduralStrategy,
    "episodic_traces": EpisodicTrace,
    "semantic_facts": SemanticFact,
    "working_sessions": WorkingSession,
    "document_chunks": DocumentChunkRecord,
    "query_metrics": QueryMetric,
}


@router.get("/tables")
def list_tables(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Returns a list of all database tables with total row counts."""
    tables_summary = []
    for table_name, model_cls in VALID_TABLES.items():
        try:
            count = db.query(model_cls).count()
        except Exception:
            count = 0
        tables_summary.append({
            "table_name": table_name,
            "row_count": count,
            "description": model_cls.__doc__ or table_name
        })
    return {"status": "success", "tables": tables_summary}


@router.get("/table/{table_name}")
def get_table_data(
    table_name: str,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Query live table rows and return formatted JSON records."""
    if table_name not in VALID_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid table name '{table_name}'. Valid options: {list(VALID_TABLES.keys())}"
        )

    model_cls = VALID_TABLES[table_name]
    total_count = db.query(model_cls).count()
    records = db.query(model_cls).offset(offset).limit(limit).all()

    # Convert SQLAlchemy models to dict
    formatted_records = []
    for r in records:
        record_dict = {}
        for col in r.__table__.columns:
            val = getattr(r, col.name)
            if hasattr(val, "isoformat"):
                val = val.isoformat()
            record_dict[col.name] = val
        formatted_records.append(record_dict)

    return {
        "status": "success",
        "table_name": table_name,
        "total_rows": total_count,
        "limit": limit,
        "offset": offset,
        "columns": [c.name for c in model_cls.__table__.columns],
        "rows": formatted_records,
    }


@router.get("/raw_query")
def raw_sql_query(
    query_str: str = Query(..., description="SELECT SQL query"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Execute read-only SQL SELECT queries for direct inspection."""
    clean_query = query_str.strip()
    if not clean_query.lower().startswith("select"):
        raise HTTPException(status_code=400, detail="Only read-only SELECT queries are allowed.")

    try:
        result = db.execute(text(clean_query))
        keys = result.keys()
        rows = [dict(zip(keys, row)) for row in result.fetchall()]
        return {
            "status": "success",
            "query": clean_query,
            "row_count": len(rows),
            "rows": rows
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Database Query Error: {str(e)}")
