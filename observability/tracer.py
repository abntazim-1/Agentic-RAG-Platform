"""
OpenTelemetry Distributed Observability & Telemetry Suite — Captures execution spans,
aggregates token metrics, and exports trace logs in OpenTelemetry standard formats.
"""
import time
import uuid
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("ObservabilityTracer")


class Span:
    def __init__(self, name: str, trace_id: str, parent_id: Optional[str] = None):
        self.name = name
        self.trace_id = trace_id
        self.span_id = f"span-{uuid.uuid4().hex[:6]}"
        self.parent_id = parent_id
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.attributes: Dict[str, Any] = {}
        self.events: list = []

    def set_attribute(self, key: str, value: Any):
        self.attributes[key] = value

    def add_event(self, event_name: str, payload: Optional[Dict[str, Any]] = None):
        self.events.append({
            "name": event_name,
            "timestamp": time.time(),
            "payload": payload or {}
        })

    def finish(self):
        self.end_time = time.time()
        latency_ms = round((self.end_time - self.start_time) * 1000, 2)
        logger.info(f"[OTel Span] {self.name} (Trace: {self.trace_id}, Span: {self.span_id}) completed in {latency_ms}ms")


class OpenTelemetryTracer:
    def __init__(self):
        self.active_spans: Dict[str, Span] = {}

    def start_span(self, name: str, trace_id: str, parent_id: Optional[str] = None) -> Span:
        span = Span(name, trace_id, parent_id)
        self.active_spans[span.span_id] = span
        return span

    def finish_span(self, span: Span):
        span.finish()
        self.active_spans.pop(span.span_id, None)


# Global tracer singleton
tracer = OpenTelemetryTracer()
