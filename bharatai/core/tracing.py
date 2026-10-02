"""
BharatAI — Centralized Tracing and Latency Tracking
Measures stage-by-stage execution time with time.perf_counter(), builds structured traces,
and persists traces to the database.
"""

import time
import uuid
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from database.db import get_db
from core.security import redact_secrets

logger = logging.getLogger("bharatai.tracing")


class StageTimer:
    def __init__(self, trace: "QueryTrace", stage_name: str):
        self.trace = trace
        self.stage_name = stage_name
        self.start_time: float = 0.0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration_ms = (time.perf_counter() - self.start_time) * 1000.0
        self.trace.record_stage_latency(self.stage_name, duration_ms)


class QueryTrace:
    def __init__(self, user_query: str, retrieval_mode: str = "HYBRID", request_id: Optional[str] = None):
        self.request_id = request_id or f"req_{uuid.uuid4().hex[:12]}"
        self.user_query = user_query
        self.retrieval_mode = retrieval_mode
        self.start_perf = time.perf_counter()
        self.timestamp = datetime.utcnow().isoformat()

        # Latency records in milliseconds
        self.latencies: Dict[str, float] = {
            "intent_analysis": 0.0,
            "source_selection": 0.0,
            "embedding": 0.0,
            "vector_search": 0.0,
            "live_retrieval": 0.0,
            "relevance_filter": 0.0,
            "groq_generation": 0.0,
            "quality_gate": 0.0,
            "retry_overhead": 0.0,
        }

        # Query metadata & decisions
        self.selected_websites: List[str] = []
        self.detected_intent: str = "general_query"
        self.detected_topic: str = "general"
        self.detected_department: str = "All"
        self.is_time_sensitive: bool = False
        self.routing_rationale: str = ""

        # Chunk counts & evidence
        self.retrieved_chunks: List[Dict[str, Any]] = []
        self.retained_chunks: List[Dict[str, Any]] = []
        self.selected_sources: List[str] = []

        # JEV Decision Layer 3 & Quality Gate
        self.final_decision: str = "ACCEPT"  # ACCEPT, RETRY, ABSTAIN
        self.decision_rationale: str = ""
        self.retry_count: int = 0
        self.generated_answer: str = ""

        # LLM Usage
        self.groq_model: str = ""
        self.groq_prompt_tokens: int = 0
        self.groq_completion_tokens: int = 0
        self.groq_total_tokens: int = 0
        self.api_status: str = "SUCCESS"
        self.error_category: Optional[str] = None

    def time_stage(self, stage_name: str) -> StageTimer:
        """Context manager to measure latency of a single processing stage."""
        return StageTimer(self, stage_name)

    def record_stage_latency(self, stage_name: str, duration_ms: float):
        self.latencies[stage_name] = self.latencies.get(stage_name, 0.0) + duration_ms

    def finish(self) -> Dict[str, Any]:
        """Finalize total latency measurement and return serializable summary."""
        total_latency_ms = (time.perf_counter() - self.start_perf) * 1000.0

        trace_dict = {
            "request_id": self.request_id,
            "timestamp": self.timestamp,
            "user_query": self.user_query,
            "retrieval_mode": self.retrieval_mode,
            "selected_websites": json.dumps(self.selected_websites),
            "detected_intent": self.detected_intent,
            "detected_topic": self.detected_topic,
            "detected_department": self.detected_department,
            "is_time_sensitive": 1 if self.is_time_sensitive else 0,
            "routing_rationale": redact_secrets(self.routing_rationale),
            "total_latency_ms": round(total_latency_ms, 2),
            "intent_analysis_latency_ms": round(self.latencies.get("intent_analysis", 0.0), 2),
            "source_selection_latency_ms": round(self.latencies.get("source_selection", 0.0), 2),
            "embedding_latency_ms": round(self.latencies.get("embedding", 0.0), 2),
            "vector_search_latency_ms": round(self.latencies.get("vector_search", 0.0), 2),
            "live_retrieval_latency_ms": round(self.latencies.get("live_retrieval", 0.0), 2),
            "relevance_filter_latency_ms": round(self.latencies.get("relevance_filter", 0.0), 2),
            "groq_generation_latency_ms": round(self.latencies.get("groq_generation", 0.0), 2),
            "quality_gate_latency_ms": round(self.latencies.get("quality_gate", 0.0), 2),
            "retry_count": self.retry_count,
            "total_retry_overhead_ms": round(self.latencies.get("retry_overhead", 0.0), 2),
            "retrieved_chunk_count": len(self.retrieved_chunks),
            "retained_chunk_count": len(self.retained_chunks),
            "selected_sources": json.dumps(self.selected_sources),
            "final_decision": self.final_decision,
            "decision_rationale": redact_secrets(self.decision_rationale),
            "generated_answer": redact_secrets(self.generated_answer),
            "groq_model": self.groq_model,
            "groq_prompt_tokens": self.groq_prompt_tokens,
            "groq_completion_tokens": self.groq_completion_tokens,
            "groq_total_tokens": self.groq_total_tokens,
            "api_status": self.api_status,
            "error_category": self.error_category,
        }

        # Persist to SQLite
        try:
            db = get_db()
            db.record_query_log(trace_dict)
            if self.retrieved_chunks:
                db.record_retrieved_chunks(self.request_id, self.retrieved_chunks)
        except Exception as e:
            logger.error(f"Failed to persist query trace {self.request_id}: {e}")

        return trace_dict
