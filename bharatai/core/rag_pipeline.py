"""
BharatAI — End-to-End Agentic RAG Pipeline
Orchestrates the 3 JEV Decision Layers, Multi-Source Retrieval, Groq Generation,
and Quality Gate with full latency and decision tracing.
"""

import time
import logging
from typing import Dict, Any, Optional, List

from core.tracing import QueryTrace
from core.source_selection import get_source_selection_agent
from core.retriever import get_retriever
from core.relevance_filter import get_relevance_filter
from core.answer_generator import get_answer_generator
from core.quality_gate import get_quality_gate
from core.embeddings import get_embedding_model

logger = logging.getLogger("bharatai.rag_pipeline")


class BharatAIRAGPipeline:
    def __init__(self):
        self.source_selector = get_source_selection_agent()
        self.retriever = get_retriever()
        self.relevance_filter = get_relevance_filter()
        self.answer_generator = get_answer_generator()
        self.quality_gate = get_quality_gate()
        self.embedding_model = get_embedding_model()

    def run(
        self,
        query: str,
        user_selected_sites: Optional[List[str]] = None,
        retrieval_mode: Optional[str] = None,
        top_k: int = 5,
        relevance_threshold: float = 0.30,
        response_language: str = "en",
        enable_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute full RAG workflow with 3 JEV decision layers and bounded retry.
        """
        trace = QueryTrace(user_query=query, retrieval_mode=retrieval_mode or "HYBRID")

        # -------------------------------------------------------------
        # STAGE 1: JEV DECISION LAYER 1 — SOURCE & TOOL SELECTION
        # -------------------------------------------------------------
        with trace.time_stage("source_selection"):
            routing_plan = self.source_selector.select_sources(
                query=query,
                user_selected_sites=user_selected_sites,
                forced_mode=retrieval_mode,
            )
            trace.selected_websites = routing_plan["selected_website_ids"]
            trace.retrieval_mode = routing_plan["retrieval_mode"]
            trace.detected_intent = routing_plan["detected_intent"]
            trace.detected_topic = routing_plan["detected_topic"]
            trace.detected_department = routing_plan["detected_department"]
            trace.is_time_sensitive = routing_plan["is_time_sensitive"]
            trace.routing_rationale = routing_plan["rationale"]

        current_query = query
        retry_count = 0
        final_answer = ""
        citation_map = []
        gate_result: Dict[str, Any] = {}

        while retry_count <= self.quality_gate.max_retries:
            # -------------------------------------------------------------
            # STAGE 2: MULTI-SOURCE RETRIEVAL & EMBEDDING
            # -------------------------------------------------------------
            with trace.time_stage("embedding"):
                # Pre-warm query embedding
                _ = self.embedding_model.embed_query(current_query)

            candidates: List[Dict[str, Any]] = []
            if trace.retrieval_mode in ("INDEXED", "HYBRID"):
                with trace.time_stage("vector_search"):
                    indexed_chunks = self.retriever._retrieve_from_index(
                        query=current_query,
                        top_k=top_k * 2,
                        selected_website_ids=trace.selected_websites,
                    )
                    candidates.extend(indexed_chunks)

            if trace.retrieval_mode in ("LIVE", "HYBRID"):
                with trace.time_stage("live_retrieval"):
                    live_chunks = self.retriever._retrieve_live_official_news(
                        query=current_query,
                        selected_website_ids=trace.selected_websites,
                    )
                    candidates.extend(live_chunks)

            trace.retrieved_chunks = candidates

            # -------------------------------------------------------------
            # STAGE 3: JEV DECISION LAYER 2 — RELEVANCE FILTERING
            # -------------------------------------------------------------
            with trace.time_stage("relevance_filter"):
                retained_chunks, filter_metrics = self.relevance_filter.filter_and_rank(
                    candidate_chunks=candidates,
                    query=current_query,
                    relevance_threshold=relevance_threshold,
                    max_chunks=top_k,
                )
                trace.retained_chunks = retained_chunks
                trace.selected_sources = [c.get("source_url", "") for c in retained_chunks if c.get("source_url")]

            # -------------------------------------------------------------
            # STAGE 4: GROQ ANSWER GENERATION
            # -------------------------------------------------------------
            with trace.time_stage("groq_generation"):
                final_answer, citation_map, groq_meta = self.answer_generator.generate_answer(
                    query=query,
                    evidence_chunks=retained_chunks,
                    response_language=response_language,
                )
                trace.groq_model = groq_meta.get("model", "")
                trace.groq_prompt_tokens = groq_meta.get("prompt_tokens", 0)
                trace.groq_completion_tokens = groq_meta.get("completion_tokens", 0)
                trace.groq_total_tokens = groq_meta.get("total_tokens", 0)
                trace.api_status = "ERROR" if "error" in groq_meta else "SUCCESS"
                if "error" in groq_meta:
                    trace.error_category = "GROQ_API_ERROR"

            # -------------------------------------------------------------
            # STAGE 5: JEV DECISION LAYER 3 — QUALITY GATE
            # -------------------------------------------------------------
            with trace.time_stage("quality_gate"):
                gate_result = self.quality_gate.evaluate_answer(
                    query=query,
                    answer=final_answer,
                    evidence_chunks=retained_chunks,
                    citation_map=citation_map,
                    current_retry_count=retry_count,
                    enable_llm_judge=enable_llm_judge,
                )

            trace.final_decision = gate_result["decision"]
            trace.decision_rationale = gate_result["rationale"]

            # Handle RETRY condition
            if gate_result.get("should_retry", False) and retry_count < self.quality_gate.max_retries:
                retry_start = time.perf_counter()
                retry_count += 1
                trace.retry_count = retry_count
                current_query = gate_result.get("reformulated_query") or f"{query} details"
                # Track retry overhead
                trace.record_stage_latency("retry_overhead", (time.perf_counter() - retry_start) * 1000.0)
                logger.info(f"Retrying pipeline with reformulated query: {current_query}")
                continue
            else:
                break

        trace.generated_answer = final_answer
        trace_summary = trace.finish()

        return {
            "request_id": trace.request_id,
            "query": query,
            "answer": final_answer,
            "citations": citation_map,
            "evidence_chunks": trace.retained_chunks,
            "quality_decision": trace.final_decision,
            "decision_rationale": trace.decision_rationale,
            "retrieval_mode": trace.retrieval_mode,
            "selected_websites": trace.selected_websites,
            "trace": trace_summary,
            "quality_metrics": gate_result,
        }


# Singleton pipeline instance
_rag_pipeline = None

def get_rag_pipeline() -> BharatAIRAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = BharatAIRAGPipeline()
    return _rag_pipeline
