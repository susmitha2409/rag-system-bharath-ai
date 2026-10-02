"""
BharatAI — Database Management Layer
Handles SQLite operations, schema initialization, trace logging, website registry persistence,
and metric queries using thread-safe connections and parameterized queries.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from contextlib import contextmanager

from config import DATABASE_PATH, OFFICIAL_WEBSITES

logger = logging.getLogger("bharatai.db")


class DatabaseManager:
    def __init__(self, db_path: str = DATABASE_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Thread-safe context manager for SQLite connections with foreign keys enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=20.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}", exc_info=True)
            raise
        finally:
            conn.close()

    def init_db(self):
        """Execute schema.sql and seed official government websites."""
        schema_path = Path(__file__).resolve().parent / "schema.sql"
        if not schema_path.exists():
            logger.error("schema.sql not found!")
            return

        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()

        with self.get_connection() as conn:
            conn.executescript(schema_sql)

        # Seed initial official websites if not present
        self.seed_official_websites()

    def seed_official_websites(self):
        """Populate websites table with the 10 official Government of India websites."""
        with self.get_connection() as conn:
            for site in OFFICIAL_WEBSITES:
                conn.execute(
                    """
                    INSERT INTO websites (id, name, base_url, canonical_domain, department, description, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(canonical_domain) DO UPDATE SET
                        name=excluded.name,
                        base_url=excluded.base_url,
                        department=excluded.department,
                        description=excluded.description
                    """,
                    (
                        site["id"],
                        site["name"],
                        site["base_url"],
                        site["canonical_domain"],
                        site["department"],
                        site.get("description", ""),
                        1 if site.get("enabled", True) else 0,
                    ),
                )

    # ----------------------------------------------------------------------
    # Websites Management
    # ----------------------------------------------------------------------
    def get_websites(self, enabled_only: bool = False) -> List[Dict[str, Any]]:
        """Retrieve list of registered websites."""
        with self.get_connection() as conn:
            query = "SELECT * FROM websites"
            if enabled_only:
                query += " WHERE enabled = 1"
            query += " ORDER BY name ASC"
            cursor = conn.execute(query)
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_website_by_id(self, website_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve website by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM websites WHERE id = ?", (website_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def update_website_status(self, website_id: str, enabled: bool):
        """Enable or disable a website in the registry."""
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE websites SET enabled = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (1 if enabled else 0, website_id),
            )

    def update_website_crawl_stats(
        self,
        website_id: str,
        status: str,
        pages_discovered: int,
        pages_indexed: int,
        pages_failed: int,
        indexed_chunks: int,
        error_message: Optional[str] = None,
        languages: Optional[List[str]] = None,
    ):
        """Update crawl counters and timestamp for a website."""
        now_str = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            lang_json = json.dumps(languages) if languages else '["en"]'
            conn.execute(
                """
                UPDATE websites SET
                    last_crawl_timestamp = ?,
                    last_crawl_status = ?,
                    pages_discovered = pages_discovered + ?,
                    pages_indexed = pages_indexed + ?,
                    pages_failed = pages_failed + ?,
                    indexed_chunks = indexed_chunks + ?,
                    supported_languages = ?,
                    last_error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    now_str,
                    status,
                    pages_discovered,
                    pages_indexed,
                    pages_failed,
                    indexed_chunks,
                    lang_json,
                    error_message,
                    website_id,
                ),
            )

    def add_custom_website(
        self,
        name: str,
        base_url: str,
        canonical_domain: str,
        department: str,
        description: str = "",
    ) -> bool:
        """Add a new official website to registry."""
        import uuid
        site_id = f"site_{uuid.uuid4().hex[:8]}"
        try:
            with self.get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO websites (id, name, base_url, canonical_domain, department, description, enabled)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                    """,
                    (site_id, name, base_url, canonical_domain, department, description),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    # ----------------------------------------------------------------------
    # Crawled Pages & Documents
    # ----------------------------------------------------------------------
    def record_crawled_page(
        self,
        page_id: str,
        website_id: str,
        url: str,
        title: str,
        status: str,
        content_hash: str,
        doc_type: str = "html",
        language: str = "en",
        has_english: bool = False,
        english_url: Optional[str] = None,
        etag: Optional[str] = None,
        last_modified: Optional[str] = None,
        http_status: int = 200,
        content_length: int = 0,
        error_msg: Optional[str] = None,
    ):
        """Record or update crawled page record."""
        now_str = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO crawled_pages (
                    id, website_id, url, title, status, content_hash, document_type,
                    language, has_english_version, english_url, etag, last_modified,
                    http_status, content_length, crawl_timestamp, error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title=excluded.title,
                    status=excluded.status,
                    content_hash=excluded.content_hash,
                    crawl_timestamp=excluded.crawl_timestamp,
                    error_message=excluded.error_message,
                    http_status=excluded.http_status
                """,
                (
                    page_id,
                    website_id,
                    url,
                    title,
                    status,
                    content_hash,
                    doc_type,
                    language,
                    1 if has_english else 0,
                    english_url,
                    etag,
                    last_modified,
                    http_status,
                    content_length,
                    now_str,
                    error_msg,
                ),
            )

    def record_document(
        self,
        doc_id: str,
        page_id: Optional[str],
        website_id: Optional[str],
        title: str,
        url: str,
        domain: str,
        department: str,
        publication_date: Optional[str],
        doc_type: str,
        source_language: str,
        original_text: str,
        english_text: str,
        is_translated: bool,
        content_hash: str,
        num_chunks: int,
    ):
        """Save a processed document before or after chunking."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, page_id, website_id, title, url, domain, department,
                    publication_date, document_type, source_language, original_text,
                    english_text, is_translated, content_hash, num_chunks
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    page_id,
                    website_id,
                    title,
                    url,
                    domain,
                    department,
                    publication_date,
                    doc_type,
                    source_language,
                    original_text,
                    english_text,
                    1 if is_translated else 0,
                    content_hash,
                    num_chunks,
                ),
            )

    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM documents WHERE url = ?", (url,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id, title, url, domain, department, publication_date, document_type, source_language, is_translated, num_chunks, created_at FROM documents ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(r) for r in cursor.fetchall()]

    def delete_document(self, doc_id: str):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    # ----------------------------------------------------------------------
    # Crawl Jobs and Errors
    # ----------------------------------------------------------------------
    def create_crawl_job(self, job_id: str, website_id: Optional[str], job_type: str = "SINGLE_SITE"):
        now_str = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO crawl_jobs (id, website_id, job_type, status, start_time)
                VALUES (?, ?, ?, 'RUNNING', ?)
                """,
                (job_id, website_id, job_type, now_str),
            )

    def finish_crawl_job(
        self,
        job_id: str,
        status: str,
        discovered: int,
        crawled: int,
        indexed: int,
        failed: int,
        duration: float,
        error_msg: Optional[str] = None,
    ):
        now_str = datetime.utcnow().isoformat()
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE crawl_jobs SET
                    status = ?,
                    pages_discovered = ?,
                    pages_crawled = ?,
                    pages_indexed = ?,
                    pages_failed = ?,
                    duration_seconds = ?,
                    end_time = ?,
                    error_message = ?
                WHERE id = ?
                """,
                (status, discovered, crawled, indexed, failed, duration, now_str, error_msg, job_id),
            )

    def record_crawl_error(
        self,
        job_id: Optional[str],
        website_id: Optional[str],
        url: str,
        error_type: str,
        status_code: Optional[int],
        error_message: str,
    ):
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO crawl_errors (job_id, website_id, url, error_type, status_code, error_message)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (job_id, website_id, url, error_type, status_code, error_message),
            )

    def get_recent_crawl_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM crawl_errors ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    # ----------------------------------------------------------------------
    # Query Tracing & Logs
    # ----------------------------------------------------------------------
    def record_query_log(self, trace_data: Dict[str, Any]):
        """Persist complete query trace into query_logs table."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO query_logs (
                    request_id, timestamp, user_query, retrieval_mode, selected_websites,
                    detected_intent, detected_topic, detected_department, is_time_sensitive,
                    routing_rationale, total_latency_ms, intent_analysis_latency_ms,
                    source_selection_latency_ms, embedding_latency_ms, vector_search_latency_ms,
                    live_retrieval_latency_ms, relevance_filter_latency_ms, groq_generation_latency_ms,
                    quality_gate_latency_ms, retry_count, total_retry_overhead_ms,
                    retrieved_chunk_count, retained_chunk_count, selected_sources,
                    final_decision, decision_rationale, generated_answer,
                    groq_model, groq_prompt_tokens, groq_completion_tokens, groq_total_tokens,
                    api_status, error_category
                )
                VALUES (
                    :request_id, :timestamp, :user_query, :retrieval_mode, :selected_websites,
                    :detected_intent, :detected_topic, :detected_department, :is_time_sensitive,
                    :routing_rationale, :total_latency_ms, :intent_analysis_latency_ms,
                    :source_selection_latency_ms, :embedding_latency_ms, :vector_search_latency_ms,
                    :live_retrieval_latency_ms, :relevance_filter_latency_ms, :groq_generation_latency_ms,
                    :quality_gate_latency_ms, :retry_count, :total_retry_overhead_ms,
                    :retrieved_chunk_count, :retained_chunk_count, :selected_sources,
                    :final_decision, :decision_rationale, :generated_answer,
                    :groq_model, :groq_prompt_tokens, :groq_completion_tokens, :groq_total_tokens,
                    :api_status, :error_category
                )
                """,
                trace_data,
            )

    def record_retrieved_chunks(self, request_id: str, chunks: List[Dict[str, Any]]):
        """Record chunks retrieved for a query."""
        with self.get_connection() as conn:
            for i, chunk in enumerate(chunks):
                conn.execute(
                    """
                    INSERT INTO retrieved_chunks (
                        request_id, chunk_id, document_id, source_url, source_title,
                        source_domain, department, publication_date, similarity_score,
                        rank_order, is_retained, is_live_retrieved, source_language,
                        original_text, english_text, is_translated
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        chunk.get("chunk_id", f"c_{i}"),
                        chunk.get("document_id"),
                        chunk.get("source_url", ""),
                        chunk.get("source_title", ""),
                        chunk.get("source_domain", ""),
                        chunk.get("department", ""),
                        chunk.get("publication_date", ""),
                        chunk.get("similarity_score", 0.0),
                        i + 1,
                        1 if chunk.get("is_retained", True) else 0,
                        1 if chunk.get("is_live_retrieved", False) else 0,
                        chunk.get("source_language", "en"),
                        chunk.get("original_text", ""),
                        chunk.get("english_text", ""),
                        1 if chunk.get("is_translated", False) else 0,
                    ),
                )

    def record_answer_evaluation(self, eval_data: Dict[str, Any]):
        """Record evaluation metrics for a query response."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO answer_evaluations (
                    id, request_id, timestamp, eval_type, faithfulness_score,
                    answer_relevance_score, completeness_score, citation_correctness_score,
                    citation_coverage_score, unsupported_claim_count, context_relevance_score,
                    precision_at_k, recall_at_k, mrr, ndcg, evaluator_model, rationale
                )
                VALUES (
                    :id, :request_id, :timestamp, :eval_type, :faithfulness_score,
                    :answer_relevance_score, :completeness_score, :citation_correctness_score,
                    :citation_coverage_score, :unsupported_claim_count, :context_relevance_score,
                    :precision_at_k, :recall_at_k, :mrr, :ndcg, :evaluator_model, :rationale
                )
                """,
                eval_data,
            )

    def record_user_feedback(self, request_id: str, rating: int, comment: Optional[str] = None):
        """Record thumbs up/down user feedback."""
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_feedback (request_id, rating, comment)
                VALUES (?, ?, ?)
                """,
                (request_id, rating, comment),
            )

    def mark_chunk_relevance(self, chunk_id: str, request_id: str, is_relevant: bool):
        """User feedback on specific retrieved chunk."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE retrieved_chunks SET user_marked_relevant = ?
                WHERE chunk_id = ? AND request_id = ?
                """,
                (1 if is_relevant else 0, chunk_id, request_id),
            )

    # ----------------------------------------------------------------------
    # Query Logs & Analytics Retrieval
    # ----------------------------------------------------------------------
    def get_query_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM query_logs ORDER BY timestamp DESC LIMIT ?", (limit,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_query_trace_by_id(self, request_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM query_logs WHERE request_id = ?", (request_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None
            trace = dict(row)
            # Fetch retrieved chunks
            c_cursor = conn.execute(
                "SELECT * FROM retrieved_chunks WHERE request_id = ? ORDER BY rank_order ASC",
                (request_id,),
            )
            trace["retrieved_chunks"] = [dict(c) for c in c_cursor.fetchall()]
            # Fetch evaluation
            e_cursor = conn.execute(
                "SELECT * FROM answer_evaluations WHERE request_id = ?", (request_id,)
            )
            e_row = e_cursor.fetchone()
            trace["evaluation"] = dict(e_row) if e_row else None
            return trace

    def get_analytics_summary(self) -> Dict[str, Any]:
        """Aggregate stats for the Evaluation Dashboard."""
        with self.get_connection() as conn:
            total_queries = conn.execute("SELECT COUNT(*) FROM query_logs").fetchone()[0]
            if total_queries == 0:
                return {
                    "total_queries": 0,
                    "avg_total_latency_ms": 0,
                    "median_total_latency_ms": 0,
                    "p95_total_latency_ms": 0,
                    "avg_groq_latency_ms": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "accept_count": 0,
                    "retry_count": 0,
                    "abstain_count": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0,
                    "avg_faithfulness": 0.0,
                    "avg_relevance": 0.0,
                    "avg_citation_correctness": 0.0,
                }

            row = conn.execute(
                """
                SELECT
                    AVG(total_latency_ms) as avg_total_lat,
                    AVG(groq_generation_latency_ms) as avg_groq_lat,
                    SUM(CASE WHEN api_status = 'SUCCESS' THEN 1 ELSE 0 END) as success_reqs,
                    SUM(CASE WHEN api_status != 'SUCCESS' THEN 1 ELSE 0 END) as failed_reqs,
                    SUM(CASE WHEN final_decision = 'ACCEPT' THEN 1 ELSE 0 END) as accept_cnt,
                    SUM(CASE WHEN final_decision = 'RETRY' THEN 1 ELSE 0 END) as retry_cnt,
                    SUM(CASE WHEN final_decision = 'ABSTAIN' THEN 1 ELSE 0 END) as abstain_cnt
                FROM query_logs
                """
            ).fetchone()

            # Latency percentiles
            latencies = [
                r[0]
                for r in conn.execute(
                    "SELECT total_latency_ms FROM query_logs ORDER BY total_latency_ms ASC"
                ).fetchall()
            ]
            try:
                import numpy as np
                median_lat = float(np.median(latencies)) if latencies else 0.0
                p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0
            except ImportError:
                n = len(latencies)
                median_lat = float(latencies[n // 2]) if n > 0 else 0.0
                p95_idx = min(int(n * 0.95), n - 1) if n > 0 else 0
                p95_lat = float(latencies[p95_idx]) if n > 0 else 0.0

            # Feedback
            fb = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN rating = 1 THEN 1 ELSE 0 END) as pos,
                    SUM(CASE WHEN rating = -1 THEN 1 ELSE 0 END) as neg
                FROM user_feedback
                """
            ).fetchone()

            # Evaluation scores
            eval_row = conn.execute(
                """
                SELECT
                    AVG(faithfulness_score) as avg_faith,
                    AVG(answer_relevance_score) as avg_rel,
                    AVG(citation_correctness_score) as avg_cite
                FROM answer_evaluations
                """
            ).fetchone()

            return {
                "total_queries": total_queries,
                "avg_total_latency_ms": round(row["avg_total_lat"] or 0, 1),
                "median_total_latency_ms": round(median_lat, 1),
                "p95_total_latency_ms": round(p95_lat, 1),
                "avg_groq_latency_ms": round(row["avg_groq_lat"] or 0, 1),
                "successful_requests": row["success_reqs"] or 0,
                "failed_requests": row["failed_reqs"] or 0,
                "accept_count": row["accept_cnt"] or 0,
                "retry_count": row["retry_cnt"] or 0,
                "abstain_count": row["abstain_cnt"] or 0,
                "positive_feedback": fb["pos"] or 0 if fb else 0,
                "negative_feedback": fb["neg"] or 0 if fb else 0,
                "avg_faithfulness": round(eval_row["avg_faith"] or 0, 2) if eval_row else 0.0,
                "avg_relevance": round(eval_row["avg_rel"] or 0, 2) if eval_row else 0.0,
                "avg_citation_correctness": round(eval_row["avg_cite"] or 0, 2) if eval_row else 0.0,
            }

    # ----------------------------------------------------------------------
    # System Settings Persistence
    # ----------------------------------------------------------------------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT value FROM system_settings WHERE key = ?", (key,))
            row = cursor.fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: str):
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO system_settings (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                """,
                (key, str(value)),
            )


# Singleton instance
_db_instance = None

def get_db() -> DatabaseManager:
    global _db_instance
    if _db_instance is None:
        _db_instance = DatabaseManager()
    return _db_instance
