-- BharatAI SQLite Database Schema
-- Stores websites, crawled pages, documents, query traces, evaluations, and feedback

-- 1. Websites Registry
CREATE TABLE IF NOT EXISTS websites (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    base_url TEXT NOT NULL,
    canonical_domain TEXT NOT NULL UNIQUE,
    department TEXT NOT NULL,
    description TEXT,
    enabled INTEGER NOT NULL DEFAULT 1,
    last_crawl_timestamp TEXT,
    last_crawl_status TEXT DEFAULT 'NOT_CRAWLED',
    pages_discovered INTEGER DEFAULT 0,
    pages_indexed INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    indexed_chunks INTEGER DEFAULT 0,
    supported_languages TEXT DEFAULT '["en"]',
    last_error TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- 2. Crawled Pages
CREATE TABLE IF NOT EXISTS crawled_pages (
    id TEXT PRIMARY KEY,
    website_id TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    canonical_url TEXT,
    title TEXT,
    language TEXT DEFAULT 'en',
    has_english_version INTEGER DEFAULT 0,
    english_url TEXT,
    content_hash TEXT,
    etag TEXT,
    last_modified TEXT,
    http_status INTEGER,
    content_length INTEGER,
    crawl_timestamp TEXT,
    indexed_timestamp TEXT,
    status TEXT DEFAULT 'DISCOVERED',
    document_type TEXT DEFAULT 'html',
    error_message TEXT,
    FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE CASCADE
);

-- 3. Documents (Processed text and metadata before chunking)
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    page_id TEXT,
    website_id TEXT,
    title TEXT,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    department TEXT NOT NULL,
    publication_date TEXT,
    document_type TEXT DEFAULT 'html',
    source_language TEXT DEFAULT 'en',
    original_text TEXT,
    english_text TEXT,
    is_translated INTEGER DEFAULT 0,
    content_hash TEXT,
    num_chunks INTEGER DEFAULT 0,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (page_id) REFERENCES crawled_pages(id) ON DELETE SET NULL,
    FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE SET NULL
);

-- 4. Crawl Jobs
CREATE TABLE IF NOT EXISTS crawl_jobs (
    id TEXT PRIMARY KEY,
    website_id TEXT,
    job_type TEXT DEFAULT 'SINGLE_SITE', -- 'SINGLE_SITE', 'ALL_SITES', 'REFRESH_CHANGED'
    status TEXT DEFAULT 'PENDING',       -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED'
    pages_discovered INTEGER DEFAULT 0,
    pages_crawled INTEGER DEFAULT 0,
    pages_indexed INTEGER DEFAULT 0,
    pages_failed INTEGER DEFAULT 0,
    start_time TEXT,
    end_time TEXT,
    duration_seconds REAL DEFAULT 0,
    error_message TEXT,
    FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE SET NULL
);

-- 5. Crawl Errors
CREATE TABLE IF NOT EXISTS crawl_errors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT,
    website_id TEXT,
    url TEXT NOT NULL,
    error_type TEXT,
    status_code INTEGER,
    error_message TEXT,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (job_id) REFERENCES crawl_jobs(id) ON DELETE CASCADE,
    FOREIGN KEY (website_id) REFERENCES websites(id) ON DELETE SET NULL
);

-- 6. Query Logs (Tracing all queries and latencies)
CREATE TABLE IF NOT EXISTS query_logs (
    request_id TEXT PRIMARY KEY,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    user_query TEXT NOT NULL,
    retrieval_mode TEXT NOT NULL,          -- 'INDEXED', 'LIVE', 'HYBRID'
    selected_websites TEXT,                -- JSON list
    detected_intent TEXT,
    detected_topic TEXT,
    detected_department TEXT,
    is_time_sensitive INTEGER DEFAULT 0,
    routing_rationale TEXT,
    total_latency_ms REAL DEFAULT 0,
    intent_analysis_latency_ms REAL DEFAULT 0,
    source_selection_latency_ms REAL DEFAULT 0,
    embedding_latency_ms REAL DEFAULT 0,
    vector_search_latency_ms REAL DEFAULT 0,
    live_retrieval_latency_ms REAL DEFAULT 0,
    relevance_filter_latency_ms REAL DEFAULT 0,
    groq_generation_latency_ms REAL DEFAULT 0,
    quality_gate_latency_ms REAL DEFAULT 0,
    retry_count INTEGER DEFAULT 0,
    total_retry_overhead_ms REAL DEFAULT 0,
    retrieved_chunk_count INTEGER DEFAULT 0,
    retained_chunk_count INTEGER DEFAULT 0,
    selected_sources TEXT,                 -- JSON list of URLs
    final_decision TEXT,                   -- 'ACCEPT', 'RETRY', 'ABSTAIN'
    decision_rationale TEXT,
    generated_answer TEXT,
    groq_model TEXT,
    groq_prompt_tokens INTEGER DEFAULT 0,
    groq_completion_tokens INTEGER DEFAULT 0,
    groq_total_tokens INTEGER DEFAULT 0,
    api_status TEXT DEFAULT 'SUCCESS',
    error_category TEXT
);

-- 7. Retrieved Chunks
CREATE TABLE IF NOT EXISTS retrieved_chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    document_id TEXT,
    source_url TEXT NOT NULL,
    source_title TEXT,
    source_domain TEXT,
    department TEXT,
    publication_date TEXT,
    similarity_score REAL,
    rank_order INTEGER,
    is_retained INTEGER DEFAULT 1,
    is_live_retrieved INTEGER DEFAULT 0,
    source_language TEXT DEFAULT 'en',
    original_text TEXT,
    english_text TEXT,
    is_translated INTEGER DEFAULT 0,
    user_marked_relevant INTEGER, -- NULL, 1 (relevant), 0 (irrelevant)
    FOREIGN KEY (request_id) REFERENCES query_logs(request_id) ON DELETE CASCADE
);

-- 8. Answer Evaluations (JEV Layer 3 and LLM-as-a-judge / automated metrics)
CREATE TABLE IF NOT EXISTS answer_evaluations (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    eval_type TEXT DEFAULT 'AUTOMATED', -- 'AUTOMATED', 'LLM_JUDGE', 'GROUND_TRUTH'
    faithfulness_score REAL,
    answer_relevance_score REAL,
    completeness_score REAL,
    citation_correctness_score REAL,
    citation_coverage_score REAL,
    unsupported_claim_count INTEGER DEFAULT 0,
    context_relevance_score REAL,
    precision_at_k REAL,
    recall_at_k REAL,
    mrr REAL,
    ndcg REAL,
    evaluator_model TEXT,
    rationale TEXT,
    FOREIGN KEY (request_id) REFERENCES query_logs(request_id) ON DELETE CASCADE
);

-- 9. User Feedback
CREATE TABLE IF NOT EXISTS user_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL,
    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
    rating INTEGER NOT NULL, -- 1 for thumbs up, -1 for thumbs down
    comment TEXT,
    feedback_category TEXT,
    FOREIGN KEY (request_id) REFERENCES query_logs(request_id) ON DELETE CASCADE
);

-- 10. System Settings (Persisted configurable values)
CREATE TABLE IF NOT EXISTS system_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_crawled_pages_website ON crawled_pages(website_id);
CREATE INDEX IF NOT EXISTS idx_crawled_pages_url ON crawled_pages(url);
CREATE INDEX IF NOT EXISTS idx_documents_url ON documents(url);
CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp ON query_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_retrieved_chunks_request ON retrieved_chunks(request_id);
CREATE INDEX IF NOT EXISTS idx_answer_evaluations_request ON answer_evaluations(request_id);
