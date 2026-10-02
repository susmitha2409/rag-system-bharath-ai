"""
BharatAI — Test Suite Runner
Executes all unit tests with full reporting.
"""

import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(root_dir))

from tests.test_llm_client import (
    test_groq_client_unconfigured,
    test_groq_client_set_credentials,
    test_redact_secrets,
)
from tests.test_crawler import test_approved_domains, test_ssrf_protection
from tests.test_document_processor import test_language_detection, test_document_chunking
from tests.test_retriever import test_vector_store_add_and_search
from tests.test_rag_pipeline import (
    test_source_selection_agent,
    test_relevance_filter_dedup_and_cutoff,
    test_quality_gate_citations,
)
from tests.test_evaluation import test_retrieval_metrics, test_benchmark_dataset_length

def main():
    print("=" * 65)
    print("  🏛️  BHARATAI AGENTIC RAG SYSTEM TEST SUITE")
    print("=" * 65)

    tests = [
        ("test_llm_client (Unconfigured check)", test_groq_client_unconfigured),
        ("test_llm_client (Set credentials)", test_groq_client_set_credentials),
        ("test_llm_client (Secret redaction)", test_redact_secrets),
        ("test_crawler (Approved Gov domains allowlist)", test_approved_domains),
        ("test_crawler (SSRF / internal IP defenses)", test_ssrf_protection),
        ("test_document_processor (Language detection)", test_language_detection),
        ("test_document_processor (Document chunking)", test_document_chunking),
        ("test_retriever (Vector store add & search)", test_vector_store_add_and_search),
        ("test_rag_pipeline (JEV Layer 1: Source Selection)", test_source_selection_agent),
        ("test_rag_pipeline (JEV Layer 2: Relevance Filter)", test_relevance_filter_dedup_and_cutoff),
        ("test_rag_pipeline (JEV Layer 3: Quality Gate)", test_quality_gate_citations),
        ("test_evaluation (Precision@K, Recall@K, MRR, NDCG)", test_retrieval_metrics),
        ("test_evaluation (Benchmark dataset 20+ Qs across 10 ministries)", test_benchmark_dataset_length),
    ]

    passed = 0
    failed = 0

    for name, func in tests:
        try:
            func()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed += 1

    print("=" * 65)
    print(f"Test Execution Summary: {passed} PASSED, {failed} FAILED (Total: {len(tests)})")
    print("=" * 65)

    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()
