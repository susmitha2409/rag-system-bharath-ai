from core.evaluation import (
    calculate_precision_recall_at_k,
    calculate_mrr,
    calculate_ndcg,
    BENCHMARK_DATASET,
)

def test_retrieval_metrics():
    retrieved = ["https://pib.gov.in/page1", "https://meity.gov.in/page2", "https://mohfw.gov.in/page3"]
    ground_truth = ["https://pib.gov.in/page1"]

    p, r = calculate_precision_recall_at_k(retrieved, ground_truth, k=3)
    assert abs(p - 0.333) < 0.02
    assert r == 1.0

    mrr = calculate_mrr(retrieved, ground_truth)
    assert mrr == 1.0

    ndcg = calculate_ndcg(retrieved, ground_truth, k=3)
    assert ndcg == 1.0

def test_benchmark_dataset_length():
    assert len(BENCHMARK_DATASET) >= 20
    # Verify all 10 ministries are covered
    target_sites = set(item["target_website_id"] for item in BENCHMARK_DATASET)
    assert len(target_sites) == 10

if __name__ == "__main__":
    test_retrieval_metrics()
    test_benchmark_dataset_length()
    print("test_evaluation: All assertions passed!")
