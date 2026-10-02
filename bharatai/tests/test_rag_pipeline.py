from core.source_selection import SourceSelectionAgent
from core.relevance_filter import DocumentRelevanceAgent
from core.quality_gate import AnswerQualityGate

def test_source_selection_agent():
    agent = SourceSelectionAgent()
    # Test temporal keywords
    res = agent.select_sources("What are the latest semiconductor announcements today?")
    assert res["is_time_sensitive"] is True
    assert res["retrieval_mode"] in ("HYBRID", "LIVE")
    assert "meity" in res["selected_website_ids"]

def test_relevance_filter_dedup_and_cutoff():
    rf = DocumentRelevanceAgent(default_threshold=0.3)
    candidates = [
        {"chunk_id": "1", "english_text": "Duplicate content passage", "similarity_score": 0.8},
        {"chunk_id": "2", "english_text": "Duplicate content passage", "similarity_score": 0.8},
        {"chunk_id": "3", "english_text": "Low score irrelevant content", "similarity_score": 0.05},
    ]
    retained, metrics = rf.filter_and_rank(candidates, "query", relevance_threshold=0.3)
    assert metrics["unique_candidate_count"] == 2
    assert len(retained) == 1
    assert retained[0]["chunk_id"] == "1"

def test_quality_gate_citations():
    gate = AnswerQualityGate(max_retries=1)
    citations = [{"citation_id": "[1]", "url": "https://pib.gov.in"}]

    # Answer citing [1]
    res = gate.evaluate_answer("What is PIB?", "PIB is the official press agency [1].", [{"english_text": "PIB press agency"}], citations)
    assert res["decision"] == "ACCEPT"

    # Answer with hallucinated citation [99]
    res_bad = gate.evaluate_answer("What is PIB?", "PIB is an agency [99].", [{"english_text": "PIB press agency"}], citations)
    assert res_bad["decision"] in ("RETRY", "ABSTAIN")

if __name__ == "__main__":
    test_source_selection_agent()
    test_relevance_filter_dedup_and_cutoff()
    test_quality_gate_citations()
    print("test_rag_pipeline: All assertions passed!")
