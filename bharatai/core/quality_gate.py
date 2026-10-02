"""
BharatAI — JEV Decision Layer 3: Answer Quality Gate
Validates answer groundedness, citation coverage and correctness, detects unsupported claims,
and issues ACCEPT, RETRY, or ABSTAIN decisions.
"""

import re
import logging
from typing import Dict, Any, List, Tuple, Optional
from config import QUALITY_GATE_MAX_RETRIES
from core.llm_client import get_groq_client, GroqLLMError

logger = logging.getLogger("bharatai.quality_gate")


class AnswerQualityGate:
    def __init__(self, max_retries: int = QUALITY_GATE_MAX_RETRIES):
        self.max_retries = max_retries
        self.groq_client = get_groq_client()

    def evaluate_answer(
        self,
        query: str,
        answer: str,
        evidence_chunks: List[Dict[str, Any]],
        citation_map: List[Dict[str, Any]],
        current_retry_count: int = 0,
        enable_llm_judge: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute JEV Decision Layer 3:
        Returns:
            Dict containing:
                - decision: 'ACCEPT', 'RETRY', or 'ABSTAIN'
                - rationale: explanation of gate decision
                - faithfulness_score: float (0.0 to 1.0)
                - answer_relevance_score: float (0.0 to 1.0)
                - citation_correctness_score: float (0.0 to 1.0)
                - citation_coverage_score: float (0.0 to 1.0)
                - unsupported_claim_count: int
                - should_retry: bool
                - reformulated_query: Optional[str]
        """
        if not evidence_chunks or not answer.strip():
            return {
                "decision": "ABSTAIN",
                "rationale": "No evidence was retrieved from official sources to support an answer.",
                "faithfulness_score": 0.0,
                "answer_relevance_score": 0.0,
                "citation_correctness_score": 0.0,
                "citation_coverage_score": 0.0,
                "unsupported_claim_count": 0,
                "should_retry": False,
                "reformulated_query": None,
            }

        # 1. Deterministic Citation Validation
        # Find all [N] bracket citations in the generated answer
        found_citations = re.findall(r"\[(\d+)\]", answer)
        found_num_set = set(int(n) for n in found_citations)
        valid_citation_nums = set(range(1, len(citation_map) + 1))

        # Check for hallucinated / phantom citations
        invalid_citations = found_num_set - valid_citation_nums
        citation_correctness = 1.0 if not found_num_set else (len(found_num_set - invalid_citations) / len(found_num_set))

        # Check citation coverage (at least 1 citation for factual answers)
        has_citations = len(found_citations) > 0
        citation_coverage = 1.0 if has_citations else 0.2

        # 2. Text Grounding / Lexical Overlap Heuristic
        # Check that key terms in answer appear in the evidence chunks
        evidence_words = set(re.findall(r"\w{4,}", " ".join([c.get("english_text", "") for c in evidence_chunks]).lower()))
        answer_words = re.findall(r"\w{4,}", answer.lower())
        overlap = [w for w in answer_words if w in evidence_words]
        faithfulness_approx = len(overlap) / max(len(answer_words), 1)
        faithfulness_score = min(1.0, max(0.4, faithfulness_approx * 1.2))

        # Answer relevance heuristic
        query_words = set(re.findall(r"\w{3,}", query.lower()))
        q_overlap = [w for w in answer_words if w in query_words]
        relevance_score = min(1.0, max(0.5, len(q_overlap) / max(len(query_words), 1)))

        unsupported_claims = len(invalid_citations)

        # 3. Optional LLM Judge using Groq
        if enable_llm_judge and self.groq_client.is_configured() and len(answer) > 40:
            llm_eval = self._evaluate_with_llm(query, answer, evidence_chunks)
            if llm_eval:
                faithfulness_score = llm_eval.get("faithfulness", faithfulness_score)
                relevance_score = llm_eval.get("relevance", relevance_score)
                unsupported_claims = llm_eval.get("unsupported_claims", unsupported_claims)

        # 4. Gate Decision Logic
        if "insufficient evidence" in answer.lower() or "could not be verified" in answer.lower():
            decision = "ABSTAIN"
            rationale = "Generated answer explicitly declared insufficient official evidence."
            should_retry = False
            reformulated_query = None

        elif invalid_citations or faithfulness_score < 0.40:
            if current_retry_count < self.max_retries:
                decision = "RETRY"
                rationale = (
                    f"Answer quality check failed (faithfulness: {faithfulness_score:.2f}, "
                    f"invalid citations: {list(invalid_citations)}). Triggering retrieval retry #{current_retry_count + 1}."
                )
                should_retry = True
                reformulated_query = f"{query} official guidelines circular notifications"
            else:
                decision = "ABSTAIN"
                rationale = f"Max retries ({self.max_retries}) exhausted without meeting groundedness threshold."
                should_retry = False
                reformulated_query = None

        else:
            decision = "ACCEPT"
            rationale = (
                f"Answer accepted by JEV Quality Gate (Faithfulness: {faithfulness_score:.2f}, "
                f"Relevance: {relevance_score:.2f}, Citation Correctness: {citation_correctness:.2f})."
            )
            should_retry = False
            reformulated_query = None

        return {
            "decision": decision,
            "rationale": rationale,
            "faithfulness_score": round(faithfulness_score, 2),
            "answer_relevance_score": round(relevance_score, 2),
            "citation_correctness_score": round(citation_correctness, 2),
            "citation_coverage_score": round(citation_coverage, 2),
            "unsupported_claim_count": unsupported_claims,
            "should_retry": should_retry,
            "reformulated_query": reformulated_query,
        }

    def _evaluate_with_llm(
        self, query: str, answer: str, evidence_chunks: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """LLM-as-a-judge evaluation of faithfulness and citation compliance using Groq."""
        evidence_summary = "\n".join([f"- {c.get('english_text', '')[:300]}" for c in evidence_chunks[:3]])
        prompt = (
            f"Evaluate this RAG generated response for a Government portal query:\n"
            f"Query: {query}\n"
            f"Retrieved Evidence:\n{evidence_summary}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Return JSON with:\n"
            f"- faithfulness: float 0.0 to 1.0 (are claims grounded in evidence?)\n"
            f"- relevance: float 0.0 to 1.0 (does it directly answer the user?)\n"
            f"- unsupported_claims: integer count of ungrounded statements"
        )
        try:
            parsed, _ = self.groq_client.generate_json_completion(
                messages=[
                    {"role": "system", "content": "You are BharatAI JEV quality gate evaluator. Output JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return parsed
        except Exception:
            return None


# Singleton instance
_quality_gate = None

def get_quality_gate() -> AnswerQualityGate:
    global _quality_gate
    if _quality_gate is None:
        _quality_gate = AnswerQualityGate()
    return _quality_gate
