"""
BharatAI — JEV Decision Layer 2: Document Relevance Filter
Filters candidate chunks, deduplicates content, applies similarity thresholds,
translates non-English passages, and constructs grounded evidence context.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from config import RELEVANCE_THRESHOLD
from core.language_processor import get_language_processor
from core.security import sanitize_retrieved_evidence

logger = logging.getLogger("bharatai.relevance_filter")


class DocumentRelevanceAgent:
    def __init__(self, default_threshold: float = RELEVANCE_THRESHOLD):
        self.default_threshold = default_threshold
        self.lang_processor = get_language_processor()

    def filter_and_rank(
        self,
        candidate_chunks: List[Dict[str, Any]],
        query: str,
        relevance_threshold: Optional[float] = None,
        max_chunks: int = 5,
        prefer_english: bool = True,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Execute JEV Decision Layer 2:
        1. Deduplicate by content hash or text similarity
        2. Filter by relevance threshold
        3. Translate non-English chunks to English if needed
        4. Enforce max chunks limit
        Returns:
            Tuple[retained_chunks, metrics_dict]
        """
        threshold = relevance_threshold if relevance_threshold is not None else self.default_threshold
        initial_count = len(candidate_chunks)

        if not candidate_chunks:
            return [], {
                "initial_candidate_count": 0,
                "unique_candidate_count": 0,
                "retained_chunk_count": 0,
                "threshold_applied": threshold,
                "scores": [],
            }

        # 1. Deduplication
        seen_hashes = set()
        unique_chunks: List[Dict[str, Any]] = []

        for chunk in candidate_chunks:
            text = chunk.get("english_text") or chunk.get("original_text") or ""
            # Simple content fingerprint
            sample = text[:150].strip().lower()
            if not sample or sample in seen_hashes:
                continue
            seen_hashes.add(sample)
            unique_chunks.append(chunk)

        # 2. Score & threshold filtering
        # Sort descending by similarity_score
        unique_chunks.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)

        retained: List[Dict[str, Any]] = []
        for chunk in unique_chunks:
            score = chunk.get("similarity_score", 0.0)
            if score >= threshold:
                chunk["is_retained"] = True
                retained.append(chunk)
            else:
                chunk["is_retained"] = False

        # If strict threshold left nothing, but top candidate is reasonable (> 0.20), keep top 1-2 to avoid over-abstaining
        if not retained and unique_chunks and unique_chunks[0].get("similarity_score", 0.0) >= 0.20:
            top_chunk = unique_chunks[0]
            top_chunk["is_retained"] = True
            retained.append(top_chunk)

        # 3. Limit to max_chunks
        retained = retained[:max_chunks]

        # 4. Multilingual handling: translate non-English retained chunks
        for chunk in retained:
            lang = chunk.get("source_language", "en")
            if lang != "en" and not chunk.get("is_translated", False):
                orig_text = chunk.get("original_text", "")
                translated, success = self.lang_processor.translate_to_english(orig_text, lang)
                if success:
                    chunk["english_text"] = translated
                    chunk["is_translated"] = True

            # Sanitize text
            chunk["english_text"] = sanitize_retrieved_evidence(chunk.get("english_text", ""))

        metrics = {
            "initial_candidate_count": initial_count,
            "unique_candidate_count": len(unique_chunks),
            "retained_chunk_count": len(retained),
            "threshold_applied": threshold,
            "scores": [c.get("similarity_score", 0.0) for c in retained],
        }

        return retained, metrics


# Singleton instance
_relevance_filter = None

def get_relevance_filter() -> DocumentRelevanceAgent:
    global _relevance_filter
    if _relevance_filter is None:
        _relevance_filter = DocumentRelevanceAgent()
    return _relevance_filter
