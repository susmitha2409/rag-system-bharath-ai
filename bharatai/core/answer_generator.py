"""
BharatAI — Answer Generation Module (Groq LLM)
Constructs grounded prompts using retrieved government evidence, generates answers
with inline citations [1], [2], and extracts verifiable citation maps.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from core.llm_client import get_groq_client, GroqLLMError
from core.security import sanitize_retrieved_evidence, redact_secrets

logger = logging.getLogger("bharatai.answer_generator")


class AnswerGenerator:
    def __init__(self):
        self.groq_client = get_groq_client()

    def generate_answer(
        self,
        query: str,
        evidence_chunks: List[Dict[str, Any]],
        response_language: str = "en",
        temperature: float = 0.1,
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate answer from retrieved evidence chunks.
        Returns:
            Tuple[answer_text, citation_map, groq_metadata]
        """
        if not evidence_chunks:
            abstain_msg = (
                "Insufficient evidence was found in the official Government of India knowledge base "
                "or live official portals to answer your query with high confidence. "
                "Please refine your query or crawl the relevant ministry website."
            )
            return abstain_msg, [], {"latency_ms": 0.0, "total_tokens": 0, "model": self.groq_client.model}

        # Build numbered evidence context
        context_parts = []
        citation_map = []

        for i, chunk in enumerate(evidence_chunks):
            citation_num = i + 1
            source_url = chunk.get("source_url", "")
            source_title = chunk.get("source_title", "Official Portal Document")
            dept = chunk.get("department", "Government of India")
            pub_date = chunk.get("publication_date", "Date not specified")
            text = chunk.get("english_text") or chunk.get("original_text") or ""
            is_trans = chunk.get("is_translated", False)

            trans_notice = " (Machine-translated from original Indian language)" if is_trans else ""

            context_parts.append(
                f"--- SOURCE [{citation_num}] ---\n"
                f"Title: {source_title}\n"
                f"Department/Ministry: {dept}\n"
                f"Publication Date: {pub_date}\n"
                f"Official URL: {source_url}{trans_notice}\n"
                f"Content Passage:\n{sanitize_retrieved_evidence(text)}\n"
            )

            citation_map.append(
                {
                    "citation_id": f"[{citation_num}]",
                    "citation_number": citation_num,
                    "title": source_title,
                    "url": source_url,
                    "department": dept,
                    "publication_date": pub_date,
                    "source_language": chunk.get("source_language", "en"),
                    "is_translated": is_trans,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "similarity_score": chunk.get("similarity_score", 0.0),
                    "passage_snippet": text[:200] + ("..." if len(text) > 200 else ""),
                }
            )

        evidence_text = "\n".join(context_parts)

        # Language instruction
        lang_instruction = "Answer strictly in English."
        if response_language and response_language.lower() != "en":
            lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi"}
            target_lang_name = lang_names.get(response_language.lower(), response_language)
            lang_instruction = f"Answer in clear, formal {target_lang_name} language while retaining official English scheme acronyms."

        system_prompt = (
            "You are BharatAI, an authoritative and objective government information assistant representing "
            "the official portals of the Government of India. "
            "Your solemn duty is to answer the citizen's inquiry truthfully, precisely, and strictly based "
            "on the provided official evidence passages.\n\n"
            "MANDATORY OPERATING RULES:\n"
            "1. Grounding: Answer ONLY using facts directly stated in the supplied evidence. Do not extrapolate, assume, or invent details.\n"
            "2. Citations: Every single factual claim or statistic MUST have an inline bracket citation referencing the source, e.g., [1] or [2].\n"
            "3. Entities & Figures: Maintain complete precision for dates, financial amounts (crores/lakhs), percentage figures, and ministry designations.\n"
            "4. Insufficient Evidence: If the provided evidence does not fully answer the question, explicitly state: 'Based on the retrieved official records, certain details could not be verified.'\n"
            "5. Neutrality: Maintain a professional, neutral civil service tone.\n"
            f"6. Language: {lang_instruction}\n"
            "7. Do NOT fabricate or alter any URL. Citations must only use numbers [1], [2], etc."
        )

        user_prompt = (
            f"CITIZEN QUESTION:\n{query}\n\n"
            f"OFFICIAL GOVERNMENT EVIDENCE PASSAGES:\n"
            f"{evidence_text}\n\n"
            f"Please generate the official response with inline citations:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            content, meta = self.groq_client.generate_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=1500,
            )
            return content.strip(), citation_map, meta
        except GroqLLMError as e:
            logger.error(f"Groq answer generation error: {e}")
            fallback_ans = f"Error generating answer via Groq API: {e}. Please check your API key in Settings."
            return fallback_ans, citation_map, {"latency_ms": 0.0, "total_tokens": 0, "error": str(e)}


# Singleton instance
_answer_gen = None

def get_answer_generator() -> AnswerGenerator:
    global _answer_gen
    if _answer_gen is None:
        _answer_gen = AnswerGenerator()
    return _answer_gen
