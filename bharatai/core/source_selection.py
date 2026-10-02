"""
BharatAI — JEV Decision Layer 1: Tool & Source Selection Agent
Analyzes user query intent, identifies relevant government departments and websites,
detects time sensitivity, and determines optimal retrieval mode (Indexed, Live, or Hybrid).
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional
from config import OFFICIAL_WEBSITES
from core.llm_client import get_groq_client, GroqLLMError

logger = logging.getLogger("bharatai.source_selection")

TEMPORAL_KEYWORDS = {
    "latest", "today", "yesterday", "current", "recent", "new", "notification",
    "announcement", "press release", "circular", "update", "this week", "this month",
    "2025", "2026", "gazette", "breaking", "amendment"
}

DEPARTMENT_KEYWORD_MAP = {
    "pmindia": ["pm", "prime minister", "narendra modi", "pm-kisan", "mann ki baat", "pmo", "cabinet decision"],
    "meity": ["it", "electronics", "digital india", "semiconductor", "cyber", "ai mission", "chips", "software", "meity", "cert-in", "data protection"],
    "cabsec": ["cabinet", "secretariat", "allocation of business", "transaction of business", "secretary"],
    "mohfw": ["health", "hospital", "ayushman", "vaccine", "covid", "disease", "medical", "mohfw", "aiims", "doctor"],
    "mha": ["police", "security", "border", "disaster", "ndrf", "internal security", "citizenship", "visa", "immigration", "mha"],
    "mib": ["broadcasting", "media", "films", "censor", "press", "journalism", "radio", "television", "mib"],
    "doe": ["expenditure", "procurement", "gem", "allowances", "pension", "finance commission", "disbursement"],
    "dea": ["economy", "economic", "budget", "gdp", "inflation", "investment", "fdi", "rupee", "forex", "borrowing"],
    "pib": ["press release", "pib", "fact check", "briefing", "statement", "government release"],
    "india_gov": ["service", "portal", "scheme", "certificate", "aadhaar", "ration card", "passport", "citizen services"],
}


class SourceSelectionAgent:
    def __init__(self):
        self.groq_client = get_groq_client()

    def select_sources(
        self,
        query: str,
        user_selected_sites: Optional[List[str]] = None,
        forced_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute JEV Decision Layer 1:
        1. Determine if query is time-sensitive (latest / recent)
        2. Select relevant official government websites
        3. Determine retrieval mode (INDEXED, LIVE, or HYBRID)
        4. Provide routing rationale and score
        """
        query_lower = query.lower()

        # 1. Temporal / Live detection
        is_time_sensitive = any(re.search(r"\b" + re.escape(kw) + r"\b", query_lower) for kw in TEMPORAL_KEYWORDS)

        # 2. Department / Website mapping
        matched_site_ids = []
        for site_id, kws in DEPARTMENT_KEYWORD_MAP.items():
            for kw in kws:
                if re.search(r"\b" + re.escape(kw) + r"\b", query_lower):
                    matched_site_ids.append(site_id)
                    break

        # If user explicitly filtered websites in UI, respect selection
        if user_selected_sites and len(user_selected_sites) > 0 and "All" not in user_selected_sites:
            target_site_ids = user_selected_sites
        elif matched_site_ids:
            target_site_ids = matched_site_ids
        else:
            # Default to PIB, National Portal, and PMO for broad queries
            target_site_ids = ["pib", "india_gov", "pmindia"]

        # 3. Determine Retrieval Mode
        if forced_mode:
            retrieval_mode = forced_mode
        elif is_time_sensitive:
            retrieval_mode = "HYBRID"
        else:
            retrieval_mode = "INDEXED"

        # 4. Use Groq for advanced intent classification if configured
        intent_info = self._analyze_intent_with_groq(query) if self.groq_client.is_configured() else None

        if intent_info:
            detected_intent = intent_info.get("intent", "information_retrieval")
            detected_topic = intent_info.get("topic", "General Government Policy")
            detected_department = intent_info.get("department", "Government of India")
            routing_score = float(intent_info.get("confidence", 0.90))
            rationale = intent_info.get(
                "rationale",
                f"Selected {len(target_site_ids)} official sources based on semantic domain relevance."
            )
            if intent_info.get("requires_live", False) and forced_mode is None:
                retrieval_mode = "HYBRID"
                is_time_sensitive = True
        else:
            detected_intent = "government_information_lookup"
            detected_topic = "Government of India Initiatives"
            detected_department = "Central Government Ministries"
            routing_score = 0.85
            rationale = (
                f"Deterministic keyword match selected {len(target_site_ids)} official domains. "
                f"Temporal markers detected: {is_time_sensitive}."
            )

        return {
            "selected_website_ids": target_site_ids,
            "retrieval_mode": retrieval_mode,
            "is_time_sensitive": is_time_sensitive,
            "detected_intent": detected_intent,
            "detected_topic": detected_topic,
            "detected_department": detected_department,
            "routing_score": routing_score,
            "rationale": rationale,
        }

    def _analyze_intent_with_groq(self, query: str) -> Optional[Dict[str, Any]]:
        """Optional LLM-based intent analysis using Groq JSON mode."""
        prompt = (
            f"Analyze this citizen query about Government of India:\n\"{query}\"\n\n"
            f"Return a JSON object with keys:\n"
            f"- intent: short string describing intent (e.g., scheme_lookup, policy_clarification, news_inquiry)\n"
            f"- topic: main subject\n"
            f"- department: primary central ministry (e.g., MeitY, MoHFW, Finance, PMO)\n"
            f"- requires_live: boolean (true if asks for recent/current/today/latest updates)\n"
            f"- confidence: float between 0.0 and 1.0\n"
            f"- rationale: 1-sentence reasoning for knowledge source selection"
        )
        try:
            parsed, _ = self.groq_client.generate_json_completion(
                messages=[
                    {"role": "system", "content": "You are BharatAI reasoning agent. Output valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            return parsed
        except Exception as e:
            logger.debug(f"LLM intent reasoning bypassed: {e}")
            return None


# Singleton instance
_source_selector = None

def get_source_selection_agent() -> SourceSelectionAgent:
    global _source_selector
    if _source_selector is None:
        _source_selector = SourceSelectionAgent()
    return _source_selector
