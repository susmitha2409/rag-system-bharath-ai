"""
BharatAI — Multilingual Language Processor
Detects Indian languages, identifies English page alternates, and translates non-English
content into English using Groq while preserving original metadata.
"""

import re
import logging
from typing import Tuple, Optional, Dict
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None
from core.llm_client import get_groq_client, GroqLLMError

logger = logging.getLogger("bharatai.language_processor")

# Unicode ranges for major Indian scripts
INDIAN_SCRIPT_RANGES = [
    ("hi", 0x0900, 0x097F),  # Devanagari (Hindi, Marathi, Sanskrit)
    ("bn", 0x0980, 0x09FF),  # Bengali, Assamese
    ("pa", 0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    ("gu", 0x0A80, 0x0AFF),  # Gujarati
    ("or", 0x0B00, 0x0B7F),  # Odia
    ("ta", 0x0B80, 0x0BFF),  # Tamil
    ("te", 0x0C00, 0x0C7F),  # Telugu
    ("kn", 0x0C80, 0x0CFF),  # Kannada
    ("ml", 0x0D00, 0x0D7F),  # Malayalam
]


class LanguageProcessor:
    def __init__(self):
        self.groq_client = get_groq_client()

    def detect_language(self, text: str) -> str:
        """
        Detect primary language of text using Unicode script frequency.
        Returns ISO code ('en', 'hi', 'ta', 'te', 'bn', etc.).
        """
        if not text or not text.strip():
            return "en"

        cleaned = re.sub(r"[0-9\s\W]+", "", text)
        if not cleaned:
            return "en"

        script_counts: Dict[str, int] = {"en": 0}
        total_chars = len(cleaned)

        for ch in cleaned:
            code = ord(ch)
            # Latin character range (English)
            if (0x0041 <= code <= 0x005A) or (0x0061 <= code <= 0x007A):
                script_counts["en"] = script_counts.get("en", 0) + 1
            else:
                matched = False
                for lang_code, start_code, end_code in INDIAN_SCRIPT_RANGES:
                    if start_code <= code <= end_code:
                        script_counts[lang_code] = script_counts.get(lang_code, 0) + 1
                        matched = True
                        break
                if not matched:
                    # Treat other European or generic characters as default
                    script_counts["en"] = script_counts.get("en", 0) + 1

        primary_lang = max(script_counts, key=script_counts.get)
        # If English accounts for at least 40% or no dominant script, default to English
        if script_counts.get("en", 0) / total_chars > 0.4 and primary_lang != "en":
            # Mixed text with significant English
            return "en"

        return primary_lang

    def find_english_alternate_url(self, html_content: str, current_url: str) -> Optional[str]:
        """
        Scan page HTML for English alternate links:
        1. <link rel="alternate" hreflang="en" href="...">
        2. Anchor tags with text "English" or "EN" or href containing "/en/" or "?lang=en"
        """
        if not html_content:
            return None

        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(html_content, "html.parser")

                # 1. Check hreflang link tag
                alt_link = soup.find("link", rel=lambda r: r and "alternate" in r, hreflang=lambda h: h and h.lower().startswith("en"))
                if alt_link and alt_link.get("href"):
                    return self._resolve_url(alt_link["href"], current_url)

                # 2. Check language switcher anchors
                for a in soup.find_all("a", href=True):
                    text = a.get_text(strip=True).lower()
                    href = a["href"].lower()
                    title = (a.get("title") or "").lower()

                    if text in ("english", "en", "eng") or "english" in title:
                        resolved = self._resolve_url(a["href"], current_url)
                        if resolved and resolved != current_url:
                            return resolved

                    if "/en/" in href or "lang=en" in href or "language=en" in href:
                        resolved = self._resolve_url(a["href"], current_url)
                        if resolved and resolved != current_url:
                            return resolved

            except Exception as e:
                logger.debug(f"Error inspecting English alternate link: {e}")
        else:
            en_matches = re.findall(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(?:English|EN|Eng)</a>', html_content, re.I)
            if en_matches:
                return self._resolve_url(en_matches[0], current_url)

        return None

    def _resolve_url(self, link: str, base: str) -> str:
        from urllib.parse import urljoin
        return urljoin(base, link)

    def translate_to_english(self, text: str, source_language: str) -> Tuple[str, bool]:
        """
        Translate Indian language passage to English using Groq.
        Preserves official entities, schemes, dates, numbers, and currency values.
        Returns: Tuple[translated_text, success_bool]
        """
        if not text or source_language == "en":
            return text, False

        if not self.groq_client.is_configured():
            logger.warning("Groq client not configured for translation; returning original text.")
            return text, False

        prompt = (
            f"You are an expert official translator for the Government of India. "
            f"Translate the following {source_language.upper()} government text into formal English. "
            f"Preserve all official scheme names (e.g., PM-KISAN, Ayushman Bharat), ministry titles, dates, "
            f"monetary figures (crores/lakhs/rupees), and designations exactly. "
            f"Return ONLY the English translation without preamble or explanatory notes.\n\n"
            f"Source Text:\n{text}"
        )

        try:
            messages = [
                {"role": "system", "content": "You are a professional government translator. Provide only the direct English translation."},
                {"role": "user", "content": prompt},
            ]
            translated, _ = self.groq_client.generate_chat_completion(
                messages=messages,
                temperature=0.0,
                max_tokens=1000,
            )
            return translated.strip(), True
        except GroqLLMError as e:
            logger.error(f"Groq translation failed: {e}")
            return text, False
        except Exception as e:
            logger.error(f"Unexpected translation error: {e}")
            return text, False


# Singleton instance
_lang_proc = None

def get_language_processor() -> LanguageProcessor:
    global _lang_proc
    if _lang_proc is None:
        _lang_proc = LanguageProcessor()
    return _lang_proc
