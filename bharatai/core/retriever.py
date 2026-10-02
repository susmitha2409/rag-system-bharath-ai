"""
BharatAI — Multi-Source Retrieval Engine
Supports Mode A (Indexed Vector RAG), Mode B (Live Official Website Retrieval),
and Mode C (Hybrid Retrieval combining persistent index and live official feeds).
"""

import time
import logging
from typing import List, Dict, Any, Optional
try:
    import requests
except ImportError:
    requests = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from config import OFFICIAL_WEBSITES, is_approved_domain, CRAWLER_USER_AGENT
from core.vector_store import get_vector_store
from core.document_processor import get_document_processor
from core.security import validate_crawl_url

logger = logging.getLogger("bharatai.retriever")


class GovernmentInformationRetriever:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.doc_processor = get_document_processor()
        self.session = requests.Session() if requests is not None else None
        if self.session is not None:
            self.session.headers.update({"User-Agent": CRAWLER_USER_AGENT})

    def retrieve(
        self,
        query: str,
        retrieval_mode: str = "HYBRID",
        selected_website_ids: Optional[List[str]] = None,
        top_k: int = 5,
        department_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Unified multi-source retriever:
        - 'INDEXED': Query ChromaDB persistent store
        - 'LIVE': Scrape latest announcements directly from official websites
        - 'HYBRID': Combine both indexed and live content
        """
        candidates: List[Dict[str, Any]] = []

        # 1. Indexed Retrieval (Mode A & Mode C)
        if retrieval_mode in ("INDEXED", "HYBRID"):
            indexed_results = self._retrieve_from_index(
                query=query,
                top_k=top_k * 2,
                department_filter=department_filter,
                selected_website_ids=selected_website_ids,
            )
            candidates.extend(indexed_results)

        # 2. Live Official Website Retrieval (Mode B & Mode C)
        if retrieval_mode in ("LIVE", "HYBRID"):
            live_results = self._retrieve_live_official_news(
                query=query,
                selected_website_ids=selected_website_ids,
                max_live_pages=3,
            )
            candidates.extend(live_results)

        return candidates

    def _retrieve_from_index(
        self,
        query: str,
        top_k: int = 5,
        department_filter: Optional[str] = None,
        selected_website_ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB vector database."""
        # Convert selected_website_ids to domain if single site selected
        domain_filter = None
        if selected_website_ids and len(selected_website_ids) == 1:
            site_info = next((s for s in OFFICIAL_WEBSITES if s["id"] == selected_website_ids[0]), None)
            if site_info:
                domain_filter = site_info["canonical_domain"]

        results = self.vector_store.similarity_search(
            query=query,
            top_k=top_k,
            department_filter=department_filter,
            domain_filter=domain_filter,
        )
        for r in results:
            r["is_live_retrieved"] = False

        return results

    def _retrieve_live_official_news(
        self,
        query: str,
        selected_website_ids: Optional[List[str]] = None,
        max_live_pages: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Live Website Retrieval:
        Directly checks configured official news sections (e.g. PIB, PMIndia, MeitY press releases),
        fetches the latest announcements, extracts content, and turns them into live candidate chunks.
        """
        target_sites = []
        if selected_website_ids:
            target_sites = [s for s in OFFICIAL_WEBSITES if s["id"] in selected_website_ids and s.get("enabled", True)]
        if not target_sites:
            # Default to PIB and PMIndia for breaking news
            target_sites = [s for s in OFFICIAL_WEBSITES if s["id"] in ("pib", "pmindia")]

        live_chunks: List[Dict[str, Any]] = []

        for site in target_sites[:2]:  # Limit to 2 sites to maintain rapid latency
            base_url = site["base_url"]
            sections = site.get("news_sections", ["/"])

            for sec in sections[:1]:
                target_url = base_url.rstrip("/") + sec
                valid, msg = validate_crawl_url(target_url)
                if not valid:
                    continue

                try:
                    resp = self.session.get(target_url, timeout=5)
                    if resp.status_code != 200:
                        continue

                    # Extract announcement links from the news section
                    soup = BeautifulSoup(resp.text, "html.parser")
                    article_links = []
                    for a in soup.find_all("a", href=True):
                        href = a["href"].strip()
                        text = a.get_text(strip=True)
                        if len(text) > 15 and is_approved_domain(href or target_url):
                            from urllib.parse import urljoin
                            full = urljoin(target_url, href)
                            if full not in article_links and full != target_url:
                                article_links.append((full, text))
                                if len(article_links) >= max_live_pages:
                                    break

                    # Fetch the top 1-2 latest articles on the fly
                    for art_url, art_title in article_links[:2]:
                        try:
                            art_resp = self.session.get(art_url, timeout=5)
                            if art_resp.status_code == 200:
                                doc_rec = self.doc_processor.process_html(
                                    html_content=art_resp.text,
                                    url=art_url,
                                    department=site["department"],
                                    domain=site["canonical_domain"],
                                    title_override=art_title,
                                )
                                for c in doc_rec.get("chunks", [])[:2]:
                                    c["is_live_retrieved"] = True
                                    # Assign reasonable baseline similarity score for live items
                                    c["similarity_score"] = 0.85
                                    live_chunks.append(c)
                        except Exception as e:
                            logger.debug(f"Failed to fetch live article {art_url}: {e}")

                except Exception as e:
                    logger.debug(f"Live retrieval error for {target_url}: {e}")

        return live_chunks


# Singleton instance
_retriever = None

def get_retriever() -> GovernmentInformationRetriever:
    global _retriever
    if _retriever is None:
        _retriever = GovernmentInformationRetriever()
    return _retriever
