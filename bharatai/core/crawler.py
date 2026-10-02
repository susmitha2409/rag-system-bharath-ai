"""
BharatAI — Official Government Website Crawler
Crawls approved government domains, respects robots.txt and delays, discovers pages and PDFs,
handles multilingual alternates, and updates SQLite and Vector Store indexes.
"""

import time
import uuid
import logging
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser
from typing import List, Dict, Any, Optional, Set, Callable
try:
    import requests
except ImportError:
    requests = None
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from config import (
    CRAWLER_MAX_PAGES_PER_SITE,
    CRAWLER_MAX_DEPTH,
    CRAWLER_REQUEST_DELAY,
    CRAWLER_TIMEOUT_SECONDS,
    CRAWLER_USER_AGENT,
    is_approved_domain,
)
from core.security import validate_crawl_url
from core.language_processor import get_language_processor
from core.document_processor import get_document_processor
from core.vector_store import get_vector_store
from database.db import get_db

logger = logging.getLogger("bharatai.crawler")


class GovernmentCrawler:
    def __init__(
        self,
        max_pages: int = CRAWLER_MAX_PAGES_PER_SITE,
        max_depth: int = CRAWLER_MAX_DEPTH,
        delay: float = CRAWLER_REQUEST_DELAY,
        timeout: int = CRAWLER_TIMEOUT_SECONDS,
    ):
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.delay = delay
        self.timeout = timeout
        self.user_agent = CRAWLER_USER_AGENT

        self.db = get_db()
        self.vector_store = get_vector_store()
        self.doc_processor = get_document_processor()
        self.lang_processor = get_language_processor()

        self.session = requests.Session() if requests is not None else None
        if self.session is not None:
            self.session.headers.update({"User-Agent": self.user_agent})

        self._stop_requested = False

    def request_stop(self):
        """Signal the crawler to gracefully stop execution."""
        self._stop_requested = True

    def crawl_website(
        self,
        website_id: str,
        job_id: Optional[str] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None,
    ) -> Dict[str, Any]:
        """
        Crawl a single configured government website starting from its base URL.
        Discovers pages, extracts text, handles PDFs, indexes into ChromaDB & SQLite.
        """
        self._stop_requested = False
        site = self.db.get_website_by_id(website_id)
        if not site:
            return {"status": "FAILED", "error": f"Website {website_id} not found."}

        base_url = site["base_url"]
        domain = site["canonical_domain"]
        dept = site["department"]

        current_job_id = job_id or f"job_{uuid.uuid4().hex[:10]}"
        self.db.create_crawl_job(current_job_id, website_id, job_type="SINGLE_SITE")

        start_time = time.time()
        discovered_urls: Set[str] = {base_url}
        visited_urls: Set[str] = set()
        queue: List[Tuple[str, int]] = [(base_url, 0)]  # (url, depth)

        pages_crawled = 0
        pages_indexed = 0
        pages_failed = 0
        indexed_chunks = 0
        detected_languages: Set[str] = {"en"}

        # Initialize robots.txt
        rp = self._get_robot_parser(base_url)

        # Also try sitemap.xml for quick seed discovery
        sitemap_seeds = self._discover_sitemap_urls(base_url)
        for s_url in sitemap_seeds[:self.max_pages]:
            if s_url not in discovered_urls:
                discovered_urls.add(s_url)
                queue.append((s_url, 1))

        logger.info(f"Starting crawl of {site['name']} ({base_url})")

        while queue and pages_crawled < self.max_pages and not self._stop_requested:
            url, depth = queue.pop(0)

            # Strip fragment
            clean_url, _ = urldefrag(url)
            if clean_url in visited_urls:
                continue

            visited_urls.add(clean_url)

            # Safety and security validation
            valid, msg = validate_crawl_url(clean_url)
            if not valid:
                logger.debug(f"Skipping invalid URL {clean_url}: {msg}")
                continue

            # Respect robots.txt
            if rp and not rp.can_fetch(self.user_agent, clean_url):
                logger.debug(f"robots.txt disallowed: {clean_url}")
                continue

            if progress_callback:
                progress_callback(f"Crawling ({pages_crawled + 1}/{self.max_pages}): {clean_url[:60]}...", pages_crawled, self.max_pages)

            time.sleep(self.delay)

            try:
                pages_crawled += 1
                resp = self.session.get(clean_url, timeout=self.timeout, allow_redirects=True)
                content_type = resp.headers.get("Content-Type", "").lower()

                if resp.status_code != 200:
                    pages_failed += 1
                    self.db.record_crawl_error(current_job_id, website_id, clean_url, "HTTP_ERROR", resp.status_code, f"Status code: {resp.status_code}")
                    continue

                # Process PDF Document
                if "application/pdf" in content_type or clean_url.lower().endswith(".pdf"):
                    doc_rec = self.doc_processor.process_pdf(
                        pdf_bytes=resp.content,
                        url=clean_url,
                        department=dept,
                        domain=domain,
                    )
                    self._persist_document(doc_rec, website_id, clean_url, resp)
                    pages_indexed += 1
                    indexed_chunks += doc_rec["num_chunks"]
                    continue

                # Process HTML Page
                html_text = resp.text
                english_alt = self.lang_processor.find_english_alternate_url(html_text, clean_url)

                doc_rec = self.doc_processor.process_html(
                    html_content=html_text,
                    url=clean_url,
                    department=dept,
                    domain=domain,
                )

                detected_languages.add(doc_rec["source_language"])

                self._persist_document(
                    doc_rec=doc_rec,
                    website_id=website_id,
                    clean_url=clean_url,
                    resp=resp,
                    english_alt_url=english_alt,
                )
                pages_indexed += 1
                indexed_chunks += doc_rec["num_chunks"]

                # Extract out-links if within depth
                if depth < self.max_depth and len(discovered_urls) < self.max_pages * 2:
                    links = self._extract_links(html_text, clean_url, domain)
                    for l in links:
                        if l not in discovered_urls:
                            discovered_urls.add(l)
                            queue.append((l, depth + 1))

            except requests.exceptions.Timeout:
                pages_failed += 1
                self.db.record_crawl_error(current_job_id, website_id, clean_url, "TIMEOUT", 408, "Request timed out")
            except requests.exceptions.RequestException as e:
                pages_failed += 1
                self.db.record_crawl_error(current_job_id, website_id, clean_url, "NETWORK_ERROR", None, str(e))
            except Exception as e:
                pages_failed += 1
                logger.error(f"Unexpected error crawling {clean_url}: {e}")
                self.db.record_crawl_error(current_job_id, website_id, clean_url, "PARSE_ERROR", None, str(e))

        duration = time.time() - start_time
        final_status = "CANCELLED" if self._stop_requested else ("COMPLETED" if pages_indexed > 0 else "FAILED")

        self.db.finish_crawl_job(
            job_id=current_job_id,
            status=final_status,
            discovered=len(discovered_urls),
            crawled=pages_crawled,
            indexed=pages_indexed,
            failed=pages_failed,
            duration=duration,
        )

        self.db.update_website_crawl_stats(
            website_id=website_id,
            status=final_status,
            pages_discovered=len(discovered_urls),
            pages_indexed=pages_indexed,
            pages_failed=pages_failed,
            indexed_chunks=indexed_chunks,
            languages=list(detected_languages),
        )

        return {
            "status": final_status,
            "website_id": website_id,
            "pages_crawled": pages_crawled,
            "pages_indexed": pages_indexed,
            "pages_failed": pages_failed,
            "indexed_chunks": indexed_chunks,
            "duration_seconds": round(duration, 2),
            "languages": list(detected_languages),
        }

    def _persist_document(
        self,
        doc_rec: Dict[str, Any],
        website_id: str,
        clean_url: str,
        resp: requests.Response,
        english_alt_url: Optional[str] = None,
    ):
        """Save crawled document to SQLite and index chunks into ChromaDB."""
        page_id = f"pg_{uuid.uuid4().hex[:10]}"
        doc_id = doc_rec["id"]

        self.db.record_crawled_page(
            page_id=page_id,
            website_id=website_id,
            url=clean_url,
            title=doc_rec["title"],
            status="INDEXED",
            content_hash=doc_rec["content_hash"],
            doc_type=doc_rec["document_type"],
            language=doc_rec["source_language"],
            has_english=bool(english_alt_url),
            english_url=english_alt_url,
            etag=resp.headers.get("ETag"),
            last_modified=resp.headers.get("Last-Modified"),
            http_status=resp.status_code,
            content_length=len(resp.content),
        )

        self.db.record_document(
            doc_id=doc_id,
            page_id=page_id,
            website_id=website_id,
            title=doc_rec["title"],
            url=clean_url,
            domain=doc_rec["domain"],
            department=doc_rec["department"],
            publication_date=doc_rec.get("publication_date"),
            doc_type=doc_rec["document_type"],
            source_language=doc_rec["source_language"],
            original_text=doc_rec["original_text"][:5000],  # preview length
            english_text=doc_rec["english_text"][:5000] if doc_rec["english_text"] else "",
            is_translated=doc_rec["is_translated"],
            content_hash=doc_rec["content_hash"],
            num_chunks=doc_rec["num_chunks"],
        )

        # Index chunks in Vector Database
        if doc_rec.get("chunks"):
            self.vector_store.add_chunks(doc_rec["chunks"])

    def _get_robot_parser(self, base_url: str) -> Optional[RobotFileParser]:
        """Fetch and parse robots.txt for domain."""
        try:
            parsed = urlparse(base_url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp
        except Exception:
            return None

    def _discover_sitemap_urls(self, base_url: str) -> List[str]:
        """Check standard sitemap.xml locations for seeds."""
        discovered = []
        try:
            parsed = urlparse(base_url)
            sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
            resp = self.session.get(sitemap_url, timeout=5)
            if resp.status_code == 200 and ("xml" in resp.headers.get("Content-Type", "") or "<urlset" in resp.text):
                soup = BeautifulSoup(resp.text, "html.parser")
                for loc in soup.find_all("loc"):
                    u = loc.get_text(strip=True)
                    if is_approved_domain(u):
                        discovered.append(u)
                        if len(discovered) >= 50:
                            break
        except Exception:
            pass
        return discovered

    def _extract_links(self, html: str, current_url: str, current_domain: str) -> List[str]:
        """Extract valid, approved in-domain links from HTML."""
        links = []
        try:
            soup = BeautifulSoup(html, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                # Exclude javascript:, mailto:, tel:, #
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue

                full_url = urljoin(current_url, href)
                parsed = urlparse(full_url)

                # Skip search queries and session loops
                if any(param in parsed.query for param in ["page_calendar", "sort=", "session="]):
                    continue

                if is_approved_domain(parsed.netloc):
                    links.append(full_url)
        except Exception:
            pass
        return links
