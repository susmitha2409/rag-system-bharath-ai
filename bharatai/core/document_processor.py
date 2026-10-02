"""
BharatAI — Document Processing and Chunking Pipeline
Extracts text from HTML, PDF, DOCX, TXT, and Markdown; normalizes content;
performs heading-aware sliding window chunking; and generates rich metadata.
"""

import re
import hashlib
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

from config import CHUNK_SIZE, CHUNK_OVERLAP
from core.language_processor import get_language_processor
from core.security import sanitize_retrieved_evidence

logger = logging.getLogger("bharatai.document_processor")


class DocumentProcessor:
    def __init__(self, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.lang_processor = get_language_processor()

    def process_html(
        self,
        html_content: str,
        url: str,
        department: str,
        domain: str,
        title_override: Optional[str] = None,
        pub_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract clean text and metadata from raw HTML page."""
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html_content, "html.parser")

            # Strip scripts, styles, navigations, footers, headers
            for tag in soup(["script", "style", "nav", "footer", "header", "noscript", "aside", "svg"]):
                tag.decompose()

            # Extract title
            title = title_override
            if not title:
                h1 = soup.find("h1")
                if h1:
                    title = h1.get_text(strip=True)
                elif soup.title:
                    title = soup.title.get_text(strip=True)
                else:
                    title = "Official Document"

            # Look for publication date in meta tags or date patterns
            if not pub_date:
                date_meta = soup.find("meta", property=re.compile(r"date|published_time", re.I)) or soup.find(
                    "meta", attrs={"name": re.compile(r"date|pubdate", re.I)}
                )
                if date_meta and date_meta.get("content"):
                    pub_date = date_meta["content"][:10]

            # Extract main text
            raw_text = soup.get_text(separator="\n", strip=True)
        else:
            no_scripts = re.sub(r"(?is)<(script|style|nav|footer|header|noscript|aside).*?>.*?</\1>", " ", html_content)
            title = title_override
            if not title:
                m_title = re.search(r"(?i)<title>(.*?)</title>", html_content)
                title = m_title.group(1).strip() if m_title else "Official Document"
            raw_text = re.sub(r"<[^>]+>", "\n", no_scripts)
        return self._build_document_record(
            raw_text=raw_text,
            url=url,
            domain=domain,
            department=department,
            title=title,
            publication_date=pub_date,
            doc_type="html",
        )

    def process_pdf(
        self,
        pdf_bytes: bytes,
        url: str,
        department: str,
        domain: str,
        title: Optional[str] = None,
        pub_date: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract text from PDF using PyMuPDF (fitz) with fallback."""
        text_content = ""
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages_text = []
            for page in doc:
                pages_text.append(page.get_text())
            text_content = "\n".join(pages_text)
        except Exception as e:
            logger.warning(f"PyMuPDF error: {e}. Attempting pypdf / raw stream extraction.")
            try:
                import pypdf
                import io
                reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
                pages = [p.extract_text() or "" for p in reader.pages]
                text_content = "\n".join(pages)
            except Exception as e2:
                logger.error(f"Failed to extract text from PDF: {e2}")

        doc_title = title or (url.split("/")[-1] if "/" in url else "Official PDF")
        return self._build_document_record(
            raw_text=text_content,
            url=url,
            domain=domain,
            department=department,
            title=doc_title,
            publication_date=pub_date,
            doc_type="pdf",
        )

    def process_docx(
        self,
        docx_bytes: bytes,
        url: str,
        department: str,
        domain: str,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract text from DOCX file."""
        text_content = ""
        try:
            import io
            import docx
            doc = docx.Document(io.BytesIO(docx_bytes))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text_content = "\n\n".join(paragraphs)
        except Exception as e:
            logger.error(f"Error reading docx: {e}")

        doc_title = title or (url.split("/")[-1] if "/" in url else "Official DOCX Document")
        return self._build_document_record(
            raw_text=text_content,
            url=url,
            domain=domain,
            department=department,
            title=doc_title,
            doc_type="docx",
        )

    def process_plain_text(
        self,
        text: str,
        url: str,
        department: str,
        domain: str,
        title: Optional[str] = None,
        doc_type: str = "txt",
    ) -> Dict[str, Any]:
        """Process plain text or markdown."""
        doc_title = title or "Official Notification"
        return self._build_document_record(
            raw_text=text,
            url=url,
            domain=domain,
            department=department,
            title=doc_title,
            doc_type=doc_type,
        )

    def _build_document_record(
        self,
        raw_text: str,
        url: str,
        domain: str,
        department: str,
        title: str,
        publication_date: Optional[str] = None,
        doc_type: str = "html",
    ) -> Dict[str, Any]:
        """Clean text, compute content hash, detect language, and chunk document."""
        clean_text = self._normalize_whitespace(raw_text)
        clean_text = sanitize_retrieved_evidence(clean_text)

        content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
        doc_id = f"doc_{content_hash[:16]}"
        detected_lang = self.lang_processor.detect_language(clean_text[:2000])

        chunks = self.chunk_text(
            text=clean_text,
            doc_id=doc_id,
            url=url,
            domain=domain,
            department=department,
            title=title,
            publication_date=publication_date,
            source_language=detected_lang,
            doc_type=doc_type,
            content_hash=content_hash,
        )

        return {
            "id": doc_id,
            "title": title,
            "url": url,
            "domain": domain,
            "department": department,
            "publication_date": publication_date,
            "document_type": doc_type,
            "source_language": detected_lang,
            "original_text": clean_text,
            "english_text": clean_text if detected_lang == "en" else "",
            "is_translated": False,
            "content_hash": content_hash,
            "chunks": chunks,
            "num_chunks": len(chunks),
        }

    def chunk_text(
        self,
        text: str,
        doc_id: str,
        url: str,
        domain: str,
        department: str,
        title: str,
        publication_date: Optional[str],
        source_language: str,
        doc_type: str,
        content_hash: str,
    ) -> List[Dict[str, Any]]:
        """
        Heading and sentence-aware sliding window chunking.
        Creates overlapping chunks with full source attribution.
        """
        if not text or len(text.strip()) == 0:
            return []

        # Split into semantic paragraphs / sections
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        chunks: List[Dict[str, Any]] = []

        current_chunk_words = []
        current_len = 0
        chunk_idx = 0
        now_ts = datetime.utcnow().isoformat()

        # Target chunk size in characters roughly converts to words (avg 5 chars per word)
        target_words = self.chunk_size // 5
        overlap_words = self.chunk_overlap // 5

        for para in paragraphs:
            words = para.split()
            if not words:
                continue

            # If adding this paragraph exceeds target words, seal current chunk
            if current_len + len(words) > target_words and current_chunk_words:
                chunk_str = " ".join(current_chunk_words)
                cid = f"{doc_id}_c{chunk_idx}"
                chunks.append(
                    {
                        "chunk_id": cid,
                        "document_id": doc_id,
                        "source_url": url,
                        "source_domain": domain,
                        "source_title": title,
                        "department": department,
                        "publication_date": publication_date or "",
                        "crawl_timestamp": now_ts,
                        "indexed_timestamp": now_ts,
                        "source_language": source_language,
                        "original_text": chunk_str,
                        "english_text": chunk_str,  # Will be translated later if needed
                        "is_translated": False,
                        "content_hash": content_hash,
                        "chunk_index": chunk_idx,
                        "document_type": doc_type,
                    }
                )
                chunk_idx += 1
                # Sliding window overlap
                current_chunk_words = current_chunk_words[-overlap_words:] if overlap_words < len(current_chunk_words) else []
                current_len = len(current_chunk_words)

            current_chunk_words.extend(words)
            current_len += len(words)

        # Remaining words
        if current_chunk_words:
            chunk_str = " ".join(current_chunk_words)
            cid = f"{doc_id}_c{chunk_idx}"
            chunks.append(
                {
                    "chunk_id": cid,
                    "document_id": doc_id,
                    "source_url": url,
                    "source_domain": domain,
                    "source_title": title,
                    "department": department,
                    "publication_date": publication_date or "",
                    "crawl_timestamp": now_ts,
                    "indexed_timestamp": now_ts,
                    "source_language": source_language,
                    "original_text": chunk_str,
                    "english_text": chunk_str,
                    "is_translated": False,
                    "content_hash": content_hash,
                    "chunk_index": chunk_idx,
                    "document_type": doc_type,
                }
            )

        return chunks

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse redundant tabs and spaces while preserving paragraph breaks."""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()


# Singleton document processor instance
_doc_processor = None

def get_document_processor() -> DocumentProcessor:
    global _doc_processor
    if _doc_processor is None:
        _doc_processor = DocumentProcessor()
    return _doc_processor
