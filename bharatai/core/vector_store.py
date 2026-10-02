"""
BharatAI — Persistent Vector Database (ChromaDB + Resilient Fallback)
Provides collection management, chunk indexing with metadata, cosine similarity search,
metadata filtering, and deletion.
"""

import os
import json
import math
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
try:
    import numpy as np
except ImportError:
    np = None

from config import CHROMA_PERSIST_DIRECTORY
from core.embeddings import get_embedding_model

logger = logging.getLogger("bharatai.vector_store")


class PersistentVectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIRECTORY):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = "bharatai_gov_docs"
        self.embedding_model = get_embedding_model()

        self._chroma_client = None
        self._collection = None
        self._is_chroma_ready = False

        # In-memory/file fallback index if chromadb cannot load due to system sqlite requirements
        self._fallback_docs: Dict[str, Dict[str, Any]] = {}
        self._fallback_vectors: Dict[str, np.ndarray] = {}
        self._fallback_path = self.persist_dir / "fallback_index.json"

        self._init_store()

    def _init_store(self):
        """Initialize ChromaDB or prepare fallback store."""
        try:
            import chromadb
            from chromadb.config import Settings

            self._chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Official Government of India Knowledge Base"},
            )
            self._is_chroma_ready = True
            logger.info("ChromaDB persistent collection initialized successfully.")
        except Exception as e:
            logger.warning(
                f"ChromaDB persistent client initialization error: {e}. Using resilient local index."
            )
            self._is_chroma_ready = False
            self._load_fallback_index()

    def _load_fallback_index(self):
        """Load fallback document index from disk."""
        if self._fallback_path.exists():
            try:
                with open(self._fallback_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._fallback_docs = data.get("docs", {})
                    for cid, vec in data.get("vectors", {}).items():
                        if np is not None:
                            self._fallback_vectors[cid] = np.array(vec, dtype=np.float32)
                        else:
                            self._fallback_vectors[cid] = vec
                logger.info(f"Loaded {len(self._fallback_docs)} chunks from local fallback index.")
            except Exception as e:
                logger.error(f"Failed to load fallback index: {e}")

    def _save_fallback_index(self):
        """Save fallback document index to disk."""
        try:
            serializable_vectors = {
                cid: (vec.tolist() if hasattr(vec, "tolist") else vec)
                for cid, vec in self._fallback_vectors.items()
            }
            with open(self._fallback_path, "w", encoding="utf-8") as f:
                json.dump({"docs": self._fallback_docs, "vectors": serializable_vectors}, f)
        except Exception as e:
            logger.error(f"Failed to save fallback index: {e}")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Add document chunks to the vector database.
        Each chunk must have: chunk_id, english_text (or original_text), source_url, etc.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for c in chunks:
            cid = c.get("chunk_id")
            text = c.get("english_text") or c.get("original_text") or ""
            if not cid or not text.strip():
                continue

            ids.append(cid)
            documents.append(text)

            # Sanitize metadata for ChromaDB (no nested dicts, string/int/float/bool only)
            meta = {
                "document_id": str(c.get("document_id", "")),
                "source_url": str(c.get("source_url", "")),
                "source_domain": str(c.get("source_domain", "")),
                "source_title": str(c.get("source_title", "")),
                "department": str(c.get("department", "")),
                "publication_date": str(c.get("publication_date", "")),
                "source_language": str(c.get("source_language", "en")),
                "is_translated": 1 if c.get("is_translated", False) else 0,
                "content_hash": str(c.get("content_hash", "")),
                "chunk_index": int(c.get("chunk_index", 0)),
                "document_type": str(c.get("document_type", "html")),
            }
            metadatas.append(meta)

        if not ids:
            return 0

        # Compute embeddings locally
        embeddings = self.embedding_model.embed_texts(documents)

        if self._is_chroma_ready and self._collection is not None:
            try:
                self._collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
                return len(ids)
            except Exception as e:
                logger.error(f"ChromaDB upsert failed ({e}), saving to fallback index.")

        # Save to fallback store
        for i, cid in enumerate(ids):
            self._fallback_docs[cid] = {
                "chunk_id": cid,
                "text": documents[i],
                "metadata": metadatas[i],
            }
            if np is not None:
                self._fallback_vectors[cid] = np.array(embeddings[i], dtype=np.float32)
            else:
                self._fallback_vectors[cid] = embeddings[i]

        self._save_fallback_index()
        return len(ids)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        department_filter: Optional[str] = None,
        domain_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic similarity search against the vector database.
        Returns list of chunks with similarity_score (0.0 to 1.0) and all metadata.
        """
        query_embedding = self.embedding_model.embed_query(query)

        # Build metadata where filter if requested
        where_filter = {}
        if department_filter and department_filter != "All":
            where_filter["department"] = department_filter
        if domain_filter and domain_filter != "All":
            where_filter["source_domain"] = domain_filter

        if self._is_chroma_ready and self._collection is not None:
            try:
                kwargs = {
                    "query_embeddings": [query_embedding],
                    "n_results": min(top_k * 2, max(self._collection.count(), 1)),
                }
                if where_filter:
                    if len(where_filter) == 1:
                        kwargs["where"] = where_filter
                    else:
                        kwargs["where"] = {"$and": [{k: v} for k, v in where_filter.items()]}

                results = self._collection.query(**kwargs)

                retrieved = []
                if results and "ids" in results and results["ids"]:
                    ids = results["ids"][0]
                    docs = results["documents"][0]
                    metas = results["metadatas"][0]
                    distances = results["distances"][0] if "distances" in results and results["distances"] else [0.0] * len(ids)

                    for i, cid in enumerate(ids):
                        dist = distances[i] if i < len(distances) else 1.0
                        similarity = max(0.0, min(1.0, 1.0 - (dist / 2.0)))

                        meta = metas[i] if i < len(metas) else {}
                        chunk_item = {
                            "chunk_id": cid,
                            "english_text": docs[i] if i < len(docs) else "",
                            "original_text": docs[i] if i < len(docs) else "",
                            "similarity_score": round(similarity, 4),
                            "source_url": meta.get("source_url", ""),
                            "source_domain": meta.get("source_domain", ""),
                            "source_title": meta.get("source_title", ""),
                            "department": meta.get("department", ""),
                            "publication_date": meta.get("publication_date", ""),
                            "source_language": meta.get("source_language", "en"),
                            "is_translated": bool(meta.get("is_translated", 0)),
                            "document_type": meta.get("document_type", "html"),
                            "document_id": meta.get("document_id", ""),
                        }
                        retrieved.append(chunk_item)

                    return retrieved[:top_k]
            except Exception as e:
                logger.error(f"ChromaDB search failed: {e}. Searching fallback index.")

        # Fallback search using cosine similarity
        retrieved = []
        if np is not None:
            q_vec = np.array(query_embedding, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0:
                q_vec = q_vec / q_norm
        else:
            q_norm = math.sqrt(sum(x * x for x in query_embedding))

        for cid, doc_info in self._fallback_docs.items():
            meta = doc_info["metadata"]

            # Filter checks
            if department_filter and department_filter != "All" and meta.get("department") != department_filter:
                continue
            if domain_filter and domain_filter != "All" and meta.get("source_domain") != domain_filter:
                continue

            d_vec = self._fallback_vectors.get(cid)
            if d_vec is not None:
                if np is not None:
                    d_norm = np.linalg.norm(d_vec)
                    score = float(np.dot(q_vec, d_vec / d_norm)) if d_norm > 0 else 0.0
                else:
                    dot_val = sum(a * b for a, b in zip(query_embedding, d_vec))
                    d_norm = math.sqrt(sum(b * b for b in d_vec))
                    score = (dot_val / (q_norm * d_norm)) if (q_norm * d_norm) > 0 else 0.0
            else:
                score = 0.0

            # Normalize to 0.0 - 1.0
            sim_score = max(0.0, min(1.0, (score + 1.0) / 2.0))

            chunk_item = {
                "chunk_id": cid,
                "english_text": doc_info["text"],
                "original_text": doc_info["text"],
                "similarity_score": round(sim_score, 4),
                "source_url": meta.get("source_url", ""),
                "source_domain": meta.get("source_domain", ""),
                "source_title": meta.get("source_title", ""),
                "department": meta.get("department", ""),
                "publication_date": meta.get("publication_date", ""),
                "source_language": meta.get("source_language", "en"),
                "is_translated": bool(meta.get("is_translated", 0)),
                "document_type": meta.get("document_type", "html"),
                "document_id": meta.get("document_id", ""),
            }
            retrieved.append(chunk_item)

        retrieved.sort(key=lambda x: x["similarity_score"], reverse=True)
        return retrieved[:top_k]

    def count(self) -> int:
        """Return total indexed chunk count."""
        if self._is_chroma_ready and self._collection is not None:
            try:
                return self._collection.count()
            except Exception:
                pass
        return len(self._fallback_docs)

    def delete_document_chunks(self, document_id: str):
        """Remove all chunks associated with a document_id."""
        if self._is_chroma_ready and self._collection is not None:
            try:
                self._collection.delete(where={"document_id": document_id})
            except Exception as e:
                logger.error(f"ChromaDB chunk deletion failed: {e}")

        to_del = [
            cid
            for cid, doc in self._fallback_docs.items()
            if doc["metadata"].get("document_id") == document_id
        ]
        for cid in to_del:
            self._fallback_docs.pop(cid, None)
            self._fallback_vectors.pop(cid, None)
        if to_del:
            self._save_fallback_index()


# Singleton vector store instance
_vector_store_instance = None

def get_vector_store() -> PersistentVectorStore:
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = PersistentVectorStore()
    return _vector_store_instance
