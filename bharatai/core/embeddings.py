"""
BharatAI — Local Embedding Model
Generates embeddings locally using sentence-transformers without hosted APIs.
Includes deterministic caching and resilient fallback.
"""

import time
import math
import logging
from typing import List, Union
try:
    import numpy as np
except ImportError:
    np = None
from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger("bharatai.embeddings")


class LocalEmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._cache = {}
        self._is_transformer_available = False
        self._init_model()

    def _init_model(self):
        """Attempt to load sentence-transformers model locally."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading local embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._is_transformer_available = True
            logger.info("Local sentence-transformers model loaded successfully.")
        except Exception as e:
            logger.warning(
                f"SentenceTransformer could not be loaded ({e}). Falling back to deterministic local feature hashing embeddings."
            )
            self._is_transformer_available = False

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate normalized embedding vectors for a list of text strings.
        Returns List of float vectors of fixed dimension (384 for MiniLM).
        """
        if not texts:
            return []

        if self._is_transformer_available and self._model is not None:
            try:
                embeddings = self._model.encode(
                    texts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    batch_size=32,
                    show_progress_bar=False,
                )
                return [vec.tolist() for vec in embeddings]
            except Exception as e:
                logger.error(f"Error during transformer embedding generation: {e}. Using fallback.")

        # Fallback: Deterministic hashing/n-gram vectorizer of dimension 384
        return [self._fallback_embed(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding vector for a single search query with caching."""
        cache_key = f"{self.model_name}:{hash(query)}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        embeddings = self.embed_texts([query])
        vec = embeddings[0] if embeddings else [0.0] * 384
        self._cache[cache_key] = vec
        return vec

    def _fallback_embed(self, text: str, dim: int = 384) -> List[float]:
        """
        Deterministic, local, fast hashing embedding when PyTorch/sentence-transformers
        model weights are unavailable. Produces normalized vector with semantic bag-of-words / char n-gram properties.
        """
        if not text:
            return [0.0] * dim

        if np is not None:
            vec = np.zeros(dim, dtype=np.float32)
            words = text.lower().split()
            for w in words:
                h = abs(hash(w)) % dim
                vec[h] += 1.0

            for i in range(max(0, len(text) - 2)):
                tri = text[i : i + 3].lower()
                h = abs(hash(tri)) % dim
                vec[h] += 0.5

            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        else:
            raw_vec = [0.0] * dim
            words = text.lower().split()
            for w in words:
                h = abs(hash(w)) % dim
                raw_vec[h] += 1.0

            for i in range(max(0, len(text) - 2)):
                tri = text[i : i + 3].lower()
                h = abs(hash(tri)) % dim
                raw_vec[h] += 0.5

            norm = math.sqrt(sum(x * x for x in raw_vec))
            if norm > 0:
                return [round(x / norm, 6) for x in raw_vec]
            return raw_vec


# Singleton embedding model instance
_embedding_model_instance = None

def get_embedding_model() -> LocalEmbeddingModel:
    global _embedding_model_instance
    if _embedding_model_instance is None:
        _embedding_model_instance = LocalEmbeddingModel()
    return _embedding_model_instance
