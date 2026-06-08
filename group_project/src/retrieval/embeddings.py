"""OpenAI embedding helpers for text-embedding-3-small."""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from openai import OpenAI

from src.config import CONFIG


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY chua co trong group_project/.env")
    return OpenAI()


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed multiple texts with OpenAI text-embedding-3-small."""
    vectors: list[list[float]] = []
    client = _client()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=CONFIG.openai_embedding_model,
            input=batch,
        )
        vectors.extend([item.embedding for item in response.data])
    return vectors


def embed_query(query: str) -> list[float]:
    """Embed one query."""
    return embed_texts([query], batch_size=1)[0]


def cosine_similarity(query_vector: list[float], matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity for local fallback search."""
    query = np.asarray(query_vector, dtype=np.float32)
    query_norm = np.linalg.norm(query)
    if query_norm:
        query = query / query_norm

    vectors = matrix.astype(np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    return vectors @ query
