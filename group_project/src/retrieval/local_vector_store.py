"""Local dense fallback using the personal Task 4 vector store."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from src.config import CONFIG
from src.retrieval.embeddings import cosine_similarity, embed_query

LOCAL_INDEX_DIR = CONFIG.local_dense_index_dir


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


@lru_cache(maxsize=1)
def _load_store() -> tuple[list[dict[str, Any]], np.ndarray]:
    chunks_path = LOCAL_INDEX_DIR / "chunks.jsonl"
    embeddings_path = LOCAL_INDEX_DIR / "embeddings.npy"
    if not chunks_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Local OpenAI vector index chua ton tai. Hay chay Task 4 truoc.")
    return _read_jsonl(chunks_path), np.load(embeddings_path).astype(np.float32)


def local_dense_search(query: str, top_k: int = 10, query_vector: list[float] | None = None) -> list[dict[str, Any]]:
    """Dense search over data/indexes/openai_text_embedding_3_small."""
    chunks, embeddings = _load_store()
    vector = query_vector if query_vector is not None else embed_query(query)
    scores = cosine_similarity(vector, embeddings)
    top_indices = np.argsort(scores)[::-1][:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(top_indices, start=1):
        row = chunks[int(index)]
        results.append(
            {
                "content": row["content"],
                "score": float(scores[int(index)]),
                "metadata": {
                    **row.get("metadata", {}),
                    "retriever": "local_dense_fallback",
                    "rank": rank,
                    "chunk_id": row.get("id") or row.get("metadata", {}).get("chunk_id"),
                },
                "source": "local_dense",
            }
        )
    return results
