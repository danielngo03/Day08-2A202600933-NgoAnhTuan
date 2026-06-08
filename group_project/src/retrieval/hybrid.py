"""Hybrid retrieval over Weaviate dense + vectorless BM25."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any

from src.config import CONFIG
from src.retrieval.local_vector_store import local_dense_search
from src.retrieval.vectorless_bm25 import vectorless_search
from src.retrieval.weaviate_store import weaviate_dense_search


def _candidate_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    return str(metadata.get("chunk_id") or item.get("id") or item.get("content", ""))


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Merge retrievers without comparing incompatible raw scores."""
    scores: dict[str, float] = defaultdict(float)
    best_items: dict[str, dict[str, Any]] = {}
    retrievers: dict[str, set[str]] = defaultdict(set)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            if not key:
                continue
            scores[key] += 1.0 / (rrf_k + rank)
            retriever_name = item.get("metadata", {}).get("retriever", item.get("source", "unknown"))
            retrievers[key].add(str(retriever_name))
            if key not in best_items or item.get("score", 0.0) > best_items[key].get("score", 0.0):
                best_items[key] = item

    fused: list[dict[str, Any]] = []
    for key, score in scores.items():
        item = copy.deepcopy(best_items[key])
        item["score"] = float(score)
        item["source"] = "hybrid_group"
        item.setdefault("metadata", {})
        item["metadata"]["fusion_method"] = "rrf"
        item["metadata"]["rrf_score"] = float(score)
        item["metadata"]["retrievers"] = sorted(retrievers[key])
        fused.append(item)

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def dense_search(query: str, top_k: int, query_vector: list[float] | None = None) -> list[dict[str, Any]]:
    """Use Weaviate when available, otherwise fallback to personal local index."""
    try:
        return weaviate_dense_search(query, top_k=top_k, query_vector=query_vector)
    except Exception as exc:
        results = local_dense_search(query, top_k=top_k, query_vector=query_vector)
        for item in results:
            item["metadata"]["weaviate_error"] = str(exc)[:240]
        return results


def hybrid_search(
    query: str,
    top_k: int | None = None,
    query_vector: list[float] | None = None,
    lexical_method: str = "bm25",
) -> list[dict[str, Any]]:
    """Run dense + vectorless retrieval and fuse with RRF."""
    selected_top_k = top_k or CONFIG.retrieval_top_k
    dense = dense_search(query, top_k=selected_top_k, query_vector=query_vector)
    vectorless = vectorless_search(query, top_k=selected_top_k, method=lexical_method)
    return reciprocal_rank_fusion([dense, vectorless], top_k=selected_top_k)

