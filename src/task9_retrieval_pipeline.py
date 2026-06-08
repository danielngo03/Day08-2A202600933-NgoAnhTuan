"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .task5_semantic_search import semantic_search
    from .task6_lexical_search import lexical_search
    from .task7_reranking import rerank, rerank_rrf
    from .task8_pageindex_vectorless import pageindex_search
except ImportError:
    from src.task5_semantic_search import semantic_search
    from src.task6_lexical_search import lexical_search
    from src.task7_reranking import rerank, rerank_rrf
    from src.task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Neu best score < threshold -> fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"
RETRIEVAL_MULTIPLIER = 4
RRF_K = 60


def _validate_query(query: str, top_k: int) -> str:
    query = query.strip()
    if not query:
        raise ValueError("query khong duoc rong.")
    if top_k <= 0:
        return ""
    return query


def _safe_search(name: str, search_fn, query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Chay tung retriever rieng biet va khong de 1 module loi lam sap pipeline.
    Loi duoc gan vao warning de demo/debug de hon.
    """
    try:
        return search_fn(query, top_k=top_k)
    except Exception as exc:
        print(f"  ! {name} search failed: {exc}")
        return []


def _copy_result(result: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(result)
    item.setdefault("content", "")
    item["score"] = float(item.get("score", 0.0) or 0.0)
    item.setdefault("metadata", {})
    return item


def _tag_results(results: list[dict[str, Any]], retriever: str) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for result in results:
        item = _copy_result(result)
        item["metadata"]["retriever"] = retriever
        item["metadata"][f"{retriever}_score"] = item["score"]
        tagged.append(item)
    return tagged


def _normalize_results(results: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for result in results:
        item = _copy_result(result)
        item["source"] = source
        item["metadata"]["source"] = source
        normalized.append(item)
    normalized.sort(key=lambda item: item["score"], reverse=True)
    return normalized


def _best_score(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return float(results[0].get("score", 0.0) or 0.0)


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Hybrid retrieval = dense semantic search + BM25 lexical search + RRF fusion.

    RRF (Reciprocal Rank Fusion) duoc chon vi no on dinh khi 2 retriever co
    thang diem khac nhau: cosine similarity cua dense va BM25 score khong can
    normalize truc tiep, chi can rank cua moi list.
    """
    query = _validate_query(query, top_k)
    if not query:
        return []

    candidate_k = max(top_k * RETRIEVAL_MULTIPLIER, top_k, 10)
    dense_results = _tag_results(
        _safe_search("semantic", semantic_search, query, candidate_k),
        "semantic",
    )
    sparse_results = _tag_results(
        _safe_search("lexical", lexical_search, query, candidate_k),
        "lexical",
    )

    merged = rerank_rrf(
        [dense_results, sparse_results],
        top_k=max(top_k * 3, top_k),
        k=RRF_K,
    )
    for item in merged:
        item.setdefault("metadata", {})
        item["source"] = "hybrid"
        item["metadata"]["source"] = "hybrid"
        item["metadata"]["fusion_method"] = "rrf"
        item["metadata"]["rrf_k"] = RRF_K
        item["metadata"]["candidate_k_per_retriever"] = candidate_k

    return merged


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = _validate_query(query, top_k)
    if not query:
        return []

    merged = hybrid_search(query, top_k=top_k)

    if use_reranking and merged:
        reranked = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
        final_results = _normalize_results(reranked, "hybrid")
        for item in final_results:
            item["metadata"]["pipeline"] = "semantic_bm25_rrf_jina"
    else:
        final_results = _normalize_results(merged[:top_k], "hybrid")
        for item in final_results:
            item["metadata"]["pipeline"] = "semantic_bm25_rrf"

    best_score = _best_score(final_results)
    if best_score < score_threshold:
        print(
            f"  ! Hybrid best score ({best_score:.3f}) < threshold "
            f"({score_threshold:.3f}). Fallback -> PageIndex"
        )
        fallback_results = _normalize_results(pageindex_search(query, top_k=top_k), "pageindex")
        for item in fallback_results:
            item["metadata"]["pipeline"] = "pageindex_fallback"
            item["metadata"]["fallback_reason"] = (
                "empty_hybrid_results" if not final_results else "low_hybrid_score"
            )
            item["metadata"]["hybrid_best_score"] = best_score
            item["metadata"]["score_threshold"] = float(score_threshold)
        if fallback_results:
            return fallback_results[:top_k]

    return final_results[:top_k]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 9 complete retrieval pipeline")
    parser.add_argument(
        "query",
        nargs="?",
        default="Hình phạt cho tội tàng trữ trái phép chất ma tuý",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = retrieve(
        args.query,
        top_k=args.top_k,
        score_threshold=args.threshold,
        use_reranking=not args.no_rerank,
    )
    print(f"\nQuery: {args.query}")
    print("-" * 80)
    for index, result in enumerate(results, 1):
        source = result.get("source", "unknown")
        score = float(result.get("score", 0.0) or 0.0)
        pipeline = result.get("metadata", {}).get("pipeline", "")
        print(f"{index}. [{score:.4f}] [{source}] [{pipeline}]")
        print(f"   {result.get('content', '')[:220]}...")
