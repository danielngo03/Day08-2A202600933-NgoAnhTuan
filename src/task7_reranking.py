"""
Task 7 - Reranking Module.

Primary method:
    Jina Reranker API - jina-reranker-v2-base-multilingual.
    Day la cross-encoder reranker: model nhin dong thoi query + document,
    roi cham relevance_score cho tung candidate. Phu hop tieng Viet va
    ngon ngu da dang trong corpus.

Fallback / bonus:
    - Lexical fallback tu implement khi khong co JINA_API_KEY hoac API loi.
    - RRF (Reciprocal Rank Fusion) tu implement de gop dense + BM25 o Task 9.

Env:
    JINA_API_KEY=...
"""

from __future__ import annotations

import argparse
import copy
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"
JINA_RERANK_MODEL = "jina-reranker-v2-base-multilingual"
DEFAULT_TOP_K = 5


def _get_jina_api_key() -> str | None:
    load_dotenv(ENV_PATH)
    return os.getenv("JINA_API_KEY") or os.getenv("Jina_API_Key") or os.getenv("JINA_APIKEY")


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return text


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


def _candidate_key(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata") or {}
    return str(
        metadata.get("chunk_id")
        or metadata.get("id")
        or candidate.get("id")
        or candidate.get("content", "")
    )


def _copy_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    cloned = copy.deepcopy(candidate)
    cloned.setdefault("metadata", {})
    return cloned


def _fallback_relevance_score(query: str, content: str, original_score: float = 0.0) -> float:
    """
    Lightweight fallback khi Jina API khong san sang.

    Ket hop lexical token overlap voi score retrieval ban dau. Fallback nay
    khong thay the cross-encoder, nhung giup module/test/demo khong bi crash.
    """
    query_tokens = set(_tokenize(query))
    doc_tokens = _tokenize(content)
    if not query_tokens or not doc_tokens:
        return float(original_score)

    doc_token_set = set(doc_tokens)
    overlap = len(query_tokens & doc_token_set) / len(query_tokens)
    density = sum(1 for token in doc_tokens if token in query_tokens) / max(len(doc_tokens), 1)
    return 0.75 * overlap + 0.15 * density + 0.10 * float(original_score)


def rerank_fallback(query: str, candidates: list[dict[str, Any]], top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """Rerank local khong can API, dung khi Jina khong kha dung."""
    scored: list[dict[str, Any]] = []

    for candidate in candidates:
        item = _copy_candidate(candidate)
        original_score = float(item.get("score", 0.0) or 0.0)
        score = _fallback_relevance_score(query, item.get("content", ""), original_score)
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_method"] = "lexical_fallback"
        item["metadata"]["rerank_method"] = "lexical_fallback"
        item["metadata"]["original_score"] = original_score
        scored.append(item)

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def rerank_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    timeout: int = 30,
    fallback_on_error: bool = True,
) -> list[dict[str, Any]]:
    """
    Rerank candidates bang Jina cross-encoder API.

    Args:
        query: Cau truy van.
        candidates: List of {"content": str, "score": float, "metadata": dict}.
        top_k: So ket qua sau rerank.
        timeout: Request timeout.
        fallback_on_error: True thi dung fallback local neu API loi.

    Returns:
        List top_k candidates, score moi la Jina relevance_score.
    """
    if top_k <= 0 or not candidates:
        return []

    api_key = _get_jina_api_key()
    if not api_key:
        if fallback_on_error:
            return rerank_fallback(query, candidates, top_k)
        raise RuntimeError("JINA_API_KEY chua co trong .env.")

    documents = [candidate.get("content", "") for candidate in candidates]
    payload = {
        "model": JINA_RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": min(top_k, len(candidates)),
    }

    try:
        response = requests.post(
            JINA_RERANK_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        if fallback_on_error:
            return rerank_fallback(query, candidates, top_k)
        raise

    reranked: list[dict[str, Any]] = []
    for result in data.get("results", []):
        index = int(result["index"])
        if index < 0 or index >= len(candidates):
            continue

        item = _copy_candidate(candidates[index])
        original_score = float(item.get("score", 0.0) or 0.0)
        relevance_score = float(result.get("relevance_score", 0.0))
        item["score"] = relevance_score
        item["rerank_score"] = relevance_score
        item["rerank_method"] = "jina_cross_encoder"
        item["metadata"]["rerank_method"] = "jina_cross_encoder"
        item["metadata"]["rerank_model"] = JINA_RERANK_MODEL
        item["metadata"]["original_score"] = original_score
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:top_k]


def rerank_rrf(
    ranked_lists: list[list[dict[str, Any]]],
    top_k: int = DEFAULT_TOP_K,
    k: int = 60,
) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) de gop ket qua tu nhieu ranker.

    Formula:
        RRF(d) = sum(1 / (k + rank_r(d)))

    - rank_r(d) bat dau tu 1.
    - k=60 la smoothing constant pho bien trong paper RRF.
    """
    if top_k <= 0:
        return []

    scores: dict[str, float] = defaultdict(float)
    best_item: dict[str, dict[str, Any]] = {}
    appearances: dict[str, int] = defaultdict(int)

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = _candidate_key(item)
            if not key:
                continue

            scores[key] += 1.0 / (k + rank)
            appearances[key] += 1
            if key not in best_item or float(item.get("score", 0.0) or 0.0) > float(best_item[key].get("score", 0.0) or 0.0):
                best_item[key] = item

    fused: list[dict[str, Any]] = []
    for key, score in scores.items():
        item = _copy_candidate(best_item[key])
        item["score"] = float(score)
        item["rerank_score"] = float(score)
        item["rerank_method"] = "rrf"
        item["metadata"]["rerank_method"] = "rrf"
        item["metadata"]["rrf_k"] = k
        item["metadata"]["rrf_appearances"] = appearances[key]
        fused.append(item)

    fused.sort(key=lambda item: item["score"], reverse=True)
    return fused[:top_k]


def rerank_mmr(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    lambda_param: float = 0.7,
) -> list[dict[str, Any]]:
    """
    Simple MMR-style fallback without embeddings.

    MMR day dung token Jaccard similarity de giam trung lap khi candidates
    khong co vector embedding.
    """
    if top_k <= 0 or not candidates:
        return []

    selected: list[dict[str, Any]] = []
    remaining = [_copy_candidate(candidate) for candidate in candidates]

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = float("-inf")

        for idx, candidate in enumerate(remaining):
            relevance = _fallback_relevance_score(
                query,
                candidate.get("content", ""),
                float(candidate.get("score", 0.0) or 0.0),
            )
            candidate_tokens = set(_tokenize(candidate.get("content", "")))
            max_similarity = 0.0
            for chosen in selected:
                chosen_tokens = set(_tokenize(chosen.get("content", "")))
                union = candidate_tokens | chosen_tokens
                similarity = len(candidate_tokens & chosen_tokens) / len(union) if union else 0.0
                max_similarity = max(max_similarity, similarity)

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_similarity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        chosen = remaining.pop(best_idx)
        chosen["score"] = float(best_score)
        chosen["rerank_score"] = float(best_score)
        chosen["rerank_method"] = "mmr_token_fallback"
        chosen["metadata"]["rerank_method"] = "mmr_token_fallback"
        selected.append(chosen)

    return selected


def rerank(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int = DEFAULT_TOP_K,
    method: str = "cross_encoder",
) -> list[dict[str, Any]]:
    """
    Re-score and re-order candidates based on relevance to query.

    Args:
        query: Cau truy van.
        candidates: Retrieval results, moi item co content/score/metadata.
        top_k: So ket qua tra ve.
        method: "cross_encoder" | "mmr" | "fallback".

    Returns:
        List top_k candidates da re-score va sort descending.
    """
    if top_k <= 0 or not candidates:
        return []

    query = query.strip()
    if not query:
        raise ValueError("query khong duoc rong.")

    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    if method == "mmr":
        return rerank_mmr(query, candidates, top_k)
    if method in {"fallback", "lexical_fallback"}:
        return rerank_fallback(query, candidates, top_k)

    raise ValueError(f"Unknown rerank method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 7 reranking")
    parser.add_argument(
        "query",
        nargs="?",
        default="hinh phat tang tru ma tuy",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--method",
        choices=["cross_encoder", "mmr", "fallback"],
        default="cross_encoder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dummy_candidates = [
        {"content": "Dieu 249: Toi tang tru trai phep chat ma tuy", "score": 0.8, "metadata": {}},
        {"content": "Nghe si bi bat vi su dung ma tuy", "score": 0.7, "metadata": {}},
        {"content": "Tai lieu ve Python programming", "score": 0.6, "metadata": {}},
    ]
    results = rerank(args.query, dummy_candidates, top_k=args.top_k, method=args.method)
    for result in results:
        print(f"[{result['score']:.4f}] {result['rerank_method']} | {result['content']}")
