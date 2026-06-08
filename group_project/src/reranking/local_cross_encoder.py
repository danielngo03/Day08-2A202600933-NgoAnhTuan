"""Local cross-encoder reranker for Vietnamese/multilingual retrieval."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Any

import numpy as np

from src.config import CONFIG


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", _normalize_text(text))


@lru_cache(maxsize=1)
def _load_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CONFIG.reranker_model)


def _fallback_score(query: str, content: str, original_score: float) -> float:
    query_tokens = set(_tokens(query))
    content_tokens = _tokens(content)
    if not query_tokens or not content_tokens:
        return float(original_score)

    content_set = set(content_tokens)
    overlap = len(query_tokens & content_set) / len(query_tokens)
    density = sum(1 for token in content_tokens if token in query_tokens) / max(len(content_tokens), 1)
    return 0.78 * overlap + 0.12 * density + 0.10 * float(original_score)


def rerank_local_cross_encoder(
    query: str,
    candidates: list[dict[str, Any]],
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Re-score candidates with BAAI/bge-reranker-v2-m3.

    This is a local cross-encoder: the model sees each (query, document) pair,
    so it captures Vietnamese semantic relevance better than pure cosine/BM25.
    If the model is not downloaded yet or the machine cannot run it, fallback
    lexical scoring keeps the chatbot usable for demo.
    """
    if not candidates:
        return []

    selected_top_k = top_k or CONFIG.rerank_top_k
    try:
        model = _load_cross_encoder()
        pairs = [(query, candidate.get("content", "")) for candidate in candidates]
        raw_scores = model.predict(
            pairs,
            batch_size=CONFIG.reranker_batch_size,
            show_progress_bar=False,
        )
        scores = np.asarray(raw_scores, dtype=np.float32)
        method = CONFIG.reranker_model
    except Exception as exc:
        scores = np.asarray(
            [
                _fallback_score(query, candidate.get("content", ""), float(candidate.get("score", 0.0) or 0.0))
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        method = "lexical_overlap_fallback"
        fallback_error = str(exc)[:240]
    else:
        fallback_error = ""

    reranked: list[dict[str, Any]] = []
    for candidate, score in zip(candidates, scores, strict=True):
        item = {
            **candidate,
            "score": float(score),
            "metadata": {
                **candidate.get("metadata", {}),
                "reranker": method,
                "pre_rerank_score": float(candidate.get("score", 0.0) or 0.0),
            },
        }
        if fallback_error:
            item["metadata"]["reranker_error"] = fallback_error
        reranked.append(item)

    reranked.sort(key=lambda item: item["score"], reverse=True)
    return reranked[:selected_top_k]

