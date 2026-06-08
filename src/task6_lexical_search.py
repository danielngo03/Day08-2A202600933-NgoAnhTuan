"""
Task 6 - Lexical Search Module (BM25).

BM25 la lexical retriever: no khong can embedding/API, ma cham diem dua tren
tu khoa xuat hien trong chunk.

Co che demo:
    - TF (term frequency): tu query xuat hien nhieu trong chunk -> diem cao.
    - IDF (inverse document frequency): tu hiem trong corpus -> quan trong hon.
    - Length normalization: chunk dai khong duoc loi qua muc.
    - BM25Okapi mac dinh trong rank-bm25 dung k1=1.5 va b=0.75.

Corpus:
    - Uu tien dung chunks.jsonl trong data/indexes/ do Task 4 da chunk san.
    - Neu chua co index, fallback doc data/standardized/**/*.md va chunk bang
      Task 4 de module van chay duoc.

Output:
    lexical_search(query, top_k) -> List[{"content": str, "score": float, "metadata": dict}]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
DEFAULT_CHUNKS_PATH = INDEX_DIR / "openai_text_embedding_3_small" / "chunks.jsonl"


def normalize_text(text: str) -> str:
    """
    Chuan hoa text cho lexical search tieng Viet.

    - Lowercase.
    - Chuyen "ma tuý" va "ma túy" ve cung dang khong dau.
    - Giu lai chu, so, underscore de match dieu luat nhu "249".
    """
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d")
    return text


def tokenize(text: str) -> list[str]:
    """
    Tokenize don gian nhung on dinh cho tieng Viet.

    rank-bm25 nhan list token; regex nay tach theo cum chu/so sau khi bo dau.
    """
    normalized = normalize_text(text)
    return re.findall(r"[a-z0-9_]+", normalized)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_chunks_from_index(chunks_path: Path) -> list[dict[str, Any]]:
    rows = _read_jsonl(chunks_path)
    corpus: list[dict[str, Any]] = []

    for row in rows:
        content = (row.get("content") or "").strip()
        if not content:
            continue
        corpus.append(
            {
                "content": content,
                "metadata": {
                    **row.get("metadata", {}),
                    "chunk_id": row.get("id"),
                    "lexical_corpus": str(chunks_path.relative_to(PROJECT_ROOT)),
                },
            }
        )

    return corpus


def _load_chunks_from_standardized() -> list[dict[str, Any]]:
    from src.task4_chunking_indexing import chunk_documents, load_documents

    chunks = chunk_documents(load_documents())
    return [
        {
            "content": chunk["content"],
            "metadata": {
                **chunk["metadata"],
                "chunk_id": chunk["id"],
                "lexical_corpus": "data/standardized",
            },
        }
        for chunk in chunks
    ]


def load_corpus(chunks_path: Path | None = None) -> list[dict[str, Any]]:
    """
    Load corpus cho BM25.

    Args:
        chunks_path: Optional path toi chunks.jsonl. Neu None, dung env
            LEXICAL_CHUNKS_PATH hoac default Task 4 OpenAI chunks.
    """
    env_path = os.getenv("LEXICAL_CHUNKS_PATH")
    selected_path = chunks_path or (Path(env_path) if env_path else DEFAULT_CHUNKS_PATH)
    if not selected_path.is_absolute():
        selected_path = PROJECT_ROOT / selected_path

    if selected_path.exists():
        return _load_chunks_from_index(selected_path)

    return _load_chunks_from_standardized()


def build_bm25_index(corpus: list[dict[str, Any]]) -> BM25Okapi:
    """
    Xay dung BM25 index tu corpus.

    Args:
        corpus: List of {"content": str, "metadata": dict}
    """
    if not corpus:
        raise ValueError("Corpus rong, khong the build BM25 index.")

    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=2)
def _get_cached_index(chunks_path_key: str = "") -> tuple[list[dict[str, Any]], BM25Okapi]:
    chunks_path = Path(chunks_path_key) if chunks_path_key else None
    corpus = load_corpus(chunks_path)
    bm25 = build_bm25_index(corpus)
    return corpus, bm25


def lexical_search(
    query: str,
    top_k: int = 10,
    chunks_path: Path | None = None,
) -> list[dict[str, Any]]:
    """
    Tim kiem tu khoa bang BM25.

    Args:
        query: Cau truy van
        top_k: So luong ket qua toi da
        chunks_path: Optional chunks.jsonl de override corpus.

    Returns:
        List of {
            "content": str,
            "score": float,
            "metadata": dict
        }
        Sorted by score descending.
    """
    query = query.strip()
    if not query:
        raise ValueError("query khong duoc rong.")
    if top_k <= 0:
        return []

    chunks_path_key = str(chunks_path) if chunks_path else ""
    corpus, bm25 = _get_cached_index(chunks_path_key)

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
    positive_indices = np.flatnonzero(scores > 0)
    if len(positive_indices) == 0:
        return []

    limit = min(top_k, len(positive_indices))
    candidate_indices = positive_indices[np.argpartition(scores[positive_indices], -limit)[-limit:]]
    sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    results: list[dict[str, Any]] = []
    for idx in sorted_indices:
        doc = corpus[int(idx)]
        metadata = {
            **doc.get("metadata", {}),
            "retriever": "bm25",
            "query_tokens": query_tokens,
        }
        results.append(
            {
                "content": doc["content"],
                "score": float(scores[int(idx)]),
                "metadata": metadata,
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 6 lexical BM25 search")
    parser.add_argument(
        "query",
        nargs="?",
        default="Điều 249 tàng trữ trái phép chất ma túy",
        help="Cau truy van BM25",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunks-path", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search_results = lexical_search(
        args.query,
        top_k=args.top_k,
        chunks_path=args.chunks_path,
    )
    for i, result in enumerate(search_results, 1):
        metadata = result["metadata"]
        print(
            f"{i}. [{result['score']:.3f}] "
            f"{metadata.get('doc_type')} | {metadata.get('source')} | "
            f"chunk {metadata.get('chunk_index')}"
        )
        print(result["content"][:240].replace("\n", " "))
        print()
