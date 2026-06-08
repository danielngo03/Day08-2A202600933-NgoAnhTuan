"""
Task 5 - Semantic Search Module.

Dense retrieval tren vector store da tao o Task 4.

Input:
    semantic_search(query: str, top_k: int = 10)

Output:
    List[{
        "content": str,
        "score": float,
        "metadata": dict,
    }]
    Sap xep theo cosine similarity giam dan.

Tuong thich voi Task 4:
    - data/indexes/openai_text_embedding_3_small/
    - data/indexes/bge_m3/
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"
ENV_PATH = PROJECT_ROOT / ".env"

OPENAI_INDEX = "openai_text_embedding_3_small"
BGE_INDEX = "bge_m3"
DEFAULT_INDEX_PRIORITY = (OPENAI_INDEX, BGE_INDEX)

INDEX_ALIASES = {
    "openai": OPENAI_INDEX,
    "text-embedding-3-small": OPENAI_INDEX,
    "openai_text_embedding_3_small": OPENAI_INDEX,
    "bge": BGE_INDEX,
    "bge-m3": BGE_INDEX,
    "bge_m3": BGE_INDEX,
    "BAAI/bge-m3": BGE_INDEX,
}


def _resolve_index_name(index_name: str | None = None) -> str:
    """Chon index semantic search."""
    requested = index_name or os.getenv("SEMANTIC_INDEX_MODEL")
    if requested:
        resolved = INDEX_ALIASES.get(requested, requested)
        if not (INDEX_DIR / resolved).exists():
            raise FileNotFoundError(
                f"Index '{resolved}' chua ton tai. Hay chay Task 4 truoc, vi du: "
                f".venv/bin/python src/task4_chunking_indexing.py --models openai"
            )
        return resolved

    for candidate in DEFAULT_INDEX_PRIORITY:
        if (INDEX_DIR / candidate).exists():
            return candidate

    raise FileNotFoundError(
        "Chua co vector index trong data/indexes/. Hay chay Task 4 truoc, vi du: "
        ".venv/bin/python src/task4_chunking_indexing.py --models openai"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@lru_cache(maxsize=4)
def load_vector_store(index_name: str | None = None) -> dict[str, Any]:
    """
    Load local vector store tu data/indexes/{index_name}/.

    Returns:
        {
            "index_name": str,
            "manifest": dict,
            "chunks": list[dict],
            "embeddings": np.ndarray
        }
    """
    resolved = _resolve_index_name(index_name)
    index_path = INDEX_DIR / resolved

    manifest_path = index_path / "manifest.json"
    chunks_path = index_path / "chunks.jsonl"
    embeddings_path = index_path / "embeddings.npy"

    missing = [
        str(path)
        for path in (manifest_path, chunks_path, embeddings_path)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Index '{resolved}' thieu file: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    chunks = _read_jsonl(chunks_path)
    embeddings = np.load(embeddings_path).astype(np.float32)

    if len(chunks) != embeddings.shape[0]:
        raise ValueError(
            f"Index '{resolved}' bi lech: {len(chunks)} chunks vs "
            f"{embeddings.shape[0]} embeddings"
        )

    expected_dim = int(manifest["embedding_dim"])
    if embeddings.shape[1] != expected_dim:
        raise ValueError(
            f"Index '{resolved}' sai dimension: expected {expected_dim}, "
            f"got {embeddings.shape[1]}"
        )

    return {
        "index_name": resolved,
        "manifest": manifest,
        "chunks": chunks,
        "embeddings": embeddings,
    }


@lru_cache(maxsize=1)
def _load_bge_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_query_openai(query: str, model_name: str) -> np.ndarray:
    load_dotenv(ENV_PATH)
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY chua co trong .env.")

    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=model_name, input=query)
    return _normalize(np.asarray(response.data[0].embedding, dtype=np.float32))


def _embed_query_bge(query: str, model_name: str) -> np.ndarray:
    model = _load_bge_model(model_name)
    embedding = model.encode(
        [query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return _normalize(np.asarray(embedding, dtype=np.float32))


def embed_query(query: str, store: dict[str, Any]) -> np.ndarray:
    """Embed query bang dung model da dung de index trong Task 4."""
    model_name = store["manifest"]["embedding_model"]
    index_name = store["index_name"]

    if index_name == OPENAI_INDEX or model_name == "text-embedding-3-small":
        query_embedding = _embed_query_openai(query, model_name)
    elif index_name == BGE_INDEX or model_name == "BAAI/bge-m3":
        query_embedding = _embed_query_bge(query, model_name)
    else:
        raise ValueError(f"Khong biet cach embed query cho index '{index_name}'.")

    expected_dim = int(store["manifest"]["embedding_dim"])
    if query_embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Query embedding sai dimension: expected {expected_dim}, "
            f"got {query_embedding.shape[0]}"
        )
    return query_embedding


def semantic_search(
    query: str,
    top_k: int = 10,
    index_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Tim kiem ngu nghia bang cosine similarity tren dense vector store.

    Args:
        query: Cau truy van
        top_k: So ket qua toi da
        index_name: Optional model/index override. Chap nhan:
            "openai", "openai_text_embedding_3_small", "bge_m3", "bge".

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

    store = load_vector_store(index_name)
    query_embedding = embed_query(query, store)

    embeddings = store["embeddings"]
    scores = embeddings @ query_embedding
    limit = min(top_k, len(scores))
    if limit == 0:
        return []

    candidate_indices = np.argpartition(scores, -limit)[-limit:]
    sorted_indices = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]

    results: list[dict[str, Any]] = []
    for idx in sorted_indices:
        chunk = store["chunks"][int(idx)]
        metadata = {
            **chunk.get("metadata", {}),
            "chunk_id": chunk.get("id"),
            "index_name": store["index_name"],
            "embedding_model": store["manifest"]["embedding_model"],
            "embedding_dim": store["manifest"]["embedding_dim"],
        }
        results.append(
            {
                "content": chunk["content"],
                "score": float(scores[int(idx)]),
                "metadata": metadata,
            }
        )

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 5 semantic search")
    parser.add_argument(
        "query",
        nargs="?",
        default="hình phạt cho tội tàng trữ ma túy",
        help="Cau truy van semantic search",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--index",
        choices=sorted(INDEX_ALIASES),
        default=None,
        help="Index/model can dung: openai hoac bge_m3",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    search_results = semantic_search(args.query, top_k=args.top_k, index_name=args.index)
    for i, result in enumerate(search_results, 1):
        metadata = result["metadata"]
        print(
            f"{i}. [{result['score']:.4f}] "
            f"{metadata.get('doc_type')} | {metadata.get('source')} | "
            f"chunk {metadata.get('chunk_index')}"
        )
        print(result["content"][:240].replace("\n", " "))
        print()
