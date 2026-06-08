"""Vectorless local retrieval with Whoosh BM25 and rank-bm25 fallback."""

from __future__ import annotations

import json
import re
import shutil
import unicodedata
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.chunking.legal_chunker import chunk_documents
from src.config import CONFIG


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    return text.replace("đ", "d")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", normalize_text(text))


def build_whoosh_index(chunks: list[dict[str, Any]] | None = None, reset: bool = False) -> Path:
    """
    Build a local Whoosh BM25 index.

    Whoosh uses BM25F scoring by default for text fields. It is vectorless and
    works without embeddings, making it a strong fallback when dense retrieval
    misses exact legal terms like "Điều 249" or drug quantities.
    """
    try:
        from whoosh import index
        from whoosh.fields import ID, NUMERIC, TEXT, Schema
    except ImportError as exc:
        raise RuntimeError("Can cai whoosh: pip install whoosh") from exc

    selected_chunks = chunks if chunks is not None else chunk_documents()
    index_dir = CONFIG.whoosh_dir
    if reset and index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    schema = Schema(
        chunk_id=ID(stored=True, unique=True),
        content=TEXT(stored=True),
        source=ID(stored=True),
        source_path=ID(stored=True),
        doc_type=ID(stored=True),
        metadata_json=TEXT(stored=True),
        global_chunk_index=NUMERIC(stored=True),
    )
    ix = index.create_in(index_dir, schema) if reset or not index.exists_in(index_dir) else index.open_dir(index_dir)
    writer = ix.writer()
    for chunk in selected_chunks:
        metadata = chunk.get("metadata", {})
        writer.update_document(
            chunk_id=str(chunk["id"]),
            content=chunk["content"],
            source=str(metadata.get("source", "")),
            source_path=str(metadata.get("source_path", "")),
            doc_type=str(metadata.get("doc_type", "")),
            metadata_json=json.dumps(metadata, ensure_ascii=False),
            global_chunk_index=int(metadata.get("global_chunk_index", 0)),
        )
    writer.commit()
    return index_dir


def whoosh_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Search Whoosh BM25 index. Build it lazily if needed."""
    try:
        from whoosh import index
        from whoosh.qparser import MultifieldParser
    except ImportError as exc:
        raise RuntimeError("Can cai whoosh: pip install whoosh") from exc

    if not CONFIG.whoosh_dir.exists() or not index.exists_in(CONFIG.whoosh_dir):
        build_whoosh_index(reset=True)

    ix = index.open_dir(CONFIG.whoosh_dir)
    parser = MultifieldParser(["content", "source", "source_path"], schema=ix.schema)
    parsed_query = parser.parse(query)
    results: list[dict[str, Any]] = []
    with ix.searcher() as searcher:
        hits = searcher.search(parsed_query, limit=top_k)
        for rank, hit in enumerate(hits, start=1):
            metadata = json.loads(hit["metadata_json"])
            metadata.update({"retriever": "whoosh_bm25", "rank": rank, "chunk_id": hit["chunk_id"]})
            results.append(
                {
                    "content": hit["content"],
                    "score": float(hit.score),
                    "metadata": metadata,
                    "source": "whoosh_bm25",
                }
            )
    return results


@lru_cache(maxsize=1)
def _rank_bm25_corpus() -> tuple[list[dict[str, Any]], BM25Okapi]:
    chunks = chunk_documents()
    tokenized = [tokenize(chunk["content"]) for chunk in chunks]
    return chunks, BM25Okapi(tokenized)


def rank_bm25_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Pure Python fallback when Whoosh is not installed."""
    chunks, bm25 = _rank_bm25_corpus()
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(range(len(chunks)), key=lambda i: scores[i], reverse=True)[:top_k]

    results: list[dict[str, Any]] = []
    for rank, index in enumerate(ranked, start=1):
        chunk = chunks[index]
        metadata = {
            **chunk.get("metadata", {}),
            "retriever": "rank_bm25_fallback",
            "rank": rank,
            "chunk_id": chunk["id"],
        }
        results.append(
            {
                "content": chunk["content"],
                "score": float(scores[index]),
                "metadata": metadata,
                "source": "rank_bm25",
            }
        )
    return results


def vectorless_search(query: str, top_k: int = 10, method: str = "bm25") -> list[dict[str, Any]]:
    """Search using BM25 or TF-IDF method."""
    if method == "tfidf":
        return tfidf_search(query, top_k=top_k)
    try:
        return whoosh_search(query, top_k=top_k)
    except Exception:
        return rank_bm25_search(query, top_k=top_k)


@lru_cache(maxsize=1)
def _load_tfidf_corpus() -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    chunks = chunk_documents()
    tokenized_docs = [tokenize(chunk["content"]) for chunk in chunks]
    
    # Compute DF (document frequency) for each unique token
    df = defaultdict(int)
    for doc in tokenized_docs:
        unique_tokens = set(doc)
        for t in unique_tokens:
            df[t] += 1
            
    # Compute IDF for each token: idf(t) = log(1 + N / (1 + df(t)))
    N = len(chunks)
    idf = {}
    for t, count in df.items():
        idf[t] = math.log(1 + N / (1 + count))
        
    # Compute TF-IDF vector for each document
    doc_tfidfs = []
    for doc in tokenized_docs:
        doc_len = len(doc)
        if doc_len == 0:
            doc_tfidfs.append({})
            continue
        token_counts = defaultdict(int)
        for t in doc:
            token_counts[t] += 1
        tfidf_vec = {}
        for t, count in token_counts.items():
            tf = count / doc_len
            tfidf_vec[t] = tf * idf[t]
        doc_tfidfs.append(tfidf_vec)
        
    return chunks, doc_tfidfs, idf


def tfidf_search(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """TF-IDF lexical search implementation in pure Python."""
    chunks, doc_tfidfs, idf = _load_tfidf_corpus()
    
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
        
    query_counts = defaultdict(int)
    for t in query_tokens:
        query_counts[t] += 1
        
    scores = []
    for idx, doc_tfidf in enumerate(doc_tfidfs):
        score = 0.0
        for t, q_count in query_counts.items():
            if t in doc_tfidf:
                # TF in query * TF in doc * IDF^2
                score += doc_tfidf[t] * idf.get(t, 0.0) * (q_count / len(query_tokens))
        if score > 0:
            scores.append((idx, score))
            
    scores.sort(key=lambda x: x[1], reverse=True)
    ranked = scores[:top_k]
    
    results: list[dict[str, Any]] = []
    for rank, (index, score) in enumerate(ranked, start=1):
        chunk = chunks[index]
        metadata = {
            **chunk.get("metadata", {}),
            "retriever": "tfidf_lexical",
            "rank": rank,
            "chunk_id": chunk["id"],
        }
        results.append(
            {
                "content": chunk["content"],
                "score": float(score),
                "metadata": metadata,
                "source": "tfidf_lexical",
            }
        )
    return results

