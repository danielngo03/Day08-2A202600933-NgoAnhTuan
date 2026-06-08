"""
Task 4 - Chunking & Indexing.

Yeu cau bai:
    1. Doc toan bo Markdown trong data/standardized/
    2. Chon chunking strategy, chunk_size, overlap va giai thich trong code
    3. Chon embedding model va ghi ro dimension
    4. Index thanh cong toan bo documents vao vector store

Thiet ke:
    - Chunking: RecursiveCharacterTextSplitter
      Ly do: corpus gom PDF phap luat da convert va news markdown; heading khong
      dong deu, nen recursive splitter an toan hon MarkdownHeaderTextSplitter.
    - chunk_size=1200 ky tu: du dai de giu ngu canh dieu/khoan va doan bao.
    - chunk_overlap=180 ky tu: giu cau/ngu canh bi cat qua bien chunk.
    - Embeddings:
        * BAAI/bge-m3: 1024 dim, multilingual, phu hop tieng Viet.
        * OpenAI text-embedding-3-small: 1536 dim, nhe va tot cho RAG.
    - Vector store:
        * Local persistent vector store trong data/indexes/.
        * Moi model co folder rieng gom manifest.json, chunks.jsonl,
          embeddings.npy. Cach nay chay duoc ngay khong can Weaviate server,
          va van co vector dense that de Task 5 query cosine similarity.

Cai dat:
    pip install langchain-text-splitters sentence-transformers weaviate-client openai python-dotenv

Neu muon tao OpenAI index:
    1. Them OPENAI_API_KEY vao .env
    2. Chay: python src/task4_chunking_indexing.py --models openai
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
INDEX_DIR = PROJECT_ROOT / "data" / "indexes"


# =============================================================================
# CONFIGURATION
# =============================================================================

# Chunking strategy: RecursiveCharacterTextSplitter.
# Ly do: markdown sau Task 3 den tu PDF phap luat va JSON news, cau truc heading
# khong dong nhat. Recursive splitter uu tien tach theo paragraph/newline truoc,
# sau do moi tach nho hon, nen it pha vo cau hon cach cat cung.
CHUNKING_METHOD = "recursive"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

# Embedding models. Chay ca hai de so sanh retrieval:
# - BAAI/bge-m3: 1024 dimensions, multilingual, manh voi tieng Viet.
# - text-embedding-3-small: 1536 dimensions theo OpenAI docs/cookbook,
#   ho tro tham so dimensions nhung o day giu native dim de toi da thong tin.
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_DIM = 1024
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_DIM = 1536

# Local vector store. Weaviate duoc khuyen cao trong README, nhung can service
# rieng. Local .npy + .jsonl dam bao Task 4 index thanh cong tren workspace nay.
VECTOR_STORE = "local_numpy_jsonl"
DEFAULT_MODELS = ("bge_m3", "openai")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_id(*parts: str) -> str:
    raw = "::".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _extract_title(content: str, fallback: str) -> str:
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line.removeprefix("# ").strip() or fallback
    return fallback


def _extract_markdown_field(content: str, label: str) -> str | None:
    pattern = rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$"
    for line in content.splitlines():
        match = re.match(pattern, line.strip())
        if match:
            return match.group(1)
    return None


def load_documents() -> list[dict[str, Any]]:
    """
    Doc toan bo markdown files tu data/standardized/.

    Returns:
        List of {"content": str, "metadata": dict}
    """
    documents: list[dict[str, Any]] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        title = _extract_title(content, md_file.stem)

        metadata: dict[str, Any] = {
            "doc_id": _stable_id(str(relative_path)),
            "source": md_file.name,
            "source_path": str(relative_path),
            "doc_type": doc_type,
            "title": title,
            "char_count": len(content),
        }

        source_url = _extract_markdown_field(content, "Source")
        if source_url:
            metadata["source_url"] = source_url

        documents.append({"content": content, "metadata": metadata})

    return documents


def chunk_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Chunk documents bang RecursiveCharacterTextSplitter.

    Returns:
        List of {"id": str, "content": str, "metadata": dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )

    chunks: list[dict[str, Any]] = []

    for doc in documents:
        splits = splitter.split_text(doc["content"])
        total_chunks = len(splits)

        for chunk_index, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            metadata = {
                **doc["metadata"],
                "chunk_index": chunk_index,
                "chunk_count": total_chunks,
                "chunking_method": CHUNKING_METHOD,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
            }
            chunk_id = _stable_id(
                metadata["source_path"],
                str(chunk_index),
                chunk_text[:120],
            )
            chunks.append(
                {
                    "id": chunk_id,
                    "content": chunk_text,
                    "metadata": metadata,
                }
            )

    return chunks


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_texts(chunks: list[dict[str, Any]]) -> list[str]:
    return [chunk["content"] for chunk in chunks]


def embed_bge_m3(chunks: list[dict[str, Any]], batch_size: int = 8) -> np.ndarray:
    """Embed chunks bang BAAI/bge-m3 local model."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BGE_MODEL_NAME)
    embeddings = model.encode(
        _read_texts(chunks),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    if embeddings_array.shape[1] != BGE_DIM:
        raise ValueError(f"Expected BGE dim {BGE_DIM}, got {embeddings_array.shape[1]}")
    return embeddings_array


def embed_openai(chunks: list[dict[str, Any]], batch_size: int = 64) -> np.ndarray:
    """Embed chunks bang OpenAI text-embedding-3-small."""
    load_dotenv(PROJECT_ROOT / ".env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY chua co trong .env. Them key roi chay lai "
            "`python src/task4_chunking_indexing.py --models openai`."
        )

    from openai import OpenAI

    client = OpenAI()
    texts = _read_texts(chunks)
    all_embeddings: list[list[float]] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        response = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        all_embeddings.extend(item.embedding for item in ordered)
        print(f"  OpenAI embedded {min(start + batch_size, len(texts))}/{len(texts)} chunks")

    embeddings_array = np.asarray(all_embeddings, dtype=np.float32)
    if embeddings_array.shape[1] != OPENAI_EMBEDDING_DIM:
        raise ValueError(
            f"Expected OpenAI dim {OPENAI_EMBEDDING_DIM}, got {embeddings_array.shape[1]}"
        )

    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
    return embeddings_array / np.maximum(norms, 1e-12)


def index_to_local_store(
    chunks: list[dict[str, Any]],
    embeddings: np.ndarray,
    model_key: str,
    model_name: str,
    embedding_dim: int,
) -> Path:
    """
    Luu chunks + embeddings vao local vector store.

    Folder output:
        data/indexes/{model_key}/manifest.json
        data/indexes/{model_key}/chunks.jsonl
        data/indexes/{model_key}/embeddings.npy
    """
    if len(chunks) != len(embeddings):
        raise ValueError("So chunks va embeddings khong khop.")

    model_dir = INDEX_DIR / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    chunk_rows = [
        {
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
        }
        for chunk in chunks
    ]

    _write_jsonl(model_dir / "chunks.jsonl", chunk_rows)
    np.save(model_dir / "embeddings.npy", embeddings.astype(np.float32))

    manifest = {
        "created_at": _now_iso(),
        "vector_store": VECTOR_STORE,
        "model_key": model_key,
        "embedding_model": model_name,
        "embedding_dim": embedding_dim,
        "embedding_normalized": True,
        "chunking_method": CHUNKING_METHOD,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "document_count": len({chunk["metadata"]["doc_id"] for chunk in chunks}),
        "chunk_count": len(chunks),
        "files": {
            "chunks": "chunks.jsonl",
            "embeddings": "embeddings.npy",
        },
    }
    (model_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return model_dir


def build_indexes(
    chunks: list[dict[str, Any]],
    models: tuple[str, ...] = DEFAULT_MODELS,
    skip_missing_openai_key: bool = True,
) -> list[Path]:
    """Embed va index chunks cho cac model duoc yeu cau."""
    outputs: list[Path] = []

    for model_key in models:
        if model_key == "bge_m3":
            print(f"\n--- Embedding + indexing: {BGE_MODEL_NAME} ({BGE_DIM} dim) ---")
            embeddings = embed_bge_m3(chunks)
            outputs.append(
                index_to_local_store(
                    chunks,
                    embeddings,
                    model_key="bge_m3",
                    model_name=BGE_MODEL_NAME,
                    embedding_dim=BGE_DIM,
                )
            )
        elif model_key == "openai":
            print(
                f"\n--- Embedding + indexing: {OPENAI_EMBEDDING_MODEL} "
                f"({OPENAI_EMBEDDING_DIM} dim) ---"
            )
            try:
                embeddings = embed_openai(chunks)
            except RuntimeError as exc:
                if skip_missing_openai_key:
                    print(f"  ! Skipping OpenAI index: {exc}")
                    continue
                raise
            outputs.append(
                index_to_local_store(
                    chunks,
                    embeddings,
                    model_key="openai_text_embedding_3_small",
                    model_name=OPENAI_EMBEDDING_MODEL,
                    embedding_dim=OPENAI_EMBEDDING_DIM,
                )
            )
        else:
            raise ValueError(f"Unknown model key: {model_key}")

    return outputs


def run_pipeline(
    models: tuple[str, ...] = DEFAULT_MODELS,
    skip_missing_openai_key: bool = True,
) -> list[Path]:
    """Chay pipeline: load -> chunk -> embed -> index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD}")
    print(f"  chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}")
    print(f"  Embedding models: {', '.join(models)}")
    print(f"  Vector store: {VECTOR_STORE}")
    print("=" * 60)

    documents = load_documents()
    print(f"\n✓ Loaded {len(documents)} markdown documents from {STANDARDIZED_DIR}")

    chunks = chunk_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")

    outputs = build_indexes(
        chunks,
        models=models,
        skip_missing_openai_key=skip_missing_openai_key,
    )

    for output in outputs:
        print(f"✓ Indexed: {output}")

    if not outputs:
        raise RuntimeError("No index was created.")

    print("\n✓ Done! Index output:", INDEX_DIR)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 4 chunking and indexing")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["bge_m3", "openai"],
        default=list(DEFAULT_MODELS),
        help="Embedding indexes to build. Default: bge_m3 openai",
    )
    parser.add_argument(
        "--require-openai-key",
        action="store_true",
        help="Fail instead of skipping OpenAI index when OPENAI_API_KEY is missing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        models=tuple(args.models),
        skip_missing_openai_key=not args.require_openai_key,
    )
