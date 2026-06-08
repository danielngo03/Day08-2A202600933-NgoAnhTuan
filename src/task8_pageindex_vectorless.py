"""
Task 8 - PageIndex Vectorless RAG.

PageIndex la reasoning-based / vectorless RAG:
    - Upload PDF len PageIndex.
    - PageIndex tao hierarchical tree index thay vi vector database.
    - Query bang retrieval API: submit_query -> get_retrieval.

SDK docs hien tai:
    from pageindex import PageIndexClient
    client = PageIndexClient(api_key="...")
    result = client.submit_document("./file.pdf")
    tree = client.get_tree(doc_id)
    query = client.submit_query(doc_id, "...")
    result = client.get_retrieval(retrieval_id)

Ghi chu:
    PageIndex Python SDK 0.2.x document processing currently accepts PDF.
    Vi vay upload_documents() upload cac file PDF trong data/landing/legal/.
    News markdown van duoc search qua fallback local BM25 neu PageIndex chua
    co result phu hop, de pipeline Task 9 khong bi gay khi service ngoai loi
    hoac document dang processing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
LANDING_LEGAL_DIR = PROJECT_ROOT / "data" / "landing" / "legal"
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
PAGEINDEX_DIR = PROJECT_ROOT / "data" / "pageindex"
REGISTRY_PATH = PAGEINDEX_DIR / "documents.json"
ENV_PATH = PROJECT_ROOT / ".env"

SUPPORTED_UPLOAD_EXTENSIONS = {".pdf"}
DEFAULT_TOP_K = 5
PAGEINDEX_QUERY_TIMEOUT_SECONDS = int(os.getenv("PAGEINDEX_QUERY_TIMEOUT_SECONDS", "6"))
PAGEINDEX_MAX_REMOTE_DOCS = int(os.getenv("PAGEINDEX_MAX_REMOTE_DOCS", "1"))

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_api_key() -> str:
    load_dotenv(ENV_PATH)
    api_key = os.getenv("PAGEINDEX_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("PAGEINDEX_API_KEY chua co trong .env.")
    return api_key


def _client():
    from pageindex import PageIndexClient

    return PageIndexClient(api_key=_load_api_key())


def _load_registry() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"documents": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def _save_registry(registry: dict[str, Any]) -> None:
    PAGEINDEX_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _document_files() -> list[Path]:
    if not LANDING_LEGAL_DIR.exists():
        return []
    return sorted(
        path
        for path in LANDING_LEGAL_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_UPLOAD_EXTENSIONS
    )


def _registry_by_path(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["source_path"]: item for item in registry.get("documents", [])}


def _get_status(client, doc_id: str) -> dict[str, Any]:
    try:
        tree = client.get_tree(doc_id)
        metadata = client.get_document(doc_id)
        return {
            "doc_id": doc_id,
            "status": tree.get("status") or metadata.get("status"),
            "retrieval_ready": bool(tree.get("retrieval_ready")),
            "metadata": metadata,
        }
    except Exception as exc:
        return {
            "doc_id": doc_id,
            "status": "unknown",
            "retrieval_ready": False,
            "error": str(exc),
        }


def check_api_key() -> bool:
    """Lightweight auth check bang list_documents(limit=1)."""
    client = _client()
    client.list_documents(limit=1)
    return True


def upload_documents(wait: bool = False, timeout_seconds: int = 300) -> list[dict[str, Any]]:
    """
    Upload PDF legal documents len PageIndex va luu doc_id vao registry.

    Returns:
        List registry entries:
        {
            "doc_id": str,
            "filename": str,
            "source_path": str,
            "doc_type": "legal",
            "status": str,
            "retrieval_ready": bool
        }
    """
    client = _client()
    registry = _load_registry()
    known = _registry_by_path(registry)

    uploaded: list[dict[str, Any]] = []
    for file_path in _document_files():
        relative_path = str(file_path.relative_to(PROJECT_ROOT))
        existing = known.get(relative_path)
        if existing:
            if existing.get("doc_id"):
                status = _get_status(client, existing["doc_id"])
                existing.update(
                    {
                        "status": status.get("status"),
                        "retrieval_ready": status.get("retrieval_ready", False),
                        "last_checked_error": status.get("error"),
                    }
                )
            else:
                existing["retrieval_ready"] = False
            uploaded.append(existing)
            print(f"  ✓ Already uploaded: {file_path.name} -> {existing['doc_id']}")
            continue

        print(f"  Uploading: {file_path.name}")
        try:
            result = client.submit_document(str(file_path))
            doc_id = result["doc_id"]
            status = "submitted"
            error = None
            print(f"  ✓ Uploaded: {file_path.name} -> {doc_id}")
        except Exception as exc:
            doc_id = ""
            status = "upload_failed"
            error = str(exc)
            print(f"  ! Upload failed: {file_path.name} ({error[:120]})")

        entry = {
            "doc_id": doc_id,
            "filename": file_path.name,
            "source_path": relative_path,
            "doc_type": "legal",
            "status": status,
            "retrieval_ready": False,
            "last_checked_error": error,
        }
        registry.setdefault("documents", []).append(entry)
        uploaded.append(entry)

    _save_registry(registry)

    if wait and uploaded:
        wait_for_documents(timeout_seconds=timeout_seconds)
        registry = _load_registry()
        uploaded_paths = {item["source_path"] for item in uploaded}
        uploaded = [
            item
            for item in registry.get("documents", [])
            if item.get("source_path") in uploaded_paths
        ]

    return uploaded


def wait_for_documents(timeout_seconds: int = 300, poll_interval: int = 10) -> list[dict[str, Any]]:
    """Poll PageIndex den khi documents retrieval_ready hoac timeout."""
    client = _client()
    registry = _load_registry()
    deadline = time.time() + timeout_seconds

    while True:
        all_ready = True
        for entry in registry.get("documents", []):
            if not entry.get("doc_id"):
                entry["retrieval_ready"] = False
                all_ready = False
                continue

            status = _get_status(client, entry["doc_id"])
            entry["status"] = status.get("status")
            entry["retrieval_ready"] = status.get("retrieval_ready", False)
            entry["last_checked_error"] = status.get("error")
            if not entry["retrieval_ready"]:
                all_ready = False

        _save_registry(registry)
        if all_ready or time.time() >= deadline:
            return registry.get("documents", [])

        time.sleep(poll_interval)


def _ready_doc_ids(limit: int | None = PAGEINDEX_MAX_REMOTE_DOCS) -> list[str]:
    registry = _load_registry()
    doc_ids = [
        item["doc_id"]
        for item in registry.get("documents", [])
        if item.get("doc_id") and item.get("retrieval_ready")
    ]
    if limit is not None and limit > 0:
        return doc_ids[:limit]
    return doc_ids


def _refresh_registry_status() -> None:
    registry = _load_registry()
    if not registry.get("documents"):
        return

    client = _client()
    for entry in registry.get("documents", []):
        if not entry.get("doc_id"):
            entry["retrieval_ready"] = False
            continue

        status = _get_status(client, entry["doc_id"])
        entry["status"] = status.get("status")
        entry["retrieval_ready"] = status.get("retrieval_ready", False)
        entry["last_checked_error"] = status.get("error")
    _save_registry(registry)


def _iter_texts(value: Any) -> list[dict[str, Any]]:
    """
    Flatten PageIndex retrieval response thanh list text snippets.

    SDK response schema co the thay doi theo version; ham nay chap nhan cac
    key pho bien nhu text/content/summary/result/results/nodes.
    """
    snippets: list[dict[str, Any]] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            snippets.append({"content": text, "metadata": {}})
        return snippets

    if isinstance(value, list):
        for item in value:
            snippets.extend(_iter_texts(item))
        return snippets

    if isinstance(value, dict):
        text = (
            value.get("text")
            or value.get("content")
            or value.get("summary")
            or value.get("answer")
            or value.get("message")
        )
        if isinstance(text, str) and text.strip():
            metadata = {
                key: value.get(key)
                for key in (
                    "doc_id",
                    "node_id",
                    "title",
                    "page_index",
                    "page",
                    "score",
                    "filename",
                    "source",
                )
                if value.get(key) is not None
            }
            snippets.append({"content": text.strip(), "metadata": metadata})

        for key in ("result", "results", "nodes", "retrieval", "blocks", "items", "data"):
            if key in value:
                snippets.extend(_iter_texts(value[key]))

    return snippets


def _query_single_doc(
    client,
    doc_id: str,
    query: str,
    timeout_seconds: int = PAGEINDEX_QUERY_TIMEOUT_SECONDS,
) -> list[dict[str, Any]]:
    submitted = client.submit_query(doc_id=doc_id, query=query, thinking=True)
    retrieval_id = submitted["retrieval_id"]
    deadline = time.time() + timeout_seconds

    while True:
        result = client.get_retrieval(retrieval_id)
        status = result.get("status")
        if status in {"completed", "done", "success"} or result.get("result") or result.get("results"):
            snippets = _iter_texts(result)
            for snippet in snippets:
                snippet["metadata"].setdefault("doc_id", doc_id)
                snippet["metadata"]["retrieval_id"] = retrieval_id
                snippet["metadata"]["retrieval_status"] = status
            return snippets

        if status in {"failed", "error"}:
            return []

        if time.time() >= deadline:
            return []

        time.sleep(3)


def _local_fallback_search(query: str, top_k: int) -> list[dict[str, Any]]:
    """
    Fallback local BM25 tren Task 6. Van danh dau source='pageindex' de Task 9
    co the xem day la fallback path, nhung metadata noi ro local_fallback.
    """
    from src.task6_lexical_search import lexical_search

    results = lexical_search(query, top_k=top_k)
    fallback: list[dict[str, Any]] = []
    for result in results:
        fallback.append(
            {
                "content": result["content"],
                "score": float(result["score"]),
                "metadata": {
                    **result.get("metadata", {}),
                    "pageindex_mode": "local_bm25_fallback",
                },
                "source": "pageindex",
            }
        )
    return fallback


def pageindex_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """
    Vectorless retrieval using PageIndex.
    Fallback khi hybrid search khong tra ve ket qua phu hop.

    Returns:
        List of {"content": str, "score": float, "metadata": dict, "source": "pageindex"}
    """
    query = query.strip()
    if not query:
        raise ValueError("query khong duoc rong.")
    if top_k <= 0:
        return []

    try:
        _refresh_registry_status()
        doc_ids = _ready_doc_ids()
        if not doc_ids:
            return _local_fallback_search(query, top_k)

        client = _client()
        pageindex_results: list[dict[str, Any]] = []
        for doc_id in doc_ids:
            snippets = _query_single_doc(client, doc_id, query)
            for snippet in snippets:
                score = float(snippet.get("metadata", {}).get("score") or 1.0)
                pageindex_results.append(
                    {
                        "content": snippet["content"],
                        "score": score,
                        "metadata": {
                            **snippet.get("metadata", {}),
                            "pageindex_mode": "remote_retrieval",
                        },
                        "source": "pageindex",
                    }
                )

        if not pageindex_results:
            return _local_fallback_search(query, top_k)

        pageindex_results.sort(key=lambda item: item["score"], reverse=True)
        return pageindex_results[:top_k]
    except Exception as exc:
        fallback = _local_fallback_search(query, top_k)
        for result in fallback:
            result["metadata"]["pageindex_error"] = str(exc)[:300]
        return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 8 PageIndex vectorless RAG")
    parser.add_argument(
        "query",
        nargs="?",
        default="hình phạt sử dụng ma túy",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--upload", action="store_true", help="Upload legal PDFs to PageIndex first")
    parser.add_argument("--wait", action="store_true", help="Wait for PageIndex processing after upload")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Checking PageIndex API key...")
    check_api_key()
    print("✓ PageIndex API key works")

    if args.upload:
        print("\nUploading documents...")
        uploaded_docs = upload_documents(wait=args.wait)
        print(f"✓ Registry now has {len(uploaded_docs)} uploaded/known documents")

    print("\nTest query:")
    results = pageindex_search(args.query, top_k=args.top_k)
    for result in results:
        mode = result.get("metadata", {}).get("pageindex_mode")
        print(f"[{result['score']:.3f}] [{mode}] {result['content'][:140]}...")
