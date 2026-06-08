"""Start Docker Weaviate, index data, and launch Chainlit."""

from __future__ import annotations

import argparse
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from src.chunking.legal_chunker import chunk_documents, load_markdown_documents
from src.config import CONFIG
from src.retrieval.vectorless_bm25 import build_whoosh_index
from src.retrieval.weaviate_store import index_chunks_to_weaviate


def _run(command: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(command)}")
    return subprocess.run(command, cwd=str(cwd or CONFIG.project_root), check=check)


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _docker_available() -> bool:
    return shutil.which("docker") is not None


def start_weaviate() -> None:
    """Start Weaviate with Docker Compose and wait until HTTP port is open."""
    if not CONFIG.auto_start_docker:
        print("AUTO_START_DOCKER=false, skipping Docker startup.")
        return

    if not _docker_available():
        raise RuntimeError("Docker CLI khong co trong PATH. Hay cai Docker Desktop truoc.")

    if not CONFIG.docker_compose_file.exists():
        raise FileNotFoundError(f"Khong tim thay docker compose: {CONFIG.docker_compose_file}")

    _run(["docker", "compose", "-f", str(CONFIG.docker_compose_file), "up", "-d"])

    deadline = time.time() + CONFIG.weaviate_start_timeout_seconds
    print(
        f"Waiting for Weaviate at {CONFIG.weaviate_host}:"
        f"{CONFIG.weaviate_http_port} ..."
    )
    while time.time() < deadline:
        if _port_open(CONFIG.weaviate_host, CONFIG.weaviate_http_port):
            print("Weaviate is reachable.")
            return
        time.sleep(2)

    raise TimeoutError(
        "Weaviate chua san sang sau "
        f"{CONFIG.weaviate_start_timeout_seconds}s. Kiem tra Docker Desktop/logs."
    )


def build_indexes(skip_weaviate: bool = False, reset: bool = False) -> None:
    """Auto-standardize through load_markdown_documents(), then build indexes."""
    docs = load_markdown_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} group chunks")

    whoosh_path = build_whoosh_index(chunks, reset=reset)
    print(f"Whoosh BM25 index ready: {whoosh_path}")

    if skip_weaviate:
        print("Skipping Weaviate indexing.")
        return

    count = index_chunks_to_weaviate(chunks, reset=reset)
    print(f"Weaviate dense index ready: {count} chunks")


def launch_chainlit() -> None:
    """Launch Chainlit using the current virtual environment."""
    chainlit_bin = CONFIG.project_root / ".venv" / "bin" / "chainlit"
    command = [
        str(chainlit_bin if chainlit_bin.exists() else "chainlit"),
        "run",
        "--host",
        CONFIG.chainlit_host,
        "--port",
        str(CONFIG.chainlit_port),
        "-h",
        "src/ui/chainlit_app.py",
    ]
    _run(command)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run group RAG chatbot end-to-end")
    parser.add_argument("--no-docker", action="store_true", help="Do not start Docker/Weaviate")
    parser.add_argument("--no-index", action="store_true", help="Skip indexing before starting UI")
    parser.add_argument("--skip-weaviate-index", action="store_true", help="Only build local Whoosh index")
    parser.add_argument("--reset-index", action="store_true", help="Reset indexes before writing")
    parser.add_argument("--no-ui", action="store_true", help="Prepare Docker/indexes but do not launch Chainlit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.no_docker:
        start_weaviate()

    should_index = CONFIG.auto_index_on_start and not args.no_index
    if should_index:
        build_indexes(
            skip_weaviate=args.no_docker or args.skip_weaviate_index,
            reset=args.reset_index or CONFIG.reset_index_on_start,
        )
    else:
        print("Skipping indexing.")

    if not args.no_ui:
        launch_chainlit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(130)
