"""
Configuration for the group RAG chatbot.

The app is designed to run from group_project with:
    .venv/bin/python run.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SRC_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_ROOT.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)


@dataclass(frozen=True)
class RagConfig:
    src_root: Path = SRC_ROOT
    project_root: Path = PROJECT_ROOT
    env_path: Path = ENV_PATH
    docker_compose_file: Path = PROJECT_ROOT / "docker-compose.yml"
    landing_dir: Path = SRC_ROOT / "data" / "landing"
    standardized_dir: Path = SRC_ROOT / "data" / "standardized"
    group_index_dir: Path = SRC_ROOT / "indexes"
    auto_standardize: bool = os.getenv("AUTO_STANDARDIZE", "true").lower() in {"1", "true", "yes", "on"}

    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    openai_embedding_dim: int = int(os.getenv("OPENAI_EMBEDDING_DIM", "1536"))
    openai_chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

    weaviate_host: str = os.getenv("WEAVIATE_LOCAL_HOST", "localhost")
    weaviate_http_port: int = int(os.getenv("WEAVIATE_LOCAL_HTTP_PORT", "8080"))
    weaviate_grpc_port: int = int(os.getenv("WEAVIATE_LOCAL_GRPC_PORT", "50051"))
    weaviate_collection: str = os.getenv("WEAVIATE_COLLECTION", "DrugLawChunk")
    weaviate_start_timeout_seconds: int = int(os.getenv("WEAVIATE_START_TIMEOUT_SECONDS", "90"))
    auto_start_docker: bool = os.getenv("AUTO_START_DOCKER", "true").lower() in {"1", "true", "yes", "on"}
    auto_index_on_start: bool = os.getenv("AUTO_INDEX_ON_START", "true").lower() in {"1", "true", "yes", "on"}
    reset_index_on_start: bool = os.getenv("RESET_INDEX_ON_START", "false").lower() in {"1", "true", "yes", "on"}

    chainlit_host: str = os.getenv("CHAINLIT_HOST", "127.0.0.1")
    chainlit_port: int = int(os.getenv("CHAINLIT_PORT", "8000"))

    reranker_model: str = os.getenv("LOCAL_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    reranker_batch_size: int = 8

    legal_chunk_size: int = 1800
    legal_chunk_overlap: int = 220
    news_chunk_size: int = 1200
    news_chunk_overlap: int = 160

    retrieval_top_k: int = 12
    rerank_top_k: int = 6
    generation_top_k: int = 5
    score_threshold: float = 0.25

    temperature: float = 0.2
    top_p: float = 0.85

    memory_recent_turns: int = 4
    memory_summary_trigger_turns: int = 6
    max_context_chars: int = 11000

    whoosh_dir: Path = SRC_ROOT / "indexes" / "whoosh_bm25"
    local_dense_index_dir: Path = Path(os.getenv("LOCAL_DENSE_INDEX_DIR", str(SRC_ROOT / "indexes" / "openai_text_embedding_3_small")))


CONFIG = RagConfig()
