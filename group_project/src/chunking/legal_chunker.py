"""
Domain-specific chunking for the group RAG chatbot.

Legal texts are not chunked by raw character count only. We first promote legal
structure markers (Chuong, Muc, Dieu, Khoan) into markdown headers, then use
MarkdownHeaderTextSplitter so chunks retain their legal hierarchy. A secondary
RecursiveCharacterTextSplitter is used only when a section is still too long.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from src.config import CONFIG
from src.ingestion.standardize import ensure_standardized


def _stable_id(*parts: str) -> str:
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()[:16]


def _extract_markdown_field(content: str, label: str) -> str | None:
    pattern = rf"^\*\*{re.escape(label)}:\*\*\s*(.+?)\s*$"
    for line in content.splitlines():
        match = re.match(pattern, line.strip(), flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _first_heading(content: str, fallback: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip() or fallback
    return fallback


def load_markdown_documents() -> list[dict[str, Any]]:
    """Load all standardized markdown files from data/standardized."""
    if CONFIG.auto_standardize:
        ensure_standardized(force=False)

    documents: list[dict[str, Any]] = []
    for path in sorted(CONFIG.standardized_dir.rglob("*.md")):
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = path.relative_to(CONFIG.standardized_dir)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        metadata: dict[str, Any] = {
            "source": path.name,
            "source_path": str(relative_path),
            "doc_type": doc_type,
            "title": _first_heading(content, path.stem),
            "char_count": len(content),
        }

        source_url = _extract_markdown_field(content, "Source")
        if source_url:
            metadata["source_url"] = source_url
        source_domain = _extract_markdown_field(content, "Source domain")
        if source_domain:
            metadata["source_domain"] = source_domain
        crawled = _extract_markdown_field(content, "Crawled")
        if crawled:
            metadata["date_crawled"] = crawled

        documents.append({"content": content, "metadata": metadata})

    return documents


def _promote_legal_headings(text: str) -> str:
    """
    Convert legal structure markers into markdown headers.

    This helps MarkdownHeaderTextSplitter preserve hierarchy:
        CHUONG I -> # CHUONG I
        Muc 1    -> ## Muc 1
        Dieu 249 -> ### Dieu 249
        1. ...   -> #### Khoan 1
    """
    output: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        normalized = re.sub(r"\s+", " ", line)

        if re.match(r"^(CHƯƠNG|CHUONG)\s+[IVXLCDM0-9]+", normalized, flags=re.IGNORECASE):
            output.append(f"# {line}")
        elif re.match(r"^(MỤC|MUC)\s+[0-9IVXLCDM]+", normalized, flags=re.IGNORECASE):
            output.append(f"## {line}")
        elif re.match(r"^Điều\s+\d+[a-zA-Z]?[.:]?", normalized, flags=re.IGNORECASE):
            output.append(f"### {line}")
        elif re.match(r"^\d+\.\s+\S+", normalized) and len(normalized) < 260:
            output.append(f"#### Khoản {line}")
        else:
            output.append(raw_line)

    return "\n".join(output)


def _legal_splitters() -> tuple[MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter]:
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "chapter"),
            ("##", "section"),
            ("###", "article"),
            ("####", "clause"),
        ],
        strip_headers=False,
    )
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.legal_chunk_size,
        chunk_overlap=CONFIG.legal_chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    return header_splitter, recursive_splitter


def _news_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CONFIG.news_chunk_size,
        chunk_overlap=CONFIG.news_chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )


def chunk_documents(documents: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """
    Chunk legal docs with MarkdownHeaderTextSplitter and news with recursive split.

    Returns:
        List of {"id", "content", "metadata"}.
    """
    docs = documents if documents is not None else load_markdown_documents()
    legal_header_splitter, legal_recursive_splitter = _legal_splitters()
    news_recursive_splitter = _news_splitter()

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        metadata = doc["metadata"]
        content = doc["content"]
        doc_type = metadata.get("doc_type", "unknown")
        source_path = metadata.get("source_path", metadata.get("source", "unknown"))

        if doc_type == "legal":
            structured_text = _promote_legal_headings(content)
            header_docs = legal_header_splitter.split_text(structured_text)
            section_texts: list[tuple[str, dict[str, Any]]] = [
                (section.page_content, dict(section.metadata))
                for section in header_docs
                if section.page_content.strip()
            ]
            if not section_texts:
                section_texts = [(content, {})]

            for section_index, (section_text, header_metadata) in enumerate(section_texts):
                sub_chunks = legal_recursive_splitter.split_text(section_text)
                for sub_index, chunk_text in enumerate(sub_chunks):
                    chunk_text = chunk_text.strip()
                    if not chunk_text:
                        continue
                    chunk_metadata = {
                        **metadata,
                        **header_metadata,
                        "chunking_strategy": "markdown_headers_then_recursive",
                        "section_index": section_index,
                        "section_chunk_index": sub_index,
                        "chunk_size": CONFIG.legal_chunk_size,
                        "chunk_overlap": CONFIG.legal_chunk_overlap,
                    }
                    chunk_id = _stable_id(source_path, str(section_index), str(sub_index), chunk_text[:160])
                    chunks.append({"id": chunk_id, "content": chunk_text, "metadata": chunk_metadata})
        else:
            sub_chunks = news_recursive_splitter.split_text(content)
            for chunk_index, chunk_text in enumerate(sub_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_metadata = {
                    **metadata,
                    "chunking_strategy": "news_recursive",
                    "chunk_index": chunk_index,
                    "chunk_size": CONFIG.news_chunk_size,
                    "chunk_overlap": CONFIG.news_chunk_overlap,
                }
                chunk_id = _stable_id(source_path, str(chunk_index), chunk_text[:160])
                chunks.append({"id": chunk_id, "content": chunk_text, "metadata": chunk_metadata})

    for index, chunk in enumerate(chunks):
        chunk["metadata"]["global_chunk_index"] = index
        chunk["metadata"]["chunk_id"] = chunk["id"]

    return chunks


if __name__ == "__main__":
    loaded = load_markdown_documents()
    chunked = chunk_documents(loaded)
    print(f"Loaded {len(loaded)} docs -> {len(chunked)} chunks")
