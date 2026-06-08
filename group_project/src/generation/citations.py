"""Citation labels and context formatting for the chatbot."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from src.config import CONFIG


def reorder_for_llm(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Place strongest chunks at the beginning and end of the prompt."""
    if len(chunks) <= 2:
        return list(chunks)
    front = [chunks[i] for i in range(0, len(chunks), 2)]
    back = [chunks[i] for i in range(1, len(chunks), 2)]
    return front + list(reversed(back))


def _source_from_domain(domain: str) -> str:
    domain = domain.lower().replace("www.", "")
    mapping = {
        "vnexpress.net": "VnExpress",
        "tuoitre.vn": "Tuoi Tre",
        "kenh14.vn": "Kenh14",
        "vtv.vn": "VTV",
        "xaydungchinhsach.chinhphu.vn": "Chinhphu.vn",
        "chinhphu.vn": "Chinhphu.vn",
    }
    return mapping.get(domain, domain.split(".")[0].title() if domain else "Unknown")


def _clean_title(value: str) -> str:
    stem = Path(value).stem
    stem = re.sub(r"^article_\d+_", "", stem)
    stem = re.sub(r"[-_]+", " ", stem).strip()
    return stem[:1].upper() + stem[1:] if stem else "Unknown"


def _extract_year(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        match = re.search(r"\b(20\d{2}|19\d{2})\b", str(value))
        if match:
            return match.group(1)
    return "n.d."


def citation_label(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    source_url = metadata.get("source_url") or metadata.get("url")
    if source_url:
        source_name = _source_from_domain(urlparse(str(source_url)).netloc)
    elif metadata.get("source_domain"):
        source_name = _source_from_domain(str(metadata["source_domain"]))
    else:
        source_name = _clean_title(str(metadata.get("title") or metadata.get("source") or "Unknown"))

    year = _extract_year(
        metadata.get("date_crawled"),
        metadata.get("published_at"),
        metadata.get("source_path"),
        metadata.get("source"),
        metadata.get("title"),
        chunk.get("content", "")[:1200],
    )
    return f"[{source_name}, {year}]"


def format_context(chunks: list[dict[str, Any]]) -> str:
    """Format reranked chunks as prompt context with exact citation labels."""
    parts: list[str] = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        metadata = chunk.get("metadata") or {}
        label = citation_label(chunk)
        header = (
            f"[Source {index} | Citation: {label} | "
            f"File: {metadata.get('source_path') or metadata.get('source')} | "
            f"Type: {metadata.get('doc_type', 'unknown')} | "
            f"Score: {float(chunk.get('score', 0.0) or 0.0):.4f}]"
        )
        url = metadata.get("source_url") or metadata.get("url")
        if url:
            header += f"\nURL: {url}"

        remaining = CONFIG.max_context_chars - used_chars - len(header)
        if remaining <= 0:
            break
        trimmed = content[:remaining].strip()
        parts.append(f"{header}\n{trimmed}")
        used_chars += len(header) + len(trimmed)

    return "\n\n---\n\n".join(parts)


def source_documents_markdown(chunks: list[dict[str, Any]]) -> str:
    """Create markdown source list for Chainlit side panel/elements."""
    lines = ["# Source documents used"]
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        label = citation_label(chunk)
        source = metadata.get("source_path") or metadata.get("source") or "unknown"
        score = float(chunk.get("score", 0.0) or 0.0)
        url = metadata.get("source_url") or metadata.get("url")
        lines.append(f"\n## {index}. {label}")
        lines.append(f"- Source: `{source}`")
        lines.append(f"- Score: `{score:.4f}`")
        if url:
            lines.append(f"- URL: {url}")
        lines.append("\n```text")
        lines.append(str(chunk.get("content", ""))[:1200])
        lines.append("```")
    return "\n".join(lines)

