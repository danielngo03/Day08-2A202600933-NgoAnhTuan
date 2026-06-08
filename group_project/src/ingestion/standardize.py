"""
Standardize group landing data into markdown.

Input:
    src/data/landing/legal/*.pdf|*.docx|*.doc|*.md
    src/data/landing/news/*.json|*.html|*.txt|*.md

Output:
    src/data/standardized/legal/*.md
    src/data/standardized/news/*.md

The indexer calls ensure_standardized() before chunking, so adding new legal
PDFs or news JSON files to landing automatically refreshes standardized files.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import CONFIG

LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc", ".md", ".txt"}
NEWS_EXTENSIONS = {".json", ".html", ".htm", ".md", ".txt"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _target_path(source_path: Path) -> Path:
    relative = source_path.relative_to(CONFIG.landing_dir)
    return CONFIG.standardized_dir / relative.with_suffix(".md")


def _needs_update(source_path: Path, target_path: Path, force: bool = False) -> bool:
    if force or not target_path.exists():
        return True
    return source_path.stat().st_mtime > target_path.stat().st_mtime


def _markdown_heading(text: str, fallback: str) -> str:
    stripped = re.sub(r"\s+", " ", text).strip()
    return stripped or fallback


def _convert_with_markitdown(source_path: Path) -> str:
    from markitdown import MarkItDown

    result = MarkItDown().convert(str(source_path))
    return (result.text_content or "").strip()


def _standardize_legal(source_path: Path) -> str:
    if source_path.suffix.lower() in {".md", ".txt"}:
        content = source_path.read_text(encoding="utf-8").strip()
    else:
        content = _convert_with_markitdown(source_path)

    title = _markdown_heading(source_path.stem.replace("-", " "), source_path.stem)
    return (
        f"# {title}\n\n"
        f"**Source file:** {source_path.name}\n"
        f"**Source path:** {source_path.relative_to(CONFIG.landing_dir)}\n"
        f"**Converted:** {_now_iso()}\n\n"
        "---\n\n"
        f"{content}\n"
    )


def _standardize_news_json(source_path: Path) -> str:
    data: dict[str, Any] = json.loads(source_path.read_text(encoding="utf-8"))
    title = data.get("title") or source_path.stem
    url = data.get("url", "")
    domain = data.get("source_domain", "")
    crawled = data.get("date_crawled", "")
    content = (
        data.get("content_markdown")
        or data.get("markdown")
        or data.get("content")
        or data.get("html")
        or ""
    )

    return (
        f"# {title}\n\n"
        f"**Source:** {url}\n"
        f"**Source domain:** {domain}\n"
        f"**Source file:** {source_path.name}\n"
        f"**Crawled:** {crawled}\n"
        f"**Converted:** {_now_iso()}\n\n"
        "---\n\n"
        f"{content.strip()}\n"
    )


def _standardize_news_text(source_path: Path) -> str:
    content = source_path.read_text(encoding="utf-8").strip()
    title = source_path.stem.replace("-", " ")
    return (
        f"# {title}\n\n"
        f"**Source file:** {source_path.name}\n"
        f"**Source path:** {source_path.relative_to(CONFIG.landing_dir)}\n"
        f"**Converted:** {_now_iso()}\n\n"
        "---\n\n"
        f"{content}\n"
    )


def standardize_file(source_path: Path, force: bool = False) -> Path | None:
    """Convert one landing file to markdown if missing or stale."""
    suffix = source_path.suffix.lower()
    if "legal" in source_path.parts:
        if suffix not in LEGAL_EXTENSIONS:
            return None
        markdown = _standardize_legal(source_path)
    elif "news" in source_path.parts:
        if suffix not in NEWS_EXTENSIONS:
            return None
        markdown = _standardize_news_json(source_path) if suffix == ".json" else _standardize_news_text(source_path)
    else:
        return None

    target = _target_path(source_path)
    if not _needs_update(source_path, target, force=force):
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return target


def ensure_standardized(force: bool = False) -> list[Path]:
    """Refresh standardized markdown for all landing files."""
    CONFIG.standardized_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not CONFIG.landing_dir.exists():
        return written

    for source_path in sorted(CONFIG.landing_dir.rglob("*")):
        if not source_path.is_file() or source_path.name.startswith("."):
            continue
        target = standardize_file(source_path, force=force)
        if target:
            written.append(target)
    return written


if __name__ == "__main__":
    outputs = ensure_standardized(force=False)
    print(f"Standardized/checked {len(outputs)} files in {CONFIG.standardized_dir}")
