"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
LEGAL_EXTENSIONS = {".pdf", ".docx", ".doc"}


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()
    converted = []

    for filepath in sorted(legal_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() not in LEGAL_EXTENSIONS:
            continue

        print(f"Converting: {filepath.name}")
        result = md.convert(filepath)
        output_path = output_dir / f"{filepath.stem}.md"
        content = result.text_content.strip()
        output_path.write_text(content + "\n", encoding="utf-8")
        converted.append(output_path)
        print(f"  ✓ Saved: {output_path}")

    return converted


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []

    for filepath in sorted(news_dir.iterdir()):
        if not filepath.is_file() or filepath.suffix.lower() != ".json":
            continue

        print(f"Converting: {filepath.name}")
        data = json.loads(filepath.read_text(encoding="utf-8"))
        output_path = output_dir / f"{filepath.stem}.md"

        title = data.get("title") or "Unknown"
        url = data.get("url") or "N/A"
        date_crawled = data.get("date_crawled") or "N/A"
        source_domain = data.get("source_domain") or "N/A"
        content_markdown = (data.get("content_markdown") or "").strip()

        header = [
            f"# {title}",
            "",
            f"**Source:** {url}",
            f"**Source domain:** {source_domain}",
            f"**Crawled:** {date_crawled}",
            f"**Converted:** {datetime.now(timezone.utc).isoformat()}",
            "",
            "---",
            "",
        ]
        content = "\n".join(header) + content_markdown + "\n"
        output_path.write_text(content, encoding="utf-8")
        converted.append(output_path)
        print(f"  ✓ Saved: {output_path}")

    return converted


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    legal_files = convert_legal_docs()

    print("\n--- News Articles ---")
    news_files = convert_news_articles()

    print(f"\n✓ Converted {len(legal_files)} legal files and {len(news_files)} news files.")
    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
