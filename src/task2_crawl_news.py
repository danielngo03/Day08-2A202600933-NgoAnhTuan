# pyright: reportMissingImports=false
"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://tuoitre.vn/rapper-binh-gold-bi-bat-vi-cuop-tai-san-duong-tinh-voi-ma-tuy-20250726185902989.htm",
    "https://vnexpress.net/nguoi-mau-andrea-aybar-cung-tro-ly-lam-tiec-ma-tuy-trong-can-ho-cao-cap-5059429.html",
    "https://kenh14.vn/ca-si-chi-dan-ru-ban-hun-tien-mua-ma-tuy-nguoi-mau-an-tay-ru-tro-ly-quay-tiktok-cung-bay-lac-tang-tru-chat-cam-ngay-tai-can-ho-hang-sang-215260402204441991.chn",
    "https://xaydungchinhsach.chinhphu.vn/khoi-to-bat-tam-giam-long-nhat-son-ngoc-minh-cung-69-bi-can-119260520124509053.htm",
    "https://vtv.vn/phap-luat/ca-si-chu-bin-bi-bat-vi-lien-quan-ma-tuy-20240607115007528.htm",
]


def _safe_filename(index: int, url: str) -> str:
    """Tạo tên file ổn định từ domain và slug URL."""
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace("www.", "").split(".")[0]
    slug = Path(parsed_url.path).stem or f"article-{index:02d}"
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", slug).strip("-").lower()
    return f"article_{index:02d}_{domain}_{slug[:80]}.json"


async def crawl_article(url: str, crawler) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    result = await crawler.arun(url=url)

    metadata = result.metadata or {}
    title = (
        metadata.get("title")
        or metadata.get("og:title")
        or metadata.get("twitter:title")
        or "Unknown"
    )
    content_markdown = result.markdown or ""

    return {
        "url": url,
        "title": title.strip(),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "success": bool(getattr(result, "success", True)),
        "status_code": getattr(result, "status_code", None),
        "source_domain": urlparse(url).netloc,
        "content_markdown": content_markdown.strip(),
    }


async def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    from crawl4ai import AsyncWebCrawler
    from crawl4ai.async_configs import BrowserConfig

    setup_directory()

    browser_config = BrowserConfig(
        browser_type="chromium",
        chrome_channel="chrome",
        headless=True,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        for i, url in enumerate(ARTICLE_URLS, 1):
            print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
            article = await crawl_article(url, crawler)

            filename = _safe_filename(i, url)
            filepath = DATA_DIR / filename
            filepath.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        asyncio.run(crawl_all())
