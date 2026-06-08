"""
Task 10 - Generation Co Citation.

Pipeline:
    1. Lay chunks tu Task 9 retrieval pipeline.
    2. Reorder chunks de giam "lost in the middle".
    3. Format context kem citation label dang [Nguon, Nam].
    4. Goi LLM voi prompt bat buoc cite tung claim.
    5. Neu khong du evidence thi tra ve "I cannot verify this information".
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from .task9_retrieval_pipeline import retrieve
except ImportError:
    from src.task9_retrieval_pipeline import retrieve

load_dotenv(ENV_PATH)


# =============================================================================
# CONFIGURATION - Giai thich lua chon
# =============================================================================

# top_k=5: lay du evidence tu ca legal/news nhung context van gon. Voi chunk
# size 1200 cua Task 4, 5 chunks thuong nam trong vung context ngan, it gay
# nhieu va van phu hop document reordering.
TOP_K = 5

# top_p=0.85: nucleus sampling du thap de cau tra loi on dinh/factual trong RAG,
# nhung khong qua chat nhu 0.1-0.3 lam van phong tieng Viet bi cung.
TOP_P = 0.85

# temperature=0.2: uu tien trung thanh evidence, giam suy doan/sang tao.
TEMPERATURE = 0.2

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MAX_CONTEXT_CHARS = 9000
MIN_EVIDENCE_SCORE = 0.05
UNVERIFIABLE_ANSWER = "I cannot verify this information"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = f"""Answer the question in Vietnamese using ONLY the provided context.

For every factual claim, immediately add a citation using one of the exact
citation labels provided in the context, for example [VnExpress, 2026].

If the context does not explicitly support the answer, reply exactly:
{UNVERIFIABLE_ANSWER}

Rules:
- Do not use outside knowledge.
- Do not invent citations.
- Every factual sentence must include a citation.
- Prefer concise paragraphs or bullets.
- If sources disagree, say that the provided sources disagree and cite both."""


# =============================================================================
# DOCUMENT REORDERING (tranh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Sap xep chunks theo pattern: quan trong nhat o dau va cuoi.

    Input da duoc sort theo score descending. LLM thuong chu y manh o dau va
    cuoi prompt, nen ta giu chunk #1 o dau, cac chunk le tiep theo o than dau,
    va dua chunk #2 ve cuoi prompt.

    Vi du 5 chunks:
        [1, 2, 3, 4, 5] -> [1, 3, 5, 4, 2]
    """
    if len(chunks) <= 2:
        return list(chunks)

    front = [chunks[index] for index in range(0, len(chunks), 2)]
    back = [chunks[index] for index in range(1, len(chunks), 2)]
    back.reverse()
    return front + back


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def _slug_to_title(value: str) -> str:
    value = Path(value).stem
    value = re.sub(r"^article_\d+_", "", value)
    value = re.sub(r"[-_]+", " ", value).strip()
    return value[:1].upper() + value[1:] if value else "Unknown source"


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
    return mapping.get(domain, domain.split(".")[0].title() if domain else "Unknown source")


def _extract_year(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        matches = re.findall(r"\b(20\d{2}|19\d{2})\b", str(value))
        if matches:
            return matches[0]
    return "n.d."


def _citation_label(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata") or {}
    source_url = metadata.get("source_url") or metadata.get("url")
    source = metadata.get("source") or metadata.get("filename") or metadata.get("source_path")
    title = metadata.get("title") or source

    if source_url:
        parsed = urlparse(str(source_url))
        source_name = _source_from_domain(parsed.netloc)
    elif metadata.get("source_domain"):
        source_name = _source_from_domain(str(metadata["source_domain"]))
    elif metadata.get("doc_type") == "legal":
        source_name = _slug_to_title(str(title or source))
    else:
        source_name = _slug_to_title(str(title or source or "Unknown source"))

    year = _extract_year(
        metadata.get("date"),
        metadata.get("published_at"),
        metadata.get("date_crawled"),
        metadata.get("crawled"),
        metadata.get("source_path"),
        metadata.get("source"),
        metadata.get("title"),
        chunk.get("content", "")[:1500],
    )
    return f"[{source_name}, {year}]"


def _source_line(chunk: dict[str, Any], index: int) -> str:
    metadata = chunk.get("metadata") or {}
    citation = _citation_label(chunk)
    source_path = metadata.get("source_path") or metadata.get("source") or f"source_{index}"
    source_url = metadata.get("source_url") or metadata.get("url") or ""
    doc_type = metadata.get("doc_type") or metadata.get("type") or "unknown"
    score = float(chunk.get("score", 0.0) or 0.0)
    chunk_index = metadata.get("chunk_index", "n/a")

    url_part = f" | URL: {source_url}" if source_url else ""
    return (
        f"[Context {index} | Citation label: {citation} | "
        f"Source: {source_path} | Type: {doc_type} | "
        f"Chunk: {chunk_index} | Score: {score:.4f}{url_part}]"
    )


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format chunks thanh context string cho prompt.

    Moi chunk co "Citation label" dang [Nguon, Nam]. LLM duoc yeu cau chi
    cite bang cac label nay, nen output dat dung format bai yeu cau.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    used_chars = 0
    for index, chunk in enumerate(chunks, start=1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        header = _source_line(chunk, index)
        remaining = MAX_CONTEXT_CHARS - used_chars - len(header) - 8
        if remaining <= 0:
            break

        trimmed_content = content[:remaining].strip()
        parts.append(f"{header}\n{trimmed_content}")
        used_chars += len(header) + len(trimmed_content)

    return "\n\n---\n\n".join(parts)


# =============================================================================
# GENERATION
# =============================================================================

def _has_enough_evidence(chunks: list[dict[str, Any]]) -> bool:
    if not chunks:
        return False
    return max(float(chunk.get("score", 0.0) or 0.0) for chunk in chunks) >= MIN_EVIDENCE_SCORE


def _fallback_answer(query: str, chunks: list[dict[str, Any]]) -> str:
    """
    Fallback khi khong goi duoc LLM. Van tao cau tra loi co citation tu evidence
    top chunks de demo/test khong bi gay, nhung khong thay the LLM generation.
    """
    if not _has_enough_evidence(chunks):
        return UNVERIFIABLE_ANSWER

    sentences: list[str] = []
    for chunk in chunks[:3]:
        citation = _citation_label(chunk)
        content = re.sub(r"\s+", " ", str(chunk.get("content", "")).strip())
        if not content:
            continue
        snippet = content[:260].rstrip(" ,;:")
        sentences.append(f"{snippet} {citation}.")

    return " ".join(sentences) if sentences else UNVERIFIABLE_ANSWER


def _call_openai_llm(query: str, context: str) -> str:
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY chua co trong .env.")

    client = OpenAI()
    user_message = f"""Context:
{context}

---

Question: {query}

Use only the context above. Remember: every factual claim needs a citation label
from the context."""

    response = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )
    return (response.choices[0].message.content or "").strip()


def generate_with_citation(
    query: str,
    context_chunks: list[dict[str, Any]] | None = None,
    top_k: int = TOP_K,
) -> dict[str, Any]:
    """
    End-to-end RAG generation co citation.

    Args:
        query: Cau hoi cua user.
        context_chunks: Optional chunks da retrieve/rerank san. Neu None,
            ham tu goi Task 9 retrieve(query, top_k).
        top_k: So chunks dua vao context neu can retrieve tu Task 9.

    Returns:
        {
            "answer": str,
            "sources": list[dict],
            "retrieval_source": str,
            "context": str,
            "model": str,
        }
    """
    query = query.strip()
    if not query:
        raise ValueError("query khong duoc rong.")
    if top_k <= 0:
        return {
            "answer": UNVERIFIABLE_ANSWER,
            "sources": [],
            "retrieval_source": "none",
            "context": "",
            "model": OPENAI_CHAT_MODEL,
        }

    chunks = context_chunks if context_chunks is not None else retrieve(query, top_k=top_k)
    chunks = list(chunks or [])[:top_k]
    reordered_chunks = reorder_for_llm(chunks)
    context = format_context(reordered_chunks)

    if not _has_enough_evidence(chunks) or not context:
        answer = UNVERIFIABLE_ANSWER
        generation_error = None
    else:
        try:
            answer = _call_openai_llm(query, context)
            generation_error = None
            if not answer:
                answer = UNVERIFIABLE_ANSWER
        except Exception as exc:
            answer = _fallback_answer(query, reordered_chunks)
            generation_error = str(exc)[:300]

    retrieval_source = chunks[0].get("source", "unknown") if chunks else "none"
    result: dict[str, Any] = {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered_chunks,
        "retrieval_source": retrieval_source,
        "context": context,
        "model": OPENAI_CHAT_MODEL,
    }
    if generation_error:
        result["generation_error"] = generation_error
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task 10 generation with citation")
    parser.add_argument(
        "query",
        nargs="?",
        default="Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
    )
    parser.add_argument("--top-k", type=int, default=TOP_K)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = generate_with_citation(args.query, top_k=args.top_k)
    print(f"\nQ: {args.query}")
    print("-" * 80)
    print(output["answer"])
    print(f"\n[Sources: {len(output['sources'])} chunks | via {output['retrieval_source']}]")
