"""GPT-4o-mini answer generation with citations."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

from src.config import CONFIG
from src.generation.citations import citation_label, format_context, reorder_for_llm
from src.memory.summary_memory import SummaryBufferMemory

UNVERIFIABLE = "Tôi không tìm thấy thông tin liên quan trong tài liệu đã truy vấn."

SYSTEM_PROMPT = f"""Bạn là chatbot RAG chuyên về luật phòng chống ma túy và tin tức liên quan, trả lời bằng tiếng Việt.

NGUYÊN TẮC CHÍNH:
1. Tổng hợp câu trả lời TỪ ngữ cảnh đã truy vấn bên dưới. Nếu thông tin có trong context, BẮT BUỘC phải sử dụng.
2. Với mỗi luận điểm thực tế, thêm nhãn trích dẫn CHÍNH XÁC từ context, ví dụ: [Luật PCMT 2021, 2021] hoặc [VnExpress, 2026].
3. Nếu context chứa thông tin liên quan (dù không hoàn toàn khớp từng chữ), HÃY tổng hợp và trả lời dựa trên đó.
4. Chỉ trả lời "{UNVERIFIABLE}" khi context HOÀN TOÀN trống hoặc không liên quan chút nào đến câu hỏi.

QUY TẮC BỔ SUNG:
- KHÔNG dùng kiến thức bên ngoài — chỉ dùng thông tin trong context được cung cấp.
- KHÔNG bịa đặt điều luật, hình phạt, ngày tháng, tên người, hay trích dẫn.
- Nếu trả lời câu hỏi tiếp theo, dùng bộ nhớ hội thoại để giải mã đại từ, nhưng trích dẫn từ context mới.
- Nếu context có thông tin nhưng không đầy đủ, hãy trả lời phần có thể và nêu rõ sự không chắc chắn."""


def _fallback_answer(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return UNVERIFIABLE

    snippets: list[str] = []
    for chunk in chunks[:3]:
        text = re.sub(r"\s+", " ", str(chunk.get("content", "")).strip())
        if text:
            snippets.append(f"{text[:280].rstrip(' ,;:')} {citation_label(chunk)}.")
    return " ".join(snippets) if snippets else UNVERIFIABLE


def generate_answer(
    query: str,
    chunks: list[dict[str, Any]],
    memory: SummaryBufferMemory | None = None,
) -> dict[str, Any]:
    """Generate answer with citations from reranked chunks."""
    reordered = reorder_for_llm(chunks[: CONFIG.generation_top_k])
    context = format_context(reordered)
    memory_context = memory.format_for_prompt() if memory else ""

    if not context:
        return {
            "answer": UNVERIFIABLE,
            "sources": [],
            "reordered_sources": [],
            "context": "",
            "model": CONFIG.openai_chat_model,
            "generation_error": "empty_context",
        }

    if not os.getenv("OPENAI_API_KEY"):
        return {
            "answer": _fallback_answer(reordered),
            "sources": chunks,
            "reordered_sources": reordered,
            "context": context,
            "model": "extractive_fallback",
            "generation_error": "missing_OPENAI_API_KEY",
        }

    user_prompt = f"""Conversation memory:
{memory_context or "(none)"}

Retrieved context:
{context}

Question:
{query}

Return a grounded answer with citations."""

    try:
        client = OpenAI()
        response = client.chat.completions.create(
            model=CONFIG.openai_chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=CONFIG.temperature,
            top_p=CONFIG.top_p,
        )
        answer = (response.choices[0].message.content or "").strip() or UNVERIFIABLE
        error = ""
        model = CONFIG.openai_chat_model
    except Exception as exc:
        answer = _fallback_answer(reordered)
        error = str(exc)[:300]
        model = "extractive_fallback"

    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "context": context,
        "model": model,
        "generation_error": error,
    }

