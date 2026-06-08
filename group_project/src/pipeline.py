"""End-to-end group RAG chatbot pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import os
from openai import OpenAI
from src.config import CONFIG
from src.generation.answer_generator import generate_answer
from src.memory.summary_memory import SummaryBufferMemory
from src.reranking.local_cross_encoder import rerank_local_cross_encoder
from src.retrieval.hybrid import hybrid_search


@dataclass
class GroupRAGPipeline:
    """
    Group RAG chatbot pipeline.

    Stack:
        MarkdownHeader legal chunking -> OpenAI embeddings -> Weaviate dense
        + Whoosh BM25 vectorless -> RRF fusion -> local cross-encoder rerank
        -> GPT-4o-mini generation with citations -> summary buffer memory.
    """

    memory: SummaryBufferMemory = field(default_factory=SummaryBufferMemory)
    use_reranking: bool = True
    use_hyde: bool = False
    lexical_method: str = "bm25"

    def _retrieval_query(self, query: str) -> str:
        if not self.memory.turns:
            return query

        if not os.getenv("OPENAI_API_KEY"):
            return query

        # Format recent turns as context for the query rewriter
        history_text = ""
        for turn in self.memory.turns[-3:]:
            history_text += f"User: {turn.user}\nAssistant: {turn.assistant}\n"

        prompt = f"""Given the following conversation history and a follow-up question, rephrase the follow-up question into a standalone, search-friendly query in Vietnamese.
If the follow-up question is already a new, independent topic/entity (e.g., "Chi Dân", "Luật phòng chống ma túy"), return it EXACTLY as it is, without changing a single word.
Do not add any explanations or preamble.

Conversation History:
{history_text}

Follow-up Question:
{query}

Standalone Query:"""

        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=CONFIG.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            rewritten = (response.choices[0].message.content or "").strip()
            print(f"[Query Rewrite] '{query}' -> '{rewritten}'")
            return rewritten
        except Exception as exc:
            print(f"[Query Rewrite Error] {exc}. Using raw query.")
            return query

    def _generate_hypothetical_document(self, query: str) -> str:
        if not os.getenv("OPENAI_API_KEY"):
            return query

        prompt = f"""Write a short paragraph in Vietnamese that directly answers the following search query.
Write it in the style of an official legal document or a factual news article about drug violations in Vietnam.
Do not include any introductory sentences, meta-commentary, or references. Just output the hypothetical text.

Query: {query}

Hypothetical Document:"""
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model=CONFIG.openai_chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            hyde_doc = (response.choices[0].message.content or "").strip()
            print(f"[HyDE] Generated hypothetical document: '{hyde_doc[:120]}...'")
            return hyde_doc
        except Exception as exc:
            print(f"[HyDE Error] {exc}. Using raw query as document.")
            return query

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        retrieval_query = self._retrieval_query(query)
        
        query_vector = None
        if self.use_hyde:
            from src.retrieval.embeddings import embed_query
            hyde_doc = self._generate_hypothetical_document(retrieval_query)
            try:
                query_vector = embed_query(hyde_doc)
            except Exception as exc:
                print(f"[HyDE Embedding Error] {exc}")
                
        candidates = hybrid_search(
            retrieval_query,
            top_k=top_k or CONFIG.retrieval_top_k,
            query_vector=query_vector,
            lexical_method=self.lexical_method,
        )
        if not self.use_reranking:
            return candidates[:CONFIG.rerank_top_k]
        reranked = rerank_local_cross_encoder(
            retrieval_query,
            candidates,
            top_k=CONFIG.rerank_top_k,
        )
        return reranked

    def ask(self, query: str) -> dict[str, Any]:
        query = query.strip()
        if not query:
            raise ValueError("query khong duoc rong.")

        sources = self.retrieve(query)
        result = generate_answer(query=query, chunks=sources, memory=self.memory)
        self.memory.add_turn(query, result["answer"])
        result["retrieval_source"] = sources[0].get("source", "none") if sources else "none"
        result["memory_summary"] = self.memory.summary
        result["config"] = {
            "embedding_model": CONFIG.openai_embedding_model,
            "chat_model": CONFIG.openai_chat_model,
            "reranker_model": CONFIG.reranker_model,
            "dense_store": "weaviate_local_with_local_fallback",
            "vectorless_store": "whoosh_bm25_with_rank_bm25_fallback",
            "chunking": "legal_markdown_headers_then_recursive",
        }
        return result

    def reset_memory(self) -> None:
        self.memory.reset()


def ask(query: str) -> dict[str, Any]:
    """Convenience one-shot helper for evaluation scripts."""
    return GroupRAGPipeline().ask(query)


if __name__ == "__main__":
    pipeline = GroupRAGPipeline()
    question = "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249?"
    output = pipeline.ask(question)
    print(output["answer"])
    print(f"Sources: {len(output['sources'])}")
