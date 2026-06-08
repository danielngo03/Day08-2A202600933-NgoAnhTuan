"""
Conversation summary buffer memory.

Instead of injecting the whole chat history into every prompt, this class keeps
the most recent turns verbatim and summarizes older turns with GPT-4o-mini.
That follows the ConversationSummaryBufferMemory idea while staying lightweight
and dependency-stable for this project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from openai import OpenAI

from src.config import CONFIG


@dataclass
class ChatTurn:
    user: str
    assistant: str


@dataclass
class SummaryBufferMemory:
    summary: str = ""
    turns: list[ChatTurn] = field(default_factory=list)
    recent_turns: int = CONFIG.memory_recent_turns
    summary_trigger_turns: int = CONFIG.memory_summary_trigger_turns

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append(ChatTurn(user=user.strip(), assistant=assistant.strip()))
        if len(self.turns) > self.summary_trigger_turns:
            self._summarize_old_turns()

    def _summarize_old_turns(self) -> None:
        if len(self.turns) <= self.recent_turns:
            return

        old_turns = self.turns[: -self.recent_turns]
        self.turns = self.turns[-self.recent_turns :]
        old_text = "\n".join(
            f"User: {turn.user}\nAssistant: {turn.assistant}"
            for turn in old_turns
        )

        if not os.getenv("OPENAI_API_KEY"):
            compressed = old_text[-1200:]
            self.summary = f"{self.summary}\n{compressed}".strip()
            return

        client = OpenAI()
        prompt = f"""Summarize this legal/news RAG conversation in Vietnamese.
Keep user intent, entities, article/law references, and unresolved follow-up
topics. Be concise.

Existing summary:
{self.summary or "(none)"}

Older turns:
{old_text}"""
        response = client.chat.completions.create(
            model=CONFIG.openai_chat_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            top_p=0.8,
        )
        self.summary = (response.choices[0].message.content or "").strip()

    def format_for_prompt(self) -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f"Conversation summary:\n{self.summary}")
        if self.turns:
            recent = "\n".join(
                f"User: {turn.user}\nAssistant: {turn.assistant}"
                for turn in self.turns[-self.recent_turns :]
            )
            parts.append(f"Recent turns:\n{recent}")
        return "\n\n".join(parts)

    def retrieval_hint(self) -> str:
        """Compact hint appended to retrieval query for follow-up questions."""
        recent_questions = " ".join(turn.user for turn in self.turns[-2:])
        if self.summary:
            return f"{self.summary}\n{recent_questions}".strip()
        return recent_questions.strip()

    def reset(self) -> None:
        self.summary = ""
        self.turns.clear()

