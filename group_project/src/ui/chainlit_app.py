"""Chainlit UI for the group RAG chatbot."""

from __future__ import annotations

import sys
from pathlib import Path

import chainlit as cl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chainlit.types import ThreadDict
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.input_widget import Select, Switch
from src.generation.citations import source_documents_markdown
from src.pipeline import GroupRAGPipeline


def init_db():
    import sqlite3
    db_path = PROJECT_ROOT / "chainlit.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        identifier TEXT NOT NULL UNIQUE,
        createdAt TEXT,
        metadata TEXT
    )
    """)
    
    # Create threads table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS threads (
        id TEXT PRIMARY KEY,
        createdAt TEXT,
        name TEXT,
        userId TEXT,
        userIdentifier TEXT,
        tags TEXT,
        metadata TEXT
    )
    """)
    
    # Create steps table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        threadId TEXT NOT NULL,
        parentId TEXT,
        streaming BOOLEAN NOT NULL,
        waitForAnswer BOOLEAN,
        isError BOOLEAN,
        metadata TEXT,
        tags TEXT,
        input TEXT,
        output TEXT,
        createdAt TEXT,
        command TEXT,
        start TEXT,
        end TEXT,
        generation TEXT,
        showInput TEXT,
        language TEXT,
        indent INTEGER,
        defaultOpen BOOLEAN,
        modes TEXT,
        disableFeedback BOOLEAN
    )
    """)
    
    # Create elements table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY,
        threadId TEXT,
        type TEXT,
        url TEXT,
        chainlitKey TEXT,
        name TEXT NOT NULL,
        display TEXT,
        objectKey TEXT,
        size TEXT,
        page INTEGER,
        language TEXT,
        forId TEXT,
        mime TEXT,
        props TEXT
    )
    """)
    
    # Create feedbacks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY,
        forId TEXT NOT NULL,
        value INTEGER NOT NULL,
        comment TEXT
    )
    """)
    
    conn.commit()
    conn.close()


@cl.data_layer
def get_data_layer():
    init_db()
    db_path = PROJECT_ROOT / "chainlit.db"
    return SQLAlchemyDataLayer(conninfo=f"sqlite+aiosqlite:///{db_path}")


@cl.password_auth_callback
async def auth(username: str, password: str):
    # Allow any login credentials for the local demo session
    return cl.User(identifier=username or "user", metadata={"role": "user"})


WELCOME = """# Drug Law RAG Chatbot

Hỏi về pháp luật ma tuý Việt Nam hoặc các tin tức nghệ sĩ liên quan tới ma tuý.

Ví dụ:
- Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249?
- Luật Phòng chống ma tuý quy định các hình thức cai nghiện nào?
- Những nghệ sĩ nào trong bộ dữ liệu bị nhắc tới vì liên quan ma tuý?

Gõ `/reset` để xoá memory hội thoại.

---
### 🛠️ Cấu hình Tìm kiếm (Chat Settings):
Bạn có thể thay đổi cấu hình tìm kiếm ở thanh bên (Sidebar):
* **Lexical Search (BM25 vs. TF-IDF)**:
  - *BM25*: Tiêu chuẩn công nghiệp, giới hạn độ bão hòa tần suất từ (TF saturation) và độ dài tài liệu.
  - *TF-IDF*: Tính điểm tuyến tính theo TF và IDF, thích hợp tìm kiếm từ khóa hiếm.
* **HyDE (Hypothetical Document Embeddings)**: Giúp LLM tự tưởng tượng ra câu trả lời trước, sau đó dùng câu trả lời giả định để so khớp vector.
* **Reranking**: Sắp xếp lại bằng Cross-Encoder (`BAAI/bge-reranker-v2-m3`) để tăng độ chính xác."""


@cl.on_settings_update
async def setup_agent(settings):
    pipeline = cl.user_session.get("pipeline")
    if pipeline is not None:
        pipeline.lexical_method = settings["lexical_method"]
        pipeline.use_hyde = settings["use_hyde"]
        pipeline.use_reranking = settings["use_reranking"]
        await cl.Message(content=f"Đã cập nhật cấu hình RAG: Lexical Method = `{pipeline.lexical_method.upper()}`, HyDE = `{pipeline.use_hyde}`, Reranking = `{pipeline.use_reranking}`").send()


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = await cl.ChatSettings([
        Select(
            id="lexical_method",
            label="Lexical Search Algorithm",
            values=["bm25", "tfidf"],
            initial_index=0,
            description="BM25: Tần suất từ khóa có bão hòa và phạt độ dài văn bản. TF-IDF: Điểm tăng tuyến tính theo số lần xuất hiện của từ khóa."
        ),
        Switch(
            id="use_hyde",
            label="Hypothetical Document Embeddings (HyDE)",
            initial=False,
            description="Sinh câu trả lời giả định trước khi embed để cải thiện kết quả tìm kiếm ngữ cảnh."
        ),
        Switch(
            id="use_reranking",
            label="Use Cross-Encoder Reranker",
            initial=True,
            description="Sử dụng BAAI/bge-reranker-v2-m3 để chấm điểm lại các tài liệu tìm thấy."
        )
    ]).send()

    pipeline = GroupRAGPipeline()
    pipeline.lexical_method = settings["lexical_method"]
    pipeline.use_hyde = settings["use_hyde"]
    pipeline.use_reranking = settings["use_reranking"]
    
    cl.user_session.set("pipeline", pipeline)
    await cl.Message(content=WELCOME).send()


@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict) -> None:
    pipeline = GroupRAGPipeline()
    
    # Reconstruct conversation memory from the database thread steps
    steps = thread.get("steps", [])
    user_msg = None
    for step in steps:
        step_type = step.get("type")
        step_name = step.get("name")
        output = (step.get("output") or "").strip()
        
        is_user = step_type == "user_message" or step_name == "User"
        is_assistant = step_type == "assistant_message" or step_name in {"Assistant", "assistant"}
        
        if is_user:
            user_msg = output
        elif is_assistant and user_msg:
            pipeline.memory.add_turn(user_msg, output)
            user_msg = None
            
    cl.user_session.set("pipeline", pipeline)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    content = message.content.strip()
    pipeline: GroupRAGPipeline = cl.user_session.get("pipeline")
    if pipeline is None:
        pipeline = GroupRAGPipeline()
        cl.user_session.set("pipeline", pipeline)

    if content.lower() in {"/reset", "reset"}:
        pipeline.reset_memory()
        await cl.Message(content="Đã xoá conversation memory. Bạn có thể hỏi lại từ đầu.").send()
        return

    thinking = cl.Message(content="Đang tìm kiếm tài liệu, rerank và tạo câu trả lời có citation...")
    await thinking.send()

    try:
        result = await cl.make_async(pipeline.ask)(content)
    except Exception as exc:
        thinking.content = f"Lỗi khi chạy pipeline: `{exc}`"
        await thinking.update()
        return

    sources_markdown = source_documents_markdown(result.get("reordered_sources") or result.get("sources", []))
    elements = [
        cl.Text(
            name="Source documents",
            content=sources_markdown,
            display="side",
        )
    ]

    model = result.get("model", "unknown")
    retrieval_source = result.get("retrieval_source", "unknown")
    footer = f"\n\n---\nModel: `{model}` | Retrieval: `{retrieval_source}`"
    if result.get("generation_error"):
        footer += f"\nFallback note: `{result['generation_error']}`"

    thinking.content = result["answer"] + footer
    thinking.elements = elements
    await thinking.update()
