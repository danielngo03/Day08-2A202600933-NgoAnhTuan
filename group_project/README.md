# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **1 trong 2 sản phẩm**:

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
Chainlit UI (http://127.0.0.1:8000)
   │
   ├─ Interactive Settings (cl.ChatSettings)
   │     ├─ Lexical Method: BM25 (Whoosh) / TF-IDF (pure Python)
   │     ├─ HyDE: bật/tắt Hypothetical Document Embeddings
   │     └─ Reranking: bật/tắt Cross-Encoder
   │
   ├─ Conversation Summary Buffer Memory
   │     ├─ giữ vài lượt gần nhất (buffer)
   │     └─ tóm tắt lượt cũ bằng GPT-4o-mini (summary)
   │
   └─ GroupRAGPipeline
        │
        ├─ [1] Query Rewriting  ← GPT-4o-mini (chỉ kích hoạt khi có lịch sử)
        │
        ├─ [2] HyDE (Bonus — tuỳ chọn)
        │        └─ GPT-4o-mini sinh văn bản giả định → embed → dense search
        │
        ├─ [3] Ingestion & Chunking
        │    ├─ Legal (PDF/DOCX/MD): MarkdownHeaderTextSplitter
        │    │       → Chương / Mục / Điều / Khoản
        │    └─ News (JSON/HTML/TXT): RecursiveCharacterTextSplitter
        │
        ├─ [4] Hybrid Retrieval
        │    ├─ Dense : OpenAI text-embedding-3-small → Weaviate (Docker)
        │    │          fallback: local vector index
        │    ├─ Lexical: Whoosh BM25 (mặc định) hoặc TF-IDF thuần Python (Bonus)
        │    └─ Fusion : Reciprocal Rank Fusion (RRF α = 0.5)
        │
        ├─ [5] Reranking (tuỳ chọn)
        │        └─ BAAI/bge-reranker-v2-m3 CrossEncoder (local, không cần GPU)
        │
        └─ [6] Generation
                 ├─ GPT-4o-mini (temperature = 0.1)
                 ├─ Citation label inline [Nguồn, Năm]
                 └─ Hiển thị source documents trong sidebar
```

### Stack kỹ thuật

| Lớp | Công nghệ |
|-----|-----------|
| UI | Chainlit 2.x + `cl.ChatSettings` |
| Embedding | OpenAI `text-embedding-3-small` (1536-d) |
| Vector DB | Weaviate Local (Docker) + local fallback |
| Lexical | Whoosh BM25 · TF-IDF thuần Python (Bonus) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| Reranker | `BAAI/bge-reranker-v2-m3` (CrossEncoder local) |
| Memory | Summary Buffer Memory (GPT-4o-mini tóm tắt) |
| Generation | OpenAI `gpt-4o-mini` |
| Evaluation | DeepEval (Faithfulness / Relevancy / Recall / Precision) |

### Cấu trúc module

Project nhóm chạy độc lập từ root `group_project/`. Code Python chính nằm trong `src/`:

| Module | Vai trò |
|--------|---------|
| `.env` | Biến môi trường riêng của bài nhóm |
| `docker-compose.yml` | Weaviate Local Docker |
| `run.py` | Một lệnh chạy Docker + index + Chainlit |
| `src/config.py` | Config tập trung (model, top-k, threshold) |
| `src/data/landing/` | Raw data: PDF legal + JSON/HTML news |
| `src/data/standardized/` | Markdown chuẩn hoá từ landing |
| `src/chunking/legal_chunker.py` | Chunk văn bản luật theo cấu trúc pháp lý |
| `src/ingestion/standardize.py` | Convert landing → standardized |
| `src/retrieval/embeddings.py` | OpenAI embedding helper |
| `src/retrieval/weaviate_store.py` | Kết nối / index / search Weaviate Local |
| `src/retrieval/vectorless_bm25.py` | Whoosh BM25 **và** TF-IDF thuần Python (Bonus) |
| `src/retrieval/hybrid.py` | Dense + Lexical + RRF fusion |
| `src/reranking/local_cross_encoder.py` | Local reranker `BAAI/bge-reranker-v2-m3` |
| `src/memory/summary_memory.py` | Conversation summary buffer memory |
| `src/generation/answer_generator.py` | GPT-4o-mini trả lời có citation |
| `src/pipeline.py` | End-to-end RAG pipeline (HyDE + rerank + memory) |
| `src/ui/chainlit_app.py` | Chainlit chatbot UI + interactive settings |
| `evaluation/golden_dataset.json` | 15 cặp Q&A ground-truth |
| `evaluation/eval_pipeline.py` | DeepEval evaluation + A/B comparison |
| `evaluation/results.md` | Báo cáo kết quả evaluation |

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| | | | |
| | | | |
| | | | |
| | | | |

---

## Hướng Dẫn Chạy

```bash
# Đứng tại thư mục group_project
cd Lab8/group_project

# Tạo môi trường riêng cho bài nhóm
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Tạo env riêng cho bài nhóm
cp .env.example .env
# Sau đó điền OPENAI_API_KEY vào .env

# Một lệnh chạy toàn bộ:
# 1. bật Weaviate Docker
# 2. đợi Weaviate sẵn sàng
# 3. auto-standardize src/data/landing -> src/data/standardized
# 4. index Whoosh + Weaviate
# 5. mở Chainlit UI
.venv/bin/python run.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:8000
```

Các lệnh phụ:

```bash
# Chỉ bật Docker + index, không mở UI
.venv/bin/python run.py --no-ui

# Không bật Docker, chỉ dùng local dense fallback + Whoosh BM25
.venv/bin/python run.py --no-docker --skip-weaviate-index

# Reset toàn bộ index trước khi ghi lại
.venv/bin/python run.py --reset-index

# Tự bật Docker thủ công nếu cần
docker compose up -d
```

Biến môi trường nằm riêng trong `group_project/.env`:

```bash
OPENAI_API_KEY=...
OPENAI_CHAT_MODEL=gpt-4o-mini
AUTO_START_DOCKER=true
AUTO_INDEX_ON_START=true
WEAVIATE_LOCAL_HOST=localhost
WEAVIATE_LOCAL_HTTP_PORT=8080
WEAVIATE_LOCAL_GRPC_PORT=50051
CHAINLIT_HOST=127.0.0.1
CHAINLIT_PORT=8000
LOCAL_RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### Thêm dữ liệu mới

- Legal PDF/DOCX: đặt vào `src/data/landing/legal/`
- News JSON/HTML/MD/TXT: đặt vào `src/data/landing/news/`
- Chạy lại:

```bash
.venv/bin/python -m src.ingestion.standardize
.venv/bin/python -m src.index_weaviate --reset
```

Khi index/chunk, hệ thống cũng tự gọi standardize nếu `AUTO_STANDARDIZE=true` trong `.env`, nên file mới ở landing sẽ được cập nhật sang `src/data/standardized/`.

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
