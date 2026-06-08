# Ngày 8 — RAG Pipeline v2

**Chủ đề: Pháp luật Việt Nam về ma tuý và các chất cấm + Tin tức nghệ sĩ liên quan**

---

## Cấu Trúc Thư Mục

```
Lab8/
├── README.md
├── .env                        ← API keys (OPENAI, JINA, PAGEINDEX)
├── .env.example
├── requirements.txt
├── data/
│   ├── landing/
│   │   ├── legal/              ← 4 văn bản pháp luật (PDF)
│   │   │   ├── bo-luat-hinh-su-2015.pdf
│   │   │   ├── luat-phong-chong-ma-tuy-2025.pdf
│   │   │   ├── nghi-dinh-19-2018-nd-cp.pdf
│   │   │   └── nghi-dinh-28-2026-nd-cp.pdf
│   │   └── news/               ← 6 bài báo crawl (JSON)
│   │       ├── article_01_vnexpress_ca-si-miu-le...json
│   │       ├── article_02_tuoitre_rapper-binh-gold...json
│   │       ├── article_03_vnexpress_nguoi-mau-andrea-aybar...json
│   │       ├── article_04_kenh14_ca-si-chi-dan...json
│   │       ├── article_05_xaydungchinhsach_khoi-to-long-nhat...json
│   │       └── article_06_vtv_ca-si-chu-bin...json
│   ├── standardized/           ← Markdown đã convert (Task 3)
│   │   ├── legal/*.md
│   │   └── news/*.md
│   ├── indexes/                ← Local vector stores (Task 4)
│   │   ├── bge_m3/             ← chunks.jsonl + embeddings.npy (BAAI/bge-m3)
│   │   └── openai_text_embedding_3_small/  ← chunks.jsonl + embeddings.npy
│   └── pageindex/              ← PageIndex cache (Task 8)
├── src/
│   ├── task1_collect_legal_docs.py
│   ├── task2_crawl_news.py
│   ├── task3_convert_markdown.py
│   ├── task4_chunking_indexing.py
│   ├── task5_semantic_search.py
│   ├── task6_lexical_search.py
│   ├── task7_reranking.py
│   ├── task8_pageindex_vectorless.py
│   ├── task9_retrieval_pipeline.py
│   └── task10_generation.py
├── tests/
│   └── test_individual.py      ← Automated test suite (pytest)
└── group_project/              ← Bài tập nhóm RAG Chatbot
    ├── README.md
    ├── run.py
    ├── docker-compose.yml
    ├── src/
    │   ├── pipeline.py
    │   ├── ui/chainlit_app.py
    │   ├── retrieval/
    │   ├── generation/
    │   ├── memory/
    │   └── reranking/
    └── evaluation/
        ├── golden_dataset.json
        ├── eval_pipeline.py
        └── results.md
```

---

## Bài Tập Cá Nhân — 10 Tasks

### Task 1 — Thu Thập Văn Bản Pháp Luật

Tải về và lưu **4 văn bản pháp luật** dạng PDF về ma tuý vào `data/landing/legal/`:

| File | Mô tả |
|------|-------|
| `bo-luat-hinh-su-2015.pdf` | Bộ luật Hình sự 2015 (sửa đổi 2017) — Chương XX: Tội phạm về ma tuý |
| `luat-phong-chong-ma-tuy-2025.pdf` | Luật Phòng, chống ma tuý 2021 (73/2021/QH15) |
| `nghi-dinh-19-2018-nd-cp.pdf` | Nghị định 19/2018/NĐ-CP về cai nghiện ma tuý |
| `nghi-dinh-28-2026-nd-cp.pdf` | Nghị định 28/2026/NĐ-CP về danh mục chất ma tuý |

**Nguồn:** thuvienphapluat.vn, vanban.chinhphu.vn

---

### Task 2 — Crawl Bài Báo

Crawl **6 bài báo** về nghệ sĩ Việt Nam liên quan ma tuý bằng **Crawl4AI**, lưu dạng JSON vào `data/landing/news/`:

| File | Nguồn | Nghệ sĩ |
|------|-------|---------|
| `article_01_vnexpress_ca-si-miu-le...json` | VnExpress | Miu Lê |
| `article_02_tuoitre_rapper-binh-gold...json` | Tuổi Trẻ | Bình Gold |
| `article_03_vnexpress_nguoi-mau-andrea-aybar...json` | VnExpress | An Tây (Andrea Aybar) |
| `article_04_kenh14_ca-si-chi-dan...json` | Kenh14 | Chi Dân + An Tây |
| `article_05_xaydungchinhsach_khoi-to-long-nhat...json` | Xây Dựng Chính Sách | Long Nhật, Sơn Ngọc Minh |
| `article_06_vtv_ca-si-chu-bin...json` | VTV | Chu Bin |

Mỗi file JSON chứa: `url`, `title`, `crawled_at`, `content` (markdown).

---

### Task 3 — Convert Sang Markdown

Dùng **MarkItDown** (Microsoft) để convert toàn bộ PDF/JSON trong `data/landing/` thành Markdown, lưu vào `data/standardized/`:

```bash
python src/task3_convert_markdown.py
```

Output giữ nguyên cấu trúc thư mục con: `standardized/legal/*.md`, `standardized/news/*.md`.

---

### Task 4 — Chunking & Indexing

**Chiến lược chunking:** `RecursiveCharacterTextSplitter`
- `chunk_size = 1200` ký tự — đủ dài để giữ ngữ cảnh điều/khoản và đoạn báo
- `chunk_overlap = 180` ký tự — tránh mất ngữ cảnh tại ranh giới chunk
- Separators: `["\n\n", "\n", ". ", "; ", ", ", " ", ""]`

**Embedding models (2 models song song):**

| Model | Dimension | Đặc điểm |
|-------|-----------|----------|
| `BAAI/bge-m3` (local) | 1024-d | Multilingual, mạnh với tiếng Việt, chạy offline |
| `OpenAI text-embedding-3-small` | 1536-d | Nhanh, dùng cho group project |

**Vector store:** Local `.npy` + `.jsonl` (không cần server), lưu tại `data/indexes/`.

```bash
# Build cả 2 index
python src/task4_chunking_indexing.py

# Chỉ build OpenAI index
python src/task4_chunking_indexing.py --models openai
```

---

### Task 5 — Semantic Search

Module dense retrieval dùng cosine similarity trên local vector store:

```python
def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    # Embed query → cosine similarity với embeddings.npy
    # Return: [{'content': str, 'score': float, 'metadata': dict}]
```

Hỗ trợ cả 2 model index (bge-m3 và openai), tự động chọn theo `.env`.

---

### Task 6 — Lexical Search

Module BM25 dùng **Whoosh** (persistent index) với fallback về **rank-bm25** (in-memory):

```python
def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    # Whoosh BM25 → persistent index tại data/indexes/whoosh/
    # Fallback: rank-bm25 in-memory nếu Whoosh chưa có index
```

**Bonus:** Bổ sung **TF-IDF thuần Python** không phụ thuộc thư viện ngoài, được tích hợp vào group project.

---

### Task 7 — Reranking

Module reranking hỗ trợ 3 phương pháp:

| Phương pháp | Mô tả | Khi nào dùng |
|-------------|-------|-------------|
| **Jina Cross-Encoder API** (`jina-reranker-v2-base-multilingual`) | Cross-encoder nhìn đồng thời query + doc, cho độ chính xác cao nhất | Primary method khi có `JINA_API_KEY` |
| **Lexical Fallback** | Token overlap + original score, không cần API | Khi Jina không khả dụng |
| **MMR** (Maximal Marginal Relevance) | Giảm trùng lặp, tăng diversity bằng Jaccard similarity | Khi cần đa dạng kết quả |
| **RRF** (Reciprocal Rank Fusion) | Gộp nhiều ranked list: `score = Σ 1/(k+rank)`, k=60 | Fusion dense + BM25 ở Task 9 |

---

### Task 8 — PageIndex Vectorless RAG

Dùng **PageIndex** (VectifyAI) — vectorless retrieval không cần embedding, tìm kiếm trực tiếp trên văn bản:

```python
def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    # Fallback khi hybrid search score < threshold
```

Documents được upload lên PageIndex, kết quả lưu cache tại `data/pageindex/`.

---

### Task 9 — Retrieval Pipeline Hoàn Chỉnh

Kết hợp tất cả modules thành pipeline thống nhất với logic fallback:

```
Query
  │
  ├─→ Semantic Search (Task 5)  ──┐
  │                                ├─→ RRF Fusion → Jina Rerank → Results
  ├─→ Lexical BM25 (Task 6)    ──┘
  │
  └─→ Nếu top score < 0.3 → Fallback: PageIndex (Task 8)
```

```python
def retrieve(query: str, top_k: int = 5, score_threshold: float = 0.3) -> list[dict]:
    # 1. semantic_search + lexical_search song song
    # 2. RRF fusion
    # 3. Jina rerank (fallback về lexical nếu không có API key)
    # 4. Nếu score < threshold → PageIndex fallback
```

---

### Task 10 — Generation Có Citation

Sắp xếp lại context (Lost-in-the-Middle), inject vào prompt, gọi GPT-4o-mini, trả về câu trả lời có citation:

**Document Reordering:** chunk quan trọng nhất ở đầu và cuối, ít quan trọng ở giữa → giảm Lost-in-the-Middle.

**Citation format:** `[Nguồn, Năm]` — ví dụ: `[Luật PCMT 2021, 2021]`, `[VnExpress, 2026]`

```python
def generate_with_citation(query: str, context_chunks: list[dict]) -> dict:
    # reorder_for_llm → format_context → inject prompt → GPT-4o-mini
    # temperature=0.1, top_p=0.9
```

---

## Bài Tập Nhóm — RAG Chatbot

### Kiến Trúc Hệ Thống

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

### Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| | | | |
| | | | |
| | | | |
| | | | |

---

## Hướng Dẫn Cài Đặt & Chạy

### Bài Cá Nhân

```bash
# Tạo và kích hoạt môi trường ảo
cd Lab8
python3.12 -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Cài dependencies
pip install -r requirements.txt

# Cấu hình API keys
cp .env.example .env
# Điền OPENAI_API_KEY, JINA_API_KEY, PAGEINDEX_API_KEY vào .env

# Chạy từng task
python src/task3_convert_markdown.py
python src/task4_chunking_indexing.py
python src/task9_retrieval_pipeline.py

# Chạy automated tests
pytest tests/ -v
```

### Bài Nhóm (Group Project)

```bash
cd Lab8/group_project

python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

cp .env.example .env
# Điền OPENAI_API_KEY vào .env

# Một lệnh chạy toàn bộ (Docker + index + Chainlit UI):
.venv/bin/python run.py
```

Mở trình duyệt tại: **http://127.0.0.1:8000**

---

## Chấm Điểm

### Bài Cá Nhân — 50 điểm

| Task | Nội dung | Điểm |
|------|----------|------|
| 1 | Thu thập ≥3 văn bản pháp luật PDF | 3 |
| 2 | Crawl ≥5 bài báo JSON | 3 |
| 3 | Convert markdown thành công | 4 |
| 4 | Chunking + Indexing (vector store có data) | 7 |
| 5 | Semantic search đúng format, sorted | 6 |
| 6 | Lexical BM25 đúng format | 6 |
| 7 | Reranking hoạt động, output re-sorted | 6 |
| 8 | PageIndex query trả về kết quả | 4 |
| 9 | Retrieval pipeline + fallback hoạt động | 7 |
| 10 | Generation có citation + reorder | 4 |
| **Tổng** | | **50** |

### Bài Nhóm — 30 điểm

| Tiêu chí | Điểm |
|----------|------|
| RAG Chatbot demo hoạt động được | 8 |
| Tích hợp pipeline các thành viên | 4 |
| Kiến trúc rõ ràng + README | 3 |
| Chất lượng câu trả lời (có citation, đúng nội dung) | 3 |
| **Evaluation pipeline** (DeepEval) | **12** |
| — Golden dataset ≥15 Q&A pairs | 3 |
| — Chạy eval với ≥4 metrics | 4 |
| — So sánh A/B ≥2 configs + phân tích | 3 |
| — Báo cáo kết quả + phân tích worst performers | 2 |

### Bonus — 20 điểm

| Bonus | Điểm |
|-------|------|
| Giải thích cơ chế lexical search khác BM25 (TF-IDF) | 5 |
| Implement HyDE (Hypothetical Document Embeddings) | 5 |
| Deploy chatbot lên cloud (Hugging Face Spaces / Render) | 5 |
| Demo câu hỏi khó mà LLM không trả lời được (mỗi câu) | 5 |

---

## Lưu Ý

Hãy giữ lại repo này nếu như bạn học **Track 3 Giai Đoạn 2** — chúng ta sẽ phát triển tiếp dự án lên **Knowledge Graph** để khắc phục các câu hỏi hóc búa đòi hỏi multi-hop reasoning.