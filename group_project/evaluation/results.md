# RAG Evaluation Results

Báo cáo đánh giá chất lượng RAG pipeline cho dự án chatbot pháp luật ma túy và tin tức nghệ sĩ liên quan.

---

## Framework sử dụng

**DeepEval** (v2.x) — framework đánh giá RAG có nhiều metric built-in, hỗ trợ async evaluation và tích hợp tốt với pytest.

- **Mô hình đánh giá (judge)**: `gpt-4o-mini` (custom wrapper với `json_object` format để tránh timeout)
- **Golden dataset**: 15 cặp Q&A được tuyển chọn thủ công, bao gồm 9 câu pháp luật và 6 câu tin tức nghệ sĩ
- **Tổng số lần chạy**: 2 configs × 15 câu = **30 lượt evaluation**

---

## Overall Scores

| Metric | Config A (hybrid + rerank) | Config B (dense-only) | Δ |
| :--- | :---: | :---: | :---: |
| **Faithfulness** | **0.93** | 0.88 | +0.05 |
| **Answer Relevance** | **0.89** | 0.83 | +0.06 |
| **Context Recall** | **0.96** | 0.90 | +0.06 |
| **Context Precision** | **0.93** | 0.94 | −0.01 |
| **Average** | **0.93** | 0.89 | +0.04 |

---

## A/B Comparison Analysis

### Config A — Hybrid Search + Reranking
- **Retrieval**: Dense (Weaviate `text-embedding-3-small`) + Lexical (Whoosh BM25) fused bằng Reciprocal Rank Fusion (RRF α = 0.5)
- **Reranking**: `BAAI/bge-reranker-v2-m3` CrossEncoder local (top-5 → top-3)
- **Điểm nổi bật**: Faithfulness và Answer Relevance cao nhất nhờ reranker sàng lọc noise; Context Recall đạt 0.96

### Config B — Dense-only (không Reranking)
- **Retrieval**: Chỉ dùng dense search (Weaviate), trả về top-5 trực tiếp không qua rerank
- **Điểm nổi bật**: Context Precision nhỉnh hơn nhẹ (0.94 so với 0.93) do ít chunk hơn được lọc; nhưng Answer Relevance thấp hơn đáng kể (0.83) vì LLM nhận được nhiều chunk nhiễu hơn

### Kết luận

**Config A (Hybrid + Reranking) vượt trội hơn trên 3/4 metric quan trọng.** Reranker đóng vai trò then chốt: nó đẩy các chunk thực sự liên quan lên đầu, giúp LLM trả lời chính xác và trung thực hơn. Config B tuy tốc độ nhanh hơn (~30%), nhưng có nguy cơ "hallucinate" cao hơn do nhận context nhiễu. Với bài toán pháp luật — nơi độ chính xác là ưu tiên số 1 — Config A là lựa chọn phù hợp.

---

## Worst Performers (Bottom 3)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
| :---: | :--- | :---: | :---: | :---: | :--- | :--- |
| 1 | Luật Phòng chống ma tuý 2021 quy định những hình thức cai nghiện nào? | 1.00 | 0.00 | 1.00 | **Generation** | LLM từ chối trả lời dù context đã có đủ thông tin — do system prompt quá nghiêm ngặt với điều kiện UNVERIFIABLE. Chunk về Chương V (cai nghiện) tồn tại nhưng nằm ở dạng tổng quát; LLM không nhận dạng được sự tương ứng với câu hỏi cụ thể. |
| 2 | Cơ quan chuyên trách phòng, chống tội phạm về ma túy gồm những cơ quan nào? | 0.50 | 0.80 | 1.00 | **Generation** | Faithfulness = 0.50 cho thấy LLM đã tổng hợp một phần thông tin ngoài context (thêm "Viện Kiểm sát" hoặc chi tiết không có trong chunk). Context recall hoàn hảo, nhưng generation bị "drift" khỏi nguồn. |
| 3 | Ca sĩ Chi Dân bị truy tố về tội danh gì? | 0.50 | 1.00 | 1.00 | **Generation** | Faithfulness = 0.50 vì câu trả lời bao gồm thông tin về "hùn tiền mua ma túy" và địa điểm "quận Tân Bình" — các chi tiết này tuy đúng nhưng không được trích xuất trực tiếp từ context mà LLM suy diễn. |

---

## Recommendations

### Cải tiến 1 — Tăng cường System Prompt (đã triển khai)
**Action:** Viết lại system prompt theo tiếng Việt, thay nguyên tắc "từ chối nếu context không hoàn toàn rõ ràng" thành "tổng hợp từ context và chỉ từ chối khi context thực sự trống". Thêm rule ưu tiên synthesize.

**Expected impact:** Giải quyết hoàn toàn Worst Performer #1 (cai nghiện: Relevance từ 0.00 → ~0.80+). Answer Relevance tổng thể tăng ~0.05–0.10.

---

### Cải tiến 2 — HyDE (Hypothetical Document Embeddings) (đã triển khai — Bonus)
**Action:** Trước khi embed câu hỏi, dùng GPT-4o-mini sinh ra một đoạn văn giả định trả lời câu hỏi đó, rồi embed đoạn văn này để search. Tích hợp vào `pipeline.py` với tuỳ chọn `use_hyde=True` qua `cl.ChatSettings`.

**Expected impact:** Cải thiện Context Recall cho các câu hỏi mang tính tổng quát (ví dụ: "cai nghiện nào") vì HyDE đưa embedding gần hơn với văn bản pháp luật thực tế hơn là câu hỏi thuần tuý. Dự kiến Context Recall tăng 0.03–0.05.

---

### Cải tiến 3 — TF-IDF Lexical Search (đã triển khai — Bonus)
**Action:** Bổ sung thuật toán TF-IDF thuần Python song song với Whoosh BM25. Cho phép người dùng chọn lexical method qua UI (`cl.ChatSettings`). Tích hợp trong `vectorless_bm25.py` với hàm `tfidf_search`.

**Expected impact:** TF-IDF cho kết quả tốt hơn BM25 trên corpus tiếng Việt ngắn (< 3.000 chunks) do không phụ thuộc vào IDF approximation của BM25. Dự kiến Context Precision tăng 0.02–0.04 cho các câu hỏi pháp luật có từ khoá đặc thù (tên điều luật, số văn bản).
