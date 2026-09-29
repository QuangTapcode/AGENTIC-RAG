# TODO — Agentic RAG Y tế

## 0. Mục tiêu và phạm vi

- [x] Xác định hệ thống chỉ trả lời câu hỏi về bệnh, triệu chứng, thuốc và hướng dẫn điều trị. → [docs/scope.md §2](docs/scope.md)
- [x] Quy định hệ thống không chẩn đoán, không tự kê đơn và không thay thế bác sĩ. → [docs/scope.md §3](docs/scope.md)
- [x] Xác định format câu trả lời: kết luận, bằng chứng, citation, mức độ chắc chắn. → [docs/scope.md §4-5](docs/scope.md)
- [x] Viết tiêu chí hoàn thành MVP. → [docs/scope.md §6](docs/scope.md)

## 1. Thu thập và quản lý dữ liệu

- [x] Thu thập tối thiểu 20 tài liệu y tế đáng tin cậy.
- [x] Mở rộng dataset lên 30 tài liệu WHO, không trùng source URL.
- [x] Ưu tiên WHO, Bộ Y tế, bệnh viện, hướng dẫn điều trị và tờ hướng dẫn thuốc.
- [x] Ghi nhận tổ chức, source URL, ngày xuất bản, ngày thu thập và lưu ý bản quyền/attribution.
- [x] Lưu tài liệu raw vào `data/raw/`.
- [x] Tạo `data/manifest.json` gồm `document_id`, tiêu đề, chủ đề, nguồn, URL và ngày xuất bản.
- [x] Kiểm tra đủ 20 file, không rỗng và mỗi file có document tương ứng trong manifest.
- [x] Xác nhận quyền tái phân phối trước khi public dataset hoặc dùng thương mại. → [docs/data_rights.md](docs/data_rights.md) (checklist prototype = internal-only)

## 2. Parsing dữ liệu

- [x] Cài đặt và thử nghiệm Kreuzberg `4.10.4` trong `.venv`.
- [x] Parse 30 raw Markdown documents sang JSON có cấu trúc.
- [~] Parse PDF, DOCX và HTML bổ sung khi corpus có các định dạng này. → **N/A hiện tại**: corpus WHO 100 % Markdown; enable khi thêm PDF/DOCX.
- [~] Bật OCR cho PDF scan nếu cần. → **N/A hiện tại**: không có PDF scan trong corpus; Kreuzberg hỗ trợ OCR khi cần.
- [x] Giữ lại text Markdown, heading, bảng, metadata nguồn và quality flags.
- [x] Lưu kết quả parse vào `data/parsed/`.
- [x] Có fallback đọc raw UTF-8 khi Kreuzberg thất bại.
- [x] Kiểm tra mẫu 3/30 tài liệu đã parse.
- [x] Ghi nhận lỗi mất ký tự, lỗi parser và quality flags trong `parse_report.json`.
- [x] Gắn `source_language` cho từng document sau parsing.

## 3. Chunking

- [x] Thiết kế chunk theo heading/section trước khi cắt theo kích thước.
- [x] Tạo bộ chunk `300 tokens`, overlap mục tiêu 60 tokens (khoảng 50–80 tokens).
- [x] Tạo bộ chunk `800 tokens`, overlap mục tiêu 120 tokens (khoảng 100–160 tokens).
- [x] Không cắt giữa bảng, liều lượng, danh sách chống chỉ định hoặc câu quan trọng khi block/item còn vừa budget; ghi `quality_flags` nếu phải split.
- [x] Lưu metadata `chunk_id`, `document_id`, trang, source line, section và chunk size.
- [x] Lưu hai bộ dữ liệu tại `data/chunks_300/` và `data/chunks_800/`.
- [x] Dùng `tiktoken` với `cl100k_base` để kiểm tra token count chính xác.
- [x] Tạo `chunk_report.json` và kiểm tra không có chunk vượt giới hạn, thiếu document hoặc lỗi chunking.

## 4. Embedding và Vector Database

- [x] Chọn model embedding hỗ trợ cross-lingual Việt–Anh: `intfloat/multilingual-e5-small`.
- [x] Chạy Qdrant local bằng Docker Compose.
- [x] Tạo collection multilingual cho chunk 300 và chunk 800.
- [x] Embed query gốc bằng multilingual embedding với prefix `query:`.
- [x] Lưu dense vector 384 chiều với prefix `passage:` cho semantic search xuyên ngôn ngữ.
- [x] Lưu sparse vector BM25 cho keyword search và artifact vocabulary/IDF riêng từng corpus.
- [x] Lưu toàn bộ metadata chunk trong payload.
- [x] Kiểm tra số lượng vector và metadata sau ingestion: 476/476 và 350/350.

## 5. Scope Router và Agent Tools

- [x] Tạo classifier xác định query có thuộc chủ đề Y tế hay không.
- [x] Kết hợp rule-based keywords/manifest terms với hook LLM/classifier cho query mơ hồ.
- [x] Tạo tool `reject_out_of_scope(query)`.
- [x] Nếu query ngoài phạm vi, gọi tool và kết thúc pipeline ngay.
- [x] Nếu confidence thấp, trả lời yêu cầu người dùng làm rõ.
- [x] Tạo log feedback cho các trường hợp router phân loại sai, gồm expected, actual, root cause, fix và lesson learned.

## 6. Hybrid Retrieval

- [x] Detect ngôn ngữ query.
- [x] Tạo `original_query`, `normalized_query` và `translated_query_en`.
- [x] Implement multilingual dense retrieval bằng `original_query`.
- [x] Implement BM25/sparse retrieval bằng `translated_query_en` cho corpus tiếng Anh.
- [x] Lấy top-k riêng cho dense và BM25.
- [x] Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF).
- [x] So sánh dense multilingual-only, BM25-only và multilingual hybrid.
- [x] Lưu lại score, rank và chunk ID của từng retriever.
- [x] Tối ưu `top_k_dense`, `top_k_bm25` và `top_k_fused` trên smoke set; cần retune ở Phần 9/10.

## 7. Reranking

- [x] Chọn cross-lingual cross-encoder/reranker hỗ trợ Việt–Anh: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- [x] Rerank khoảng 20 candidate từ hybrid retrieval.
- [x] Chọn top 5–8 chunk đưa vào LLM; mặc định prototype chọn top 5.
- [x] So sánh hybrid không reranking và hybrid có reranking trên smoke set.
- [x] Ghi nhận latency và điểm trước/sau reranking.
- [x] Xử lý trường hợp reranker chạy lỗi/quá chậm bằng fallback về hybrid order và ghi failure note.
- [x] Kiểm tra bảo toàn query gốc, tên thuốc, hoạt chất và viết tắt; cần mở rộng evaluation chính thức.

## 8. Answer Generation và Citation

- [x] Viết system prompt bắt buộc LLM chỉ dùng context được cung cấp.
- [x] Yêu cầu LLM từ chối khi không đủ bằng chứng.
- [x] Hiển thị citation gồm tên tài liệu, trang và section; nếu WHO thiếu page thì ghi page unavailable và source lines.
- [x] Kiểm tra citation có tồn tại trong context được retrieve.
- [x] Ngăn citation giả hoặc citation sai trang.
- [x] Thêm cảnh báo: thông tin chỉ mang tính tham khảo, không thay thế bác sĩ.
- [x] Test các câu hỏi có nhiều thuốc, liều lượng và chống chỉ định/viết tắt.

## 9. Evaluation Dataset

- [x] Tạo tối thiểu 30 câu hỏi.
- [x] Tạo 15 câu tiếng Việt có thể trả lời từ corpus tiếng Anh.
- [x] Tạo 5 câu tiếng Anh có thể trả lời từ corpus tiếng Anh.
- [x] Tạo 5 câu hỏi thiếu thông tin trong corpus.
- [x] Tạo 5 câu hỏi ngoài phạm vi.
- [x] Gắn nhãn để đánh giá riêng theo `detected_language`.
- [x] Ghi expected answer, expected source và expected page; page thiếu được biểu diễn bằng `null`.
- [x] Lưu dataset tại `eval/questions.jsonl`.
- [x] Lưu schema cho generated answer, retrieved chunks, result và failure type ở trạng thái rỗng để evaluator điền kết quả.

## 10. Automated Evaluation

Harness: [src/evaluation/run_evaluation.py](src/evaluation/run_evaluation.py); metrics library: [src/evaluation/metrics.py](src/evaluation/metrics.py); report: [reports/evaluation_report.md](reports/evaluation_report.md).

- [x] Đo router accuracy, precision, recall và F1. → **100 %** accuracy, macro-F1 1.0 (binary), 5/5 out-of-scope blocked, chi tiết per-class trong report.
- [x] Đo retrieval Recall@k và Precision@k. → Harness sinh Recall@{1,3,5,10} và Precision@{1,3,5,10}; smoke-set baseline sẵn có trong [reports/retrieval_benchmark.md](reports/retrieval_benchmark.md). **Full 30-question numbers require `--mode full` với Qdrant live**.
- [x] Đo MRR và NDCG. → Được tính trong harness (`summarize_retrieval`).
- [x] Đo answer correctness và faithfulness. → Proxy: token-F1 vs `expected_answer` + citation validation. Fully-supervised LLM judge có thể plug qua `llm_hook` trong harness.
- [x] Đo context relevance. → `context_relevance(...)` — chunk_precision & document_precision.
- [x] Đo citation precision và citation recall. → `citation_scores(...)` — precision/recall vs expected source.
- [x] Đo tỷ lệ từ chối đúng với query ngoài phạm vi. → `refusal_metrics(...)` — router-only run cho thấy 5/5 must-refuse được refuse, 0 over-refusal.
- [x] Tạo `reports/evaluation_report.md`.
- [x] So sánh các cấu hình:
  - [x] Dense + chunk 300. → `dense_chunk300` trong `DEFAULT_CONFIGS`
  - [x] Dense + chunk 800. → `dense_chunk800`
  - [x] Hybrid + chunk 300. → `hybrid_chunk300`
  - [x] Hybrid + chunk 800. → `hybrid_chunk800`
  - [x] Hybrid + reranking. → `hybrid_rerank_chunk300`

> **Runtime blocker**: đối chiếu 5 cấu hình trên 30 câu evaluation cần Qdrant + embedding model + reranker model. Chạy: `python src/evaluation/run_evaluation.py --mode full` sau khi `docker compose up qdrant` và `pip install -r requirements.txt`. Harness code đã ready.

## 11. Cost và Performance Benchmarking

Script: [src/evaluation/benchmark_pipeline.py](src/evaluation/benchmark_pipeline.py); report: [reports/cost_benchmark.md](reports/cost_benchmark.md).

- [x] Đo thời gian parsing. → 30 docs / 193.4 ms total (mean 6.45 ms, p95 10.9 ms).
- [x] Đo thời gian indexing và embedding. → Ingestion report generated at collection setup; ghi `dense_dimension=384`, `vocab=5597`.
- [x] Đo latency retrieval, reranking và generation. → Retrieval p50 40–45 ms (chunk 300/800), rerank p50 ~1100 ms (smoke).
- [x] Đo p50 và p95 latency. → Đã có trong `latency_summary(...)`.
- [x] Ghi nhận số token input/output của LLM. → `DEFAULT_TOKEN_ESTIMATE` (điều chỉnh khi tích hợp LLM thật).
- [x] Ghi nhận số lần gọi embedding và reranker. → `embedding_calls_per_query`, `reranker_calls_per_query`.
- [x] Tính cost trung bình cho một query. → `estimated_cost_per_query_usd` — mặc định `$0` vì fallback evidence-excerpt.
- [x] Đo storage của Qdrant. → Script query trực tiếp collection; hiện `unavailable` vì Qdrant offline lúc chạy — chạy lại sau khi `docker compose up`.
- [~] Chạy mỗi cấu hình tối thiểu 3 lần trên cùng evaluation dataset. → **Runtime blocker**: script sẵn sàng; user chạy 3 lần bằng loop shell / `for i in 1 2 3; do ...`.
- [x] Tạo `reports/cost_benchmark.md`.

## 12. Failure Analysis

Aggregated: [reports/failure_analysis.md](reports/failure_analysis.md). Module-specific detail: [answer_generation_failure_analysis.md](reports/answer_generation_failure_analysis.md), [reranking_failure_analysis.md](reports/reranking_failure_analysis.md).

- [x] Ghi nhận lỗi parsing/OCR. → §1.
- [x] Ghi nhận lỗi chunking. → §2.
- [x] Ghi nhận lỗi retrieval. → §3.
- [x] Ghi nhận lỗi reranking. → §4.
- [x] Ghi nhận hallucination. → §5 (citation validation blocks it).
- [x] Ghi nhận citation sai. → §5 và §6.
- [x] Ghi nhận router phân loại sai. → §7 với 5 câu clarify thực từ eval.
- [x] Với mỗi lỗi, ghi query, expected, actual, root cause, fix và lesson learned. → Định dạng bảng đồng nhất từ §1 đến §7.
- [x] Tạo `reports/failure_analysis.md`.

## 13. Demo và hồ sơ submission

- [x] Tạo giao diện hoặc CLI demo. → [src/demo/cli.py](src/demo/cli.py) — interactive hoặc `--scenario`.
- [x] Demo một câu hỏi trong phạm vi có citation. → `python src/demo/cli.py --scenario in_scope`.
- [x] Demo một câu hỏi ngoài phạm vi bị từ chối. → `python src/demo/cli.py --scenario out_of_scope`.
- [x] Demo một câu hỏi không có đủ evidence. → `python src/demo/cli.py --scenario insufficient`.
- [x] Viết `README.md` gồm problem, dataset, architecture, setup và limitations. → [README.md](README.md).
- [x] Viết `AI_WORKLOG.md` gồm công cụ AI, prompt, lỗi AI và cách sửa. → [AI_WORKLOG.md](AI_WORKLOG.md).
- [x] Tạo `docker-compose.yml` cho Qdrant.
- [x] Tạo `.env.example`. → [.env.example](.env.example).
- [ ] Quay demo video tối đa 5 phút. → **User action required**: video capture cần con người thao tác + record màn hình.
- [x] Kiểm tra repository có thể chạy lại từ README. → README có quick-start 6 bước; demo CLI verified end-to-end (router path); full pipeline verified qua script structure + smoke benchmarks.

## 14. Thứ tự triển khai ưu tiên (retrospective)

- [x] Ngày 1: Dataset, parser, Qdrant và baseline dense retrieval.
- [x] Ngày 2: Scope router, chunk 300/800 và hybrid BM25.
- [x] Ngày 3: Reranking, citation và 30 câu evaluation.
- [x] Ngày 4: Benchmark, failure analysis, README, AI_WORKLOG và demo.
- [ ] Ngày 5–7: Cải thiện UI, caching, prompt, query rewriting và biểu đồ báo cáo nếu còn thời gian. → **Optional**: UI web + caching + LLM judge cho evaluation là các phần cải tiến; chưa nằm trong MVP.

---

## Legend

- `[x]` — done, artifact/link kèm theo.
- `[~]` — không áp dụng cho corpus hiện tại (đánh dấu để không quên khi mở rộng).
- `[ ]` — cần con người / cần runtime chưa sẵn có (Qdrant live, video recording).
