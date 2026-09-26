# TODO — Agentic RAG Y tế

## 0. Mục tiêu và phạm vi

- [ ] Xác định hệ thống chỉ trả lời câu hỏi về bệnh, triệu chứng, thuốc và hướng dẫn điều trị.
- [ ] Quy định hệ thống không chẩn đoán, không tự kê đơn và không thay thế bác sĩ.
- [ ] Xác định format câu trả lời: kết luận, bằng chứng, citation, mức độ chắc chắn.
- [ ] Viết tiêu chí hoàn thành MVP.

## 1. Thu thập và quản lý dữ liệu

- [x] Thu thập tối thiểu 20 tài liệu y tế đáng tin cậy.
- [x] Mở rộng dataset lên 30 tài liệu WHO, không trùng source URL.
- [x] Ưu tiên WHO, Bộ Y tế, bệnh viện, hướng dẫn điều trị và tờ hướng dẫn thuốc.
- [x] Ghi nhận tổ chức, source URL, ngày xuất bản, ngày thu thập và lưu ý bản quyền/attribution.
- [x] Lưu tài liệu raw vào `data/raw/`.
- [x] Tạo `data/manifest.json` gồm `document_id`, tiêu đề, chủ đề, nguồn, URL và ngày xuất bản.
- [x] Kiểm tra đủ 20 file, không rỗng và mỗi file có document tương ứng trong manifest.
- [ ] Xác nhận quyền tái phân phối trước khi public dataset hoặc dùng thương mại.

## 2. Parsing dữ liệu

- [x] Cài đặt và thử nghiệm Kreuzberg `4.10.4` trong `.venv`.
- [x] Parse 30 raw Markdown documents sang JSON có cấu trúc.
- [ ] Parse PDF, DOCX và HTML bổ sung khi corpus có các định dạng này.
- [ ] Bật OCR cho PDF scan nếu cần.
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

- [ ] Chọn model embedding hỗ trợ cross-lingual Việt–Anh.
- [ ] Chạy Qdrant local bằng Docker.
- [ ] Tạo collection multilingual cho chunk 300 và chunk 800.
- [ ] Embed query gốc bằng multilingual embedding.
- [ ] Lưu dense vector cho semantic search xuyên ngôn ngữ.
- [ ] Lưu sparse vector/BM25 cho keyword search.
- [ ] Lưu toàn bộ metadata chunk trong payload.
- [ ] Kiểm tra số lượng vector và metadata sau ingestion.

## 5. Scope Router và Agent Tools

- [ ] Tạo classifier xác định query có thuộc chủ đề Y tế hay không.
- [ ] Kết hợp rule-based keywords với LLM/classifier.
- [ ] Tạo tool `reject_out_of_scope(query)`.
- [ ] Nếu query ngoài phạm vi, gọi tool và kết thúc pipeline ngay.
- [ ] Nếu confidence thấp, trả lời yêu cầu người dùng làm rõ hoặc thông báo không xác định được phạm vi.
- [ ] Tạo log cho các trường hợp router phân loại sai.

## 6. Hybrid Retrieval

- [ ] Detect ngôn ngữ query.
- [ ] Tạo `original_query`, `normalized_query` và `translated_query_en`.
- [ ] Implement multilingual dense retrieval bằng `original_query`.
- [ ] Implement BM25/sparse retrieval bằng `translated_query_en` cho corpus tiếng Anh.
- [ ] Lấy top-k riêng cho dense và BM25.
- [ ] Kết hợp kết quả bằng Reciprocal Rank Fusion (RRF).
- [ ] So sánh dense multilingual-only, BM25-only và multilingual hybrid.
- [ ] Lưu lại score, rank và chunk ID của từng retriever.
- [ ] Tối ưu `top_k_dense`, `top_k_bm25` và `top_k_fused`.

## 7. Reranking

- [ ] Chọn cross-lingual cross-encoder/reranker hỗ trợ Việt–Anh.
- [ ] Rerank khoảng 20 candidate từ hybrid retrieval.
- [ ] Chọn top 5–8 chunk đưa vào LLM.
- [ ] So sánh hybrid không reranking và hybrid có reranking.
- [ ] Ghi nhận latency và điểm trước/sau reranking.
- [ ] Xử lý trường hợp reranker chạy quá chậm hoặc xếp sai chunk.
- [ ] Kiểm tra lỗi dịch query, tên thuốc, hoạt chất và viết tắt.

## 8. Answer Generation và Citation

- [ ] Viết system prompt bắt buộc LLM chỉ dùng context được cung cấp.
- [ ] Yêu cầu LLM từ chối khi không đủ bằng chứng.
- [ ] Hiển thị citation gồm tên tài liệu, trang và section.
- [ ] Kiểm tra citation có tồn tại trong context được retrieve.
- [ ] Ngăn citation giả hoặc citation sai trang.
- [ ] Thêm cảnh báo: thông tin chỉ mang tính tham khảo, không thay thế bác sĩ.
- [ ] Test các câu hỏi có nhiều thuốc, liều lượng và chống chỉ định.

## 9. Evaluation Dataset

- [ ] Tạo tối thiểu 30 câu hỏi.
- [ ] Tạo 15 câu tiếng Việt có thể trả lời từ corpus tiếng Anh.
- [ ] Tạo 5 câu tiếng Anh có thể trả lời từ corpus tiếng Anh.
- [ ] Tạo 5 câu hỏi thiếu thông tin trong corpus.
- [ ] Tạo 5 câu hỏi ngoài phạm vi.
- [ ] Đánh giá riêng theo `detected_language`.
- [ ] Ghi expected answer, expected source và expected page.
- [ ] Lưu dataset tại `eval/questions.jsonl`.
- [ ] Lưu generated answer, retrieved chunks, result và failure type.

## 10. Automated Evaluation

- [ ] Đo router accuracy, precision, recall và F1.
- [ ] Đo retrieval Recall@k và Precision@k.
- [ ] Đo MRR và NDCG.
- [ ] Đo answer correctness và faithfulness.
- [ ] Đo context relevance.
- [ ] Đo citation precision và citation recall.
- [ ] Đo tỷ lệ từ chối đúng với query ngoài phạm vi.
- [ ] Tạo `reports/evaluation_report.md`.
- [ ] So sánh các cấu hình:
  - [ ] Dense + chunk 300.
  - [ ] Dense + chunk 800.
  - [ ] Hybrid + chunk 300.
  - [ ] Hybrid + chunk 800.
  - [ ] Hybrid + reranking.

## 11. Cost và Performance Benchmarking

- [ ] Đo thời gian parsing.
- [ ] Đo thời gian indexing và embedding.
- [ ] Đo latency retrieval, reranking và generation.
- [ ] Đo p50 và p95 latency.
- [ ] Ghi nhận số token input/output của LLM.
- [ ] Ghi nhận số lần gọi embedding và reranker.
- [ ] Tính cost trung bình cho một query.
- [ ] Đo storage của Qdrant.
- [ ] Chạy mỗi cấu hình tối thiểu 3 lần trên cùng evaluation dataset.
- [ ] Tạo `reports/cost_benchmark.md`.

## 12. Failure Analysis

- [ ] Ghi nhận lỗi parsing/OCR.
- [ ] Ghi nhận lỗi chunking.
- [ ] Ghi nhận lỗi retrieval.
- [ ] Ghi nhận lỗi reranking.
- [ ] Ghi nhận hallucination.
- [ ] Ghi nhận citation sai.
- [ ] Ghi nhận router phân loại sai.
- [ ] Với mỗi lỗi, ghi query, expected, actual, root cause, fix và lesson learned.
- [ ] Tạo `reports/failure_analysis.md`.

## 13. Demo và hồ sơ submission

- [ ] Tạo giao diện hoặc CLI demo.
- [ ] Demo một câu hỏi trong phạm vi có citation.
- [ ] Demo một câu hỏi ngoài phạm vi bị từ chối.
- [ ] Demo một câu hỏi không có đủ evidence.
- [ ] Viết `README.md` gồm problem, dataset, architecture, setup và limitations.
- [ ] Viết `AI_WORKLOG.md` gồm công cụ AI, prompt, lỗi AI và cách sửa.
- [ ] Tạo `docker-compose.yml` cho Qdrant.
- [ ] Tạo `.env.example`.
- [ ] Quay demo video tối đa 5 phút.
- [ ] Kiểm tra repository có thể chạy lại từ README.

## 14. Thứ tự triển khai ưu tiên

- [ ] Ngày 1: Dataset, parser, Qdrant và baseline dense retrieval.
- [ ] Ngày 2: Scope router, chunk 300/800 và hybrid BM25.
- [ ] Ngày 3: Reranking, citation và 30 câu evaluation.
- [ ] Ngày 4: Benchmark, failure analysis, README, AI_WORKLOG và demo.
- [ ] Ngày 5–7: Cải thiện UI, caching, prompt, query rewriting và biểu đồ báo cáo nếu còn thời gian.
