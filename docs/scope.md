# Phạm vi và mục tiêu — Agentic RAG Y tế

## 1. Mục tiêu

Hệ thống là một trợ lý trả lời câu hỏi tham khảo về y tế, có kiểm chứng bằng chứng
qua Retrieval-Augmented Generation (RAG). Người dùng đặt câu hỏi tiếng Việt hoặc
tiếng Anh về **bệnh**, **triệu chứng**, **thuốc/hoạt chất** và **hướng dẫn điều
trị/phòng ngừa**; hệ thống truy hồi từ corpus WHO đã được ingest và sinh câu trả
lời có citation.

## 2. Phạm vi trả lời

Hệ thống **chỉ** trả lời khi câu hỏi:

- thuộc một trong bốn chủ đề trên (bệnh, triệu chứng, thuốc, hướng dẫn điều
  trị/phòng ngừa);
- có đủ bằng chứng trong context được retrieve.

Với mọi câu hỏi khác, hệ thống phải:

- gọi tool `reject_out_of_scope(query)` nếu classifier xác định ngoài phạm vi;
- yêu cầu người dùng làm rõ nếu confidence thấp (`decision=clarify`);
- trả về `INSUFFICIENT_EVIDENCE` (rồi thông báo bằng ngôn ngữ người dùng) nếu
  không đủ evidence.

## 3. Những gì hệ thống KHÔNG làm

- **Không chẩn đoán bệnh** cho một người cụ thể.
- **Không kê đơn** hoặc gợi ý thay đổi liều lượng.
- **Không thay thế bác sĩ**; mọi câu trả lời phải kèm cảnh báo tham khảo (đã hiện
  thực trong [answer_generator.py](../src/answering/answer_generator.py) —
  `MEDICAL_WARNING_VI` / `MEDICAL_WARNING_EN`).
- **Không sinh nội dung ngoài corpus**; nếu LLM cite chunk không có trong
  context sẽ bị loại bởi `CitationValidationError`.
- **Không dịch/rewrite tên thuốc**, hoạt chất, liều lượng, chống chỉ định.

## 4. Format câu trả lời

Mỗi câu trả lời trả về theo cấu trúc `AnswerResult`:

| Trường | Ý nghĩa |
|---|---|
| `status` | `answered`, `answered_fallback`, `insufficient_evidence`, `citation_error` |
| `answer` | **Kết luận** ngôn ngữ người dùng + citation marker `[S1]`, `[S2]`... + cảnh báo tham khảo |
| `citations` | Danh sách nguồn: `title`, `document_id`, `chunk_id`, `page_label`, `section`, `source_url`, dải `source_line_start/end` |
| `warning` | Cảnh báo cố định: thông tin chỉ tham khảo, không thay thế bác sĩ |
| `used_llm` | LLM đã được gọi hay dùng fallback evidence-excerpt |
| `context_count` | Số chunk được đưa vào prompt |
| `failure_reason` | Điền khi status không phải `answered` |
| `notes` | Trace mềm cho debug |

Với corpus WHO hiện tại không có page number thật, `page_label` = `page unavailable`
và citation kèm `source_line_start/end` (đã kiểm chứng ở
[answer_generator.py:73-78](../src/answering/answer_generator.py#L73-L78)).

## 5. Mức độ chắc chắn (confidence signal)

Hiện tại prototype không xuất ra một số confidence duy nhất cho câu trả lời — thay
vào đó nhiều tín hiệu độc lập được ghi vào trace:

- `ScopeClassification.confidence` cho quyết định routing.
- `RetrievalHit.score` (dense & BM25) và `FusedHit.rrf_score` cho retrieval.
- Rerank score cho từng chunk trong `RerankedChunk`.
- `context_count` và số citation dùng thực tế.

Người tích hợp có thể tổng hợp một confidence gauge từ các tín hiệu này; mặc định
prototype ưu tiên **an toàn hơn một con số duy nhất** để tránh over-confidence.

## 6. Tiêu chí hoàn thành MVP

MVP được coi là hoàn thành khi:

1. **Dataset**: ≥ 30 tài liệu WHO parse thành công, manifest đầy đủ metadata,
   quality flags được ghi. (đã đạt — xem
   [parse_report.json](../data/parsed/parse_report.json))
2. **Ingest**: cả `medical_chunks_300` và `medical_chunks_800` được ingest với
   dense + sparse vector, verification pass.
3. **Router**: F1 ≥ 0.85 trên eval set (25 in-scope, 5 out-of-scope), không có
   query ngoài phạm vi nào bị đi tiếp vào retrieval mà không bị chặn.
4. **Retrieval**: Hybrid Recall@5 ≥ 0.80 trên 20 câu answerable của eval set;
   MRR ≥ 0.60.
5. **Reranking**: Hybrid+rerank không làm Hit@5 giảm; nếu giảm phải ghi rõ.
6. **Answer**: 100 % câu trả lời "answered" có citation trong context; 0 citation
   giả; 0 hallucinated dose/tên thuốc; 5/5 câu insufficient được refuse.
7. **Demo**: CLI/UI chạy 3 kịch bản (in-scope answered, out-of-scope refused,
   insufficient-evidence refused).
8. **Reports**: `evaluation_report.md`, `cost_benchmark.md`, `failure_analysis.md`
   được sinh và checked-in.
9. **Reproducibility**: từ README có thể build lại: docker-compose up qdrant,
   parse → chunk → embed → eval.

Các tiêu chí trên tương ứng 1-1 với check-list trong [TODO.md](../TODO.md).
