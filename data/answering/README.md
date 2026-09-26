# Answer Generation và Citation

Phần 8 nhận query và các chunk đã retrieve/rerank, sau đó tạo câu trả lời grounded:

```text
scope router -> retrieval -> reranking -> answer generator -> citation validator -> response
```

## Grounding contract

- `SYSTEM_PROMPT` yêu cầu LLM chỉ dùng các block `<SOURCE>` được cung cấp.
- Source text được đánh dấu là evidence không đáng tin cậy về mặt instruction; các câu lệnh nằm trong tài liệu không được thực thi.
- Mọi factual claim từ LLM phải có marker `[CITATION:chunk_id]`.
- Validator chỉ chấp nhận `chunk_id` thuộc context hiện tại, sau đó đổi marker thành `[S1]`, `[S2]`… và trả metadata tài liệu, trang, section, URL.
- Nếu WHO payload không có page, citation ghi `page=None`/`page unavailable` và source line; hệ thống không tự bịa số trang.
- Nếu context rỗng, LLM nói `INSUFFICIENT_EVIDENCE`, hoặc citation không hợp lệ, hệ thống từ chối/fallback an toàn.

## LLM provider hook

Không khóa vào nhà cung cấp cụ thể. Inject callable có dạng `llm(system_prompt, user_prompt) -> str` vào `AnswerGenerator(llm=...)`. Khi chưa cấu hình LLM, hệ thống trả evidence excerpts có citation hợp lệ; khi provider lỗi, cũng fallback về excerpts thay vì sinh nội dung không grounded.

## Cảnh báo y tế

Mọi response đều thêm cảnh báo rằng thông tin chỉ mang tính tham khảo và không thay thế chẩn đoán/tư vấn của bác sĩ.

## Test

`tests/test_answer_generator.py` kiểm tra context rỗng, citation giả, citation thiếu, nhiều thuốc, hoạt chất/viết tắt, query gốc và prompt injection trong source text.
