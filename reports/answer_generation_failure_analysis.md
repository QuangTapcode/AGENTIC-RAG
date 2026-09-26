# Answer Generation Failure Analysis

| Failure mode | Detection | Safe behavior | Lesson learned |
|---|---|---|---|
| Context rỗng hoặc không có text | Không tạo được citation record | Refuse với `status=insufficient_evidence` | Không để LLM tự trả lời ngoài corpus |
| LLM trả citation không có trong context | `CitationValidationError` | Discard toàn bộ draft, trả `status=citation_error` | Không tin citation ID do LLM tự tạo |
| LLM trả factual claim không có marker | Không có `[CITATION:chunk_id]` | Discard draft, không hiển thị câu trả lời | Citation là điều kiện bắt buộc, không chỉ là metadata |
| LLM/provider lỗi | Exception được ghi theo type | Dùng evidence excerpts có citation, không hallucinate | Provider outage không được làm mất safety boundary |
| Payload không có page | `page_start/page_end` là `None` | Hiển thị `page unavailable` và source lines | Không bịa page; cần parser PDF để có page thật |
| Prompt injection trong source | Source block được đánh dấu untrusted | System prompt bỏ qua instruction trong source | Retrieved text là evidence, không phải system instruction |
