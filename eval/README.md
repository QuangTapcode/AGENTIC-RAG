# Evaluation Dataset

`questions.jsonl` là bộ evaluation cho Phần 9, gồm 30 câu hỏi:

| Nhóm | Số lượng | Mục đích |
|---|---:|---|
| `in_scope_answerable` | 20 | 15 câu tiếng Việt và 5 câu tiếng Anh có evidence trong corpus WHO |
| `in_scope_insufficient_corpus` | 5 | Câu hỏi y tế nhưng corpus 30 tài liệu hiện tại không đủ evidence; hệ thống phải từ chối |
| `out_of_scope` | 5 | Câu hỏi không thuộc y tế; router phải kết thúc bằng `reject_out_of_scope` |

Mỗi record có `question`, `language`, `detected_language`, `category`, `expected_answer`, `expected_page`, `expected_source`, `must_refuse` và các field output ban đầu (`generated_answer`, `retrieved_chunks`, `result`, `failure_type`).

## Expected source

Các câu answerable trỏ tới `document_id`, `chunk_id`, `section`, source URL và source line range của `data/chunks_300/chunks.jsonl`. Raw WHO Markdown hiện không có số trang nên `expected_page` là `null`; đây là giá trị đã biết, không phải page bị đoán.

## Validate

```powershell
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe src\evaluation\validate_questions.py
```

Validator kiểm tra đúng 30 record, category/language coverage, ID duy nhất, schema output, chunk anchor, section, URL, source lines và page expectation. Chỉ sau khi chạy evaluation, các field output mới được điền; dataset gốc giữ các field đó rỗng để tái lập benchmark.
