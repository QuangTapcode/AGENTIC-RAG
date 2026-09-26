# Parsed dataset

Các file JSON trong thư mục này là output của Kreuzberg từ 30 tài liệu tại `data/raw/`.

## Ghi chú quan trọng

- Tài liệu nguồn hiện tại là Markdown tiếng Anh được trích xuất từ các trang WHO.
- `source_url` trong mỗi file vẫn trỏ về trang WHO gốc để giữ provenance và phục vụ citation.
- Kreuzberg được cấu hình với `OutputFormat.MARKDOWN` để giữ heading, danh sách và cấu trúc Markdown trước khi chunking.
- Dữ liệu parsed là đầu vào cho Bước 3; output chunking nằm tại `data/chunks_300/` và `data/chunks_800/`.
- Corpus hiện chưa có PDF scan hoặc ảnh nên OCR chưa được kích hoạt.
- Nếu Kreuzberg lỗi, parser giữ lại raw UTF-8 bằng fallback và ghi lỗi vào trường `error`.
- Không nên sửa trực tiếp các file JSON trong thư mục này; hãy sửa dữ liệu nguồn hoặc parser rồi chạy lại.

## Schema chính

- `document_id`, `title`, `topic`: định danh từ `data/manifest.json`.
- `source_url`, `published_date`, `source_language`: provenance.
- `parser`, `parser_version`, `output_format`: thông tin parser.
- `content`: nội dung đã trích xuất ở Markdown.
- `headings`: heading kèm level và line number.
- `tables`: bảng nếu parser phát hiện được.
- `parser_metadata`: metadata do Kreuzberg trả về.
- `quality_flags`, `error`: cảnh báo và lỗi từng tài liệu.

## Kết quả lần chạy hiện tại

- Parser: Kreuzberg `4.10.4`
- Input: 30 Markdown files
- Thành công: 30/30
- Fallback: 0
- Quality flags: 0
- Report: `data/parsed/parse_report.json`

Các lỗi cần theo dõi ở những lần chạy sau:

- mất ký tự tiếng Việt hoặc ký tự thay thế `�`;
- thiếu heading hoặc bảng;
- nội dung rỗng/quá ngắn;
- parser phải dùng fallback;
- metadata hoặc `source_url` không khớp với `data/manifest.json`.

## Chạy lại

```powershell
.\.venv\Scripts\python.exe src\parser\parse_documents.py
```
