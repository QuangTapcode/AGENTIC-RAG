# Chunks 800

Đây là bộ chunk phục vụ thí nghiệm RAG với giới hạn tối đa **800 tokens**.

- Tokenizer: `cl100k_base` (`tiktoken==0.14.0`).
- Overlap mục tiêu: 120 tokens, nằm trong khoảng 100–160 tokens.
- Chunk được chia theo heading/section trước, sau đó mới greedy-pack các block Markdown.
- Paragraph, bảng, fenced block và danh sách được giữ như block nguyên tử khi có thể.
- Danh sách dài được tách theo ranh giới item; không cắt giữa hai item. Metadata có `quality_flags` để ghi nhận trường hợp không thể giữ nguyên block.
- `page_start` và `page_end` là `null` vì corpus hiện tại là Markdown, chưa có số trang.

Files:

- `chunks.jsonl`: mỗi dòng là một chunk cùng metadata nguồn.
- `chunk_report.json`: thống kê số chunk, token, overlap, quality flags và lỗi.

## Failure và bài học

- Lần kiểm tra đầu tiên đếm token theo tổng block nên có chunk vượt 1 token do separator Markdown; implementation hiện đếm trên text đã render hoàn chỉnh.
- Danh sách dài được xử lý lại theo ranh giới từng item trước khi split tokenizer, nên không cắt giữa các item trong corpus hiện tại.
- Vì ưu tiên section trước, một số chunk ngắn hơn target là chủ ý; retrieval sẽ được so sánh với bộ 300 tokens ở các bước sau.
