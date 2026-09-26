# Dataset provenance

Dataset hiện tại gồm 30 WHO fact sheets về bệnh và sức khỏe.

- Catalog nguồn: <https://www.who.int/news-room/fact-sheets>
- Tổ chức: World Health Organization (WHO)
- Ngôn ngữ tài liệu nguồn: tiếng Anh (`en`)
- Ngôn ngữ query hỗ trợ: tiếng Việt (`vi`) và tiếng Anh (`en`)
- Chiến lược truy hồi: multilingual dense retrieval + BM25 + RRF + cross-lingual reranking
- Ngày thu thập: 2026-09-27
- File raw: `data/raw/*.md`
- File parsed: `data/parsed/*.json`
- Metadata và source URL: `data/manifest.json`

## Cách xử lý query tiếng Việt

Hệ thống không dịch toàn bộ corpus sang tiếng Việt ở giai đoạn đầu. Mỗi query được lưu thành hai phiên bản:

1. `original_query`: câu hỏi gốc của người dùng, dùng cho multilingual dense retrieval và reranking.
2. `translated_query_en`: bản dịch sang tiếng Anh, dùng cho BM25 vì corpus hiện tại là tiếng Anh.

Sau khi retrieve và rerank, LLM trả lời bằng ngôn ngữ của query, mặc định là tiếng Việt nếu người dùng hỏi tiếng Việt. Citation luôn trỏ về tài liệu WHO gốc.

## Cách thu thập

WHO chặn tải trực tiếp từ môi trường phát triển bằng Cloudflare. Nội dung được lấy qua reader proxy, nhưng mỗi bản ghi vẫn giữ `source_url` trỏ tới trang WHO gốc để kiểm chứng provenance.

## Bản quyền và sử dụng

Giữ nguyên attribution và URL nguồn WHO. Trước khi public dataset hoặc dùng cho mục đích thương mại, cần kiểm tra điều khoản bản quyền và quyền tái phân phối của WHO.

## Kiểm tra nhanh

Dataset phải có 30 file Markdown trong `data/raw/`, mỗi file có nội dung không rỗng và tương ứng với một `document_id` trong `data/manifest.json`.

Sau khi parsing, mỗi document có một file JSON tương ứng trong `data/parsed/` và báo cáo tổng hợp tại `data/parsed/parse_report.json`.
