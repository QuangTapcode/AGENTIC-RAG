# Vector store

Phần 4 lưu hai collection Qdrant local:

- `medical_chunks_300`: dense multilingual E5 + sparse BM25 cho `data/chunks_300/`.
- `medical_chunks_800`: dense multilingual E5 + sparse BM25 cho `data/chunks_800/`.

Dense embedding dùng `intfloat/multilingual-e5-small` (384 chiều). Tài liệu được encode với tiền tố `passage:`; query ở các bước retrieval sau phải dùng tiền tố `query:`. Sparse vector dùng tokenizer Unicode case-fold và trọng số BM25; vocabulary/IDF của từng corpus nằm trong `bm25_index.json`.

Payload Qdrant giữ toàn bộ metadata của chunk: `chunk_id`, `document_id`, `title`, `topic`, `source_url`, `source_language`, `published_date`, `chunk_size`, `section`, line/page metadata, `text` và quality flags.

## Chạy

```powershell
docker compose up -d qdrant
..\.venv\Scripts\python.exe src\embedding\ingest_qdrant.py --recreate
```

Kiểm tra số point, payload và vector ở `ingestion_report.json`.

## Failure và bài học

- Lần tải model đầu tiên hết dung lượng ổ D vì model khoảng 471 MB; cache Hugging Face được chuyển sang ổ C, không đặt trong repository.
- Docker CLI ban đầu chưa kết nối Engine; sau khi Docker Desktop chạy, Qdrant được khởi động và kiểm tra collection ở trạng thái `green`.
- Sparse BM25 cần vocabulary/IDF riêng cho từng bộ chunk; không dùng chung artifact 300 và 800 vì độ dài tài liệu trung bình khác nhau.
