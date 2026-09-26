# Hybrid Retrieval

Phần 6 dùng hai retriever trên cùng collection Qdrant:

1. Dense multilingual retrieval embed query gốc với `intfloat/multilingual-e5-small` và prefix `query:`.
2. BM25 sparse retrieval dùng `translated_query_en` vì corpus WHO hiện tại là tiếng Anh.
3. RRF (Reciprocal Rank Fusion) kết hợp rank của hai danh sách, đồng thời lưu `score`, `rank` và `chunk_id` của từng retriever.

## Query processing

Mỗi query tạo `original_query`, `normalized_query`, `detected_language` và `translated_query_en`.
`MedicalQueryTranslator` có dictionary cho các thuật ngữ y tế Việt–Anh phổ biến và nhận thêm `external_translator` nếu sau này dùng translation service/LLM. Nếu không dịch được, hệ thống giữ query và ghi rõ `translation_method` để trace không bị đánh tráo chất lượng.

## Chạy một query

Từ thư mục `AISOEASY`:

```powershell
$env:HF_HOME = "C:\Users\LOQ\.cache\huggingface"
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe src\retrieval\hybrid_retriever.py `
  "Bệnh tiểu đường có triệu chứng gì?" `
  --chunk-size 300 --top-k-dense 10 --top-k-bm25 10 --top-k-fused 5 `
  --output reports\retrieval_trace.json
```

## Benchmark

`smoke_queries.jsonl` là smoke set có expected document để so sánh dense-only, BM25-only và hybrid. Đây chưa phải evaluation dataset chính thức của Phần 9/10; nó chỉ kiểm tra pipeline retrieval và lựa chọn top-k ban đầu.

```powershell
$env:HF_HOME = "C:\Users\LOQ\.cache\huggingface"
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe src\retrieval\benchmark_retrieval.py
```

Kết quả được ghi tại `reports/retrieval_benchmark.json` và `reports/retrieval_benchmark.md`.

## Giới hạn và bài học

- Dictionary translation không thay thế được translation model: query Việt dài hoặc tên hoạt chất mới có thể không có từ khóa BM25. Dense multilingual vẫn là đường lui chính.
- RRF dùng rank, không so sánh trực tiếp thang điểm cosine và BM25; đây là lý do score/rank gốc của mỗi retriever được lưu riêng.
- Smoke set nhỏ không đủ để kết luận chất lượng y khoa. Cần evaluation dataset có expected source/chunk và đánh giá Recall@k, MRR, NDCG ở Phần 9/10.
