# Reranking

Phần 7 nhận tối đa 20 candidate từ hybrid retrieval và dùng cross-encoder multilingual:

```text
cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Model nhận cặp `(original_query, chunk_text)`, không nhận `translated_query_en`. Cách này giữ nguyên ý định người dùng, tên thuốc, hoạt chất và viết tắt khi query tiếng Việt được chấm với corpus WHO tiếng Anh.

## Pipeline

```text
hybrid top-20 -> cross-encoder score -> sort giảm dần -> top-5 đưa vào LLM
```

Mỗi kết quả rerank lưu `before_rank`, `before_rrf_score`, `rerank_score`, `chunk_id`, document và section. `max_length=512` được ghi trong trace; text nguồn không bị sửa, còn tokenizer của model sẽ truncate khi cần. Benchmark preload model trước khi đo latency từng query; thời gian khởi động được ghi riêng. Mặc định có latency budget 5000 ms; vượt budget sẽ fallback về hybrid order.

## Chạy benchmark

```powershell
$env:HF_HOME = "C:\Users\LOQ\.cache\huggingface"
$env:PYTHONPATH = "src"
..\.venv\Scripts\python.exe src\reranking\benchmark_reranking.py `
  --candidate-k 20 --output-k 5
```

Kết quả nằm tại `reports/reranking_benchmark.json` và `reports/reranking_benchmark.md`. Lần chạy đầu có thể tải model từ Hugging Face; reranking chạy CPU nên latency cao hơn retrieval.

## Failure handling và bài học

- Nếu model không tải được, inference lỗi, trả sai số lượng score hoặc vượt latency budget, pipeline trả `status=fallback` và giữ nguyên hybrid order.
- Query gốc được đưa thẳng vào reranker để tránh lỗi dịch; benchmark test có `metformin`, `SGLT-2` và kiểm tra text không bị mutate.
- Cross-encoder có giới hạn context; chunk quá dài có thể bị truncate ở `max_length=512`. Cần kiểm tra lại trên evaluation dataset với các câu hỏi về liều lượng, chống chỉ định và tên hoạt chất.
- Reranking có thể làm giảm chất lượng. Chỉ bật mặc định sau khi so sánh Recall/MRR trên evaluation dataset chính thức; smoke benchmark chỉ là kiểm tra pipeline.
