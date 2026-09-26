# Reranking Failure Analysis

Phần 7 ghi nhận các failure mode cần theo dõi:

| Failure mode | Detection | Fallback/fix | Lesson learned |
|---|---|---|---|
| Model không tải được hoặc inference lỗi | `status=fallback`, note `reranker_failure` | Giữ hybrid order, không làm dừng pipeline | Reranker là lớp cải thiện, không phải dependency bắt buộc của retrieval |
| Query Việt bị dịch sai | `query.original_query` khác `translated_query_en` và trace reranker | Cross-encoder dùng `original_query`, không dịch lại query | Tách query dùng cho BM25 khỏi query dùng cho cross-lingual reranker |
| Tên thuốc/hoạt chất/viết tắt bị biến dạng | Test cặp query/text giữ nguyên `metformin`, `SGLT-2` | Không rewrite source text; ghi score trước/sau | Citation và tên hoạt chất phải lấy từ payload gốc |
| Chunk dài bị truncate | Trace có `cross_encoder_max_length=512` | Chunking top-20, chọn section phù hợp; đánh giá lại với chunk 300/800 | Reranker latency và context limit cần benchmark riêng |
| Reranker xếp sai chunk | So sánh Hit@k/MRR trước và sau | Không công bố reranker tốt hơn nếu benchmark giảm; có thể fallback theo ngưỡng ở bước sau | Cần evaluation dataset thật, không kết luận từ smoke set |

Benchmark tách thời gian preload model khỏi latency từng query. Cold-start latency vẫn được ghi riêng vì ứng dụng thực tế nên load model khi khởi động.
