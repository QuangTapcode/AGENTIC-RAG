# Multilingual Retrieval Architecture

## Mục tiêu

Cho phép người dùng hỏi bằng tiếng Việt trong khi corpus WHO hiện tại chủ yếu là tiếng Anh. Hệ thống phải tìm đúng evidence xuyên ngôn ngữ, sau đó trả lời bằng ngôn ngữ của người dùng.

## Luồng xử lý

```text
User query
   ↓
Language detection
   ↓
Scope router
   ↓
Query object
   ├── original_query: tiếng Việt hoặc tiếng Anh
   ├── normalized_query: chuẩn hóa thuật ngữ, tên thuốc, viết tắt
   └── translated_query_en: dùng cho BM25 trên corpus tiếng Anh
   ↓
Dense multilingual retrieval
   + BM25 retrieval bằng translated_query_en
   ↓
RRF fusion
   ↓
Cross-lingual reranking
   ↓
Evidence check
   ↓
LLM answer bằng ngôn ngữ của query + citation
```

## Cấu trúc dữ liệu đề xuất

```text
data/
├── raw/                  # Tài liệu nguồn tiếng Anh từ WHO
├── parsed/               # Parsed text, heading, table, source metadata
├── normalized/           # Chuẩn hóa thuật ngữ và tên thuốc
├── chunks_300/           # Chunk size 300 tokens
├── chunks_800/           # Chunk size 800 tokens
├── terminology/
│   ├── medical_terms_vi_en.json
│   └── drug_aliases.json
├── manifest.json
└── README.md

src/
├── language/
│   ├── detect.py
│   ├── normalize.py
│   └── translate_query.py
├── embeddings/
│   └── multilingual_embedder.py
├── retrieval/
│   ├── dense.py
│   ├── bm25.py
│   ├── fusion.py
│   └── reranker.py
└── agent/
    └── pipeline.py
```

## Query object

```json
{
  "query_id": "q_001",
  "original_query": "Triệu chứng của bệnh tiểu đường là gì?",
  "detected_language": "vi",
  "normalized_query": "triệu chứng bệnh tiểu đường",
  "translated_query_en": "What are the symptoms of diabetes?",
  "answer_language": "vi"
}
```

## Retrieval strategy

### Dense retrieval

- Dùng multilingual embedding để query tiếng Việt và tài liệu tiếng Anh nằm trong cùng semantic space.
- Embed `original_query`.
- Lấy top-k dense candidates.

### BM25 retrieval

- Dịch query sang tiếng Anh nhưng giữ nguyên tên thuốc, hoạt chất và viết tắt.
- Embed/index corpus tiếng Anh bằng sparse BM25.
- Dùng `translated_query_en` để tìm exact terms như tên bệnh, hoạt chất và dosage.

### Fusion và reranking

- Kết hợp dense và BM25 bằng RRF.
- Rerank các candidate bằng cross-lingual reranker.
- So sánh ba cấu hình:
- Dense multilingual-only.
  - Dense multilingual + BM25.
  - Dense multilingual + BM25 + reranking.

## Nguyên tắc dịch query

- Không dịch câu trả lời trước khi đưa vào LLM.
- Không dịch mù tên thuốc hoặc hoạt chất.
- Nếu dịch thất bại, vẫn chạy dense multilingual bằng query gốc.
- Log cả query gốc và query dịch để phân tích lỗi.

## Evaluation bắt buộc

Evaluation dataset cần chia theo ngôn ngữ:

- 15 câu tiếng Việt trên tài liệu tiếng Anh.
- 10 câu tiếng Anh trên tài liệu tiếng Anh.
- 5 câu ngoài phạm vi hoặc thiếu evidence.

Theo dõi riêng Recall@k, MRR, citation quality, answer faithfulness và latency cho query tiếng Việt/Anh. Không chỉ đo điểm trung bình chung vì điểm chung có thể che giấu việc retrieval tiếng Việt hoạt động kém.

## Failure cases cần theo dõi

- Dịch sai tên thuốc hoặc hoạt chất.
- Query tiếng Việt có từ viết tắt mà BM25 không nhận ra.
- Dense retrieval đúng chủ đề nhưng sai bệnh cụ thể.
- BM25 tìm đúng từ nhưng sai ngữ cảnh.
- Reranker multilingual chậm hoặc ưu tiên chunk không trả lời query.
- LLM trả lời tiếng Việt nhưng thêm kiến thức không có trong context.
