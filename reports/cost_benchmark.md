# Cost & Performance Benchmark — Agentic RAG Y tế

Generated: `2026-09-29T17:50:20.674791+00:00`

This report aggregates artifacts from the parse, chunk, embed, retrieve, and rerank stages. Numbers that require a live Qdrant instance are marked explicitly when they were not available.

## Parsing

- Source artifact: `parse_report.json`
- Documents: **30/30** ok, fallback: **0**, quality flags: **0**
- Total parsing time: **193.4 ms** for 388,147 chars
- Per-document latency (ms): mean **6.45**, p50 **5.51**, p95 **10.89**, max **17.80**

## Chunking

| Size | Total chunks | Total tokens | Token mean | Token max | Overlap target |
|---|---:|---:|---:|---:|---:|
| 300 | 476 | 92,419 | 194.2 | 300 | 60 |
| 800 | 350 | 89,952 | 257.0 | 799 | 120 |

## Embedding + Ingest

- Model: `intfloat/multilingual-e5-small`  (ingested 2026-09-28T19:34:10.485518+00:00)

| Collection | Chunks | Dense dim | Sparse vocab | Avg BM25 doc len | Verification |
|---|---:|---:|---:|---:|---|
| `medical_chunks_300` | 476 | 384 | 5,597 | 122.4 | ok |
| `medical_chunks_800` | 350 | 384 | 5,597 | 162.4 | ok |

## Retrieval latency (smoke set)

| Chunk size | Queries | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| 300 | 6 | 2065.2 | 40.1 | 9154.9 | 12192.7 |
| 800 | 6 | 1824.5 | 45.2 | 8052.8 | 10716.7 |

The `max` value in the smoke set reflects a cold-start where the embedding model and Qdrant client both initialize. In production, load the model once at startup — p50 is the value to design for.

## Reranking latency (smoke set)

| Chunk size | Queries | Mean ms | p50 ms | p95 ms | Max ms |
|---|---:|---:|---:|---:|---:|
| 300 | 6 | 1101.3 | 1117.6 | 1156.4 | 1160.1 |
| 800 | 6 | 1232.8 | 1224.7 | 1332.8 | 1336.5 |

## Qdrant storage

- `medical_chunks_300`: 476 points, status `green`, segments 6, indexed vectors 476
- `medical_chunks_800`: 350 points, status `green`, segments 6, indexed vectors 350

## Cost estimate

- Basis: **30** queries (evaluation set size).
- Assumed tokens/query: input **1800**, output **220**.
- Assumed pricing (USD/1K tokens): input **$0.0**, output **$0.0**.
- Estimated LLM cost: **$0.000000** (54,000 in / 6,600 out).
- Estimated per-query cost: **$0.000000**.

Note: prototype default pricing is `0` because the answer generator falls back to grounded evidence excerpts (no LLM call). Provide a `--pricing-json` file when you wire in a paid LLM to get real numbers.

## Repeated runs

For statistical significance, run this benchmark script at least 3 times and average the p50/p95 across runs. The parse and chunk artifacts do not change between runs unless the corpus changes; retrieval/reranking do vary run-to-run and benefit from repetition.
