# Evaluation Report — Agentic RAG Y tế

Generated: `2026-09-29T18:13:43.266722+00:00` — mode: `full` — dataset: `questions.jsonl`

## Router

Expected labels are derived from `questions.jsonl.category` (`out_of_scope` → out; everything else → in-scope; `clarify` folded as `in_scope`).

### Binary (in-scope / out-of-scope)

- Accuracy: **100.00%**
- Macro F1: **1.0000**
- Weighted F1: **1.0000**
- Out-of-scope blocked: **5/5** (100%)

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| in_scope | 1.0000 | 1.0000 | 1.0000 | 25 |
| out_of_scope | 1.0000 | 1.0000 | 1.0000 | 5 |

### Three-way (with `clarify` visible)

- Accuracy: **83.33%**
- Macro F1: **0.6296**

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| in_scope | 1.0000 | 0.8000 | 0.8889 | 25 |
| out_of_scope | 1.0000 | 1.0000 | 1.0000 | 5 |
| clarify | 0.0000 | 0.0000 | 0.0000 | 0 |

## Configuration comparison — document level

Document-level Recall/MRR is the fair cross-config metric: `expected_source.chunk_id` in the eval set is anchored on chunks_300, so chunk-level Recall on chunks_800 is not comparable. Document-level uses `expected_source.document_id`.

| Config | Doc Recall@5 | Doc Recall@10 | Doc MRR | Doc NDCG@5 | Citation P | Citation R | Ans token F1 | Refuse (must) | Over-refuse |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `dense_chunk300` | 0.45 | 0.45 | 0.39 | 0.41 | 0.35 | 0.35 | 0.05 | 0.50 | 0.25 |
| `dense_chunk800` | 0.60 | 0.60 | 0.43 | 0.47 | 0.35 | 0.35 | 0.05 | 0.50 | 0.25 |
| `hybrid_chunk300` | 0.75 | 0.75 | 0.58 | 0.62 | 0.45 | 0.45 | 0.05 | 0.50 | 0.25 |
| `hybrid_chunk800` | 0.70 | 0.70 | 0.62 | 0.64 | 0.55 | 0.55 | 0.05 | 0.50 | 0.25 |
| `hybrid_rerank_chunk300` | 0.75 | 0.75 | 0.54 | 0.59 | 0.40 | 0.40 | 0.05 | 0.50 | 0.25 |

### Chunk-level (chunks_300 configs only — strict)

| Config | Recall@5 | Recall@10 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|
| `dense_chunk300` | 0.45 | 0.45 | 0.37 | 0.39 |
| `hybrid_chunk300` | 0.75 | 0.75 | 0.48 | 0.54 |
| `hybrid_rerank_chunk300` | 0.75 | 0.75 | 0.50 | 0.56 |

## Latency (ms) — per query

| Config | Retrieval p50 | Retrieval p95 | Rerank p50 | Rerank p95 | Answer p50 | Answer p95 |
|---|---:|---:|---:|---:|---:|---:|
| `dense_chunk300` | 28.0 | 542.5 | 0.0 | 0.0 | 0.3 | 0.6 |
| `dense_chunk800` | 32.4 | 550.4 | 0.0 | 0.0 | 0.4 | 0.5 |
| `hybrid_chunk300` | 34.9 | 546.9 | 0.0 | 0.0 | 0.4 | 0.6 |
| `hybrid_chunk800` | 22.7 | 494.0 | 0.0 | 0.0 | 0.3 | 0.6 |
| `hybrid_rerank_chunk300` | 24.7 | 488.3 | 1555.0 | 1760.9 | 0.3 | 0.6 |

## Configurations

- `dense_chunk300`: chunk=300, mode=dense_only, rerank=False, top_k_output=5
- `dense_chunk800`: chunk=800, mode=dense_only, rerank=False, top_k_output=5
- `hybrid_chunk300`: chunk=300, mode=hybrid, rerank=False, top_k_output=5
- `hybrid_chunk800`: chunk=800, mode=hybrid, rerank=False, top_k_output=5
- `hybrid_rerank_chunk300`: chunk=300, mode=hybrid, rerank=True, top_k_output=5

## Metric definitions

- **Recall@k / Precision@k**: chunk-level, based on `expected_source.chunk_id`. Aggregated over answerable questions only (`in_scope_answerable`).
- **MRR / NDCG@k**: same expected-chunk basis. NDCG uses binary relevance.
- **Citation precision**: fraction of shown citation documents equal to expected document.
- **Citation recall**: 1 if expected chunk (or fallback: expected document) appears in shown citations; 0 otherwise.
- **Answer token F1**: bag-of-token F1 between generated answer and `expected_answer` (Vietnamese/English NFKC casefold). Lexical proxy for correctness — for a full correctness/faithfulness reading, an LLM judge should be added.
- **Refusal recall on must-refuse**: fraction of `must_refuse=true` questions that the pipeline actually refused (out-of-scope or insufficient-evidence).
- **Over-refusal rate**: fraction of answerable questions the pipeline mistakenly refused.
