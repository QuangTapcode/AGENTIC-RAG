# Retrieval Benchmark

Generated: `2026-09-26T19:58:42Z`

Smoke-set comparison (document-level expected source; not the final evaluation dataset):

| Chunk size | Dense-only Hit@5 | BM25-only Hit@5 | Hybrid Hit@5 | Hybrid MRR |
|---:|---:|---:|---:|---:|
| 300 | 0.33 | 1.00 | 1.00 | 0.67 |
| 800 | 0.33 | 1.00 | 0.83 | 0.64 |

## Top-k candidates

| Chunk size | Dense k | BM25 k | Fused k | Hit@5 | MRR |
|---:|---:|---:|---:|---:|---:|
| 300 | 5 | 5 | 5 | 1.00 | 0.62 |
| 300 | 10 | 10 | 5 | 1.00 | 0.62 |
| 300 | 20 | 20 | 5 | 1.00 | 0.67 |
| 300 | 10 | 20 | 5 | 1.00 | 0.62 |
| 300 | 20 | 10 | 5 | 1.00 | 0.68 |
| 800 | 5 | 5 | 5 | 1.00 | 0.60 |
| 800 | 10 | 10 | 5 | 1.00 | 0.66 |
| 800 | 20 | 20 | 5 | 0.83 | 0.62 |
| 800 | 10 | 20 | 5 | 1.00 | 0.66 |
| 800 | 20 | 10 | 5 | 1.00 | 0.65 |

## Recommendation

Use chunk `300` with `top_k_dense=20`, `top_k_bm25=10` and `top_k_fused=5` for the current smoke set (Hit@5=1.00, MRR=0.68).
This is a starting configuration; re-tune it on the official evaluation dataset in Parts 9–10.

## Caveat

BM25 quality for Vietnamese depends on the observable dictionary/external translation method. Dense multilingual retrieval remains the fallback when the translated query has no corpus terms.
