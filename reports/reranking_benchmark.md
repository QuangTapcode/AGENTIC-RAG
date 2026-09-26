# Reranking Benchmark

Generated: `2026-09-26T20:14:37Z`
Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`; candidates: `20`; output: `5`
Model preload: `7084.4 ms` (ok)

Smoke-set comparison (document-level expected source; not the final evaluation dataset):

| Chunk size | Hybrid Hit@5 | Reranked Hit@5 | Hybrid MRR | Reranked MRR | Retrieval p50 ms | Rerank p50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 300 | 1.00 | 1.00 | 0.67 | 0.66 | 56.3 | 1102.9 |
| 800 | 0.83 | 0.83 | 0.62 | 0.50 | 37.0 | 1211.6 |

## Failure handling

The reranker uses the original query and preserves source chunk text. Model loading/inference errors return the hybrid order with `status=fallback`; inspect the JSON trace for `reranker_failure` and `fallback_to_hybrid_order`.

Reranking quality is not declared improved unless the official evaluation dataset confirms it. This smoke set is only a pipeline check.
