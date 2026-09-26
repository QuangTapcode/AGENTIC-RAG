"""Benchmark hybrid retrieval before and after cross-encoder reranking."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Sequence

try:
    from retrieval.benchmark_retrieval import read_queries, hit_at_k, reciprocal_rank
    from retrieval.hybrid_retriever import HybridRetriever
    from reranking.cross_encoder_reranker import (
        DEFAULT_CANDIDATE_K,
        DEFAULT_LATENCY_BUDGET_MS,
        DEFAULT_OUTPUT_K,
        RERANKER_MODEL_NAME,
        CrossEncoderReranker,
        RerankedHit,
    )
except ImportError:  # Allows direct execution from the repository root.
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from retrieval.benchmark_retrieval import read_queries, hit_at_k, reciprocal_rank  # type: ignore[no-redef]
    from retrieval.hybrid_retriever import HybridRetriever  # type: ignore[no-redef]
    from reranking.cross_encoder_reranker import (  # type: ignore[no-redef]
        DEFAULT_CANDIDATE_K,
        DEFAULT_LATENCY_BUDGET_MS,
        DEFAULT_OUTPUT_K,
        RERANKER_MODEL_NAME,
        CrossEncoderReranker,
        RerankedHit,
    )


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return round(ordered[index], 3)


def document_ids(results: Sequence[Any]) -> list[str]:
    return [str(result.payload.get("document_id", "")) for result in results]


def summarize(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        "hit_at_1": round(statistics.mean(hit_at_k(row[key], row["expected"], 1) for row in rows), 4),
        "hit_at_5": round(statistics.mean(hit_at_k(row[key], row["expected"], 5) for row in rows), 4),
        "mrr": round(statistics.mean(reciprocal_rank(row[key], row["expected"]) for row in rows), 4),
    }


def compact_fused_hit(hit: Any) -> dict[str, Any]:
    payload = hit.payload
    return {
        "chunk_id": hit.chunk_id,
        "document_id": payload.get("document_id"),
        "section": payload.get("section"),
        "rank": hit.rank,
        "rrf_score": getattr(hit, "rrf_score", None),
        "rerank_score": getattr(hit, "rerank_score", None),
        "before_rank": getattr(hit, "before_rank", None),
        "before_rrf_score": getattr(hit, "before_rrf_score", None),
        "dense_rank": getattr(hit, "dense_rank", None),
        "bm25_rank": getattr(hit, "bm25_rank", None),
        "sources": getattr(hit, "sources", None),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Reranking Benchmark",
        "",
        f"Generated: `{report['generated_at']}`",
        f"Model: `{report['model']}`; candidates: `{report['candidate_limit']}`; output: `{report['output_k']}`",
        f"Model preload: `{report['model_load_latency_ms']:.1f} ms` ({report['model_load_status']})",
        "",
        "Smoke-set comparison (document-level expected source; not the final evaluation dataset):",
        "",
        "| Chunk size | Hybrid Hit@5 | Reranked Hit@5 | Hybrid MRR | Reranked MRR | Retrieval p50 ms | Rerank p50 ms |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for size, item in report["chunk_sizes"].items():
        baseline = item["before"]
        reranked = item["after"]
        lines.append(
            f"| {size} | {baseline['metrics']['hit_at_5']:.2f} | {reranked['metrics']['hit_at_5']:.2f} | "
            f"{baseline['metrics']['mrr']:.2f} | {reranked['metrics']['mrr']:.2f} | "
            f"{item['latency_ms']['retrieval_p50']:.1f} | {item['latency_ms']['rerank_p50']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Failure handling",
            "",
            "The reranker uses the original query and preserves source chunk text. Model loading/inference errors return the hybrid order with `status=fallback`; inspect the JSON trace for `reranker_failure` and `fallback_to_hybrid_order`.",
            "",
            "Reranking quality is not declared improved unless the official evaluation dataset confirms it. This smoke set is only a pipeline check.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    parser.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL", RERANKER_MODEL_NAME))
    parser.add_argument("--queries", type=Path, default=project_root / "data" / "retrieval" / "smoke_queries.jsonl")
    parser.add_argument("--reports-dir", type=Path, default=project_root / "reports")
    parser.add_argument("--candidate-k", type=int, default=DEFAULT_CANDIDATE_K)
    parser.add_argument("--output-k", type=int, default=DEFAULT_OUTPUT_K)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latency-budget-ms", type=float, default=DEFAULT_LATENCY_BUDGET_MS)
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.candidate_k < 1 or args.output_k < 1 or args.output_k > args.candidate_k:
        raise SystemExit("Require 1 <= --output-k <= --candidate-k")
    project_root: Path = args.project_root
    query_path = args.queries if args.queries.is_absolute() else project_root / args.queries
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else project_root / args.reports_dir
    queries = read_queries(query_path)
    reranker = CrossEncoderReranker(
        model_name=args.reranker_model,
        batch_size=args.batch_size,
        latency_budget_ms=args.latency_budget_ms,
    )
    model_load_started = time.perf_counter()
    model_load_status = "ok"
    model_load_error = None
    try:
        reranker.preload()
    except Exception as exc:  # Per-query rerank still records fallback behavior.
        model_load_status = "failed"
        model_load_error = type(exc).__name__
    model_load_latency_ms = round((time.perf_counter() - model_load_started) * 1000.0, 3)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_file": str(query_path.relative_to(project_root)).replace("\\", "/"),
        "embedding_model": args.embedding_model,
        "model": args.reranker_model,
        "candidate_limit": args.candidate_k,
        "output_k": args.output_k,
        "model_load_status": model_load_status,
        "model_load_error": model_load_error,
        "model_load_latency_ms": model_load_latency_ms,
        "chunk_sizes": {},
    }

    for chunk_size in (300, 800):
        retriever = HybridRetriever(
            qdrant_url=args.qdrant_url,
            collection_name=f"medical_chunks_{chunk_size}",
            bm25_artifact_path=project_root / "data" / "vector_store" / f"chunks_{chunk_size}" / "bm25_index.json",
            model_name=args.embedding_model,
        )
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        retrieval_latencies: list[float] = []
        rerank_latencies: list[float] = []
        statuses: dict[str, int] = {}
        for item in queries:
            retrieval = retriever.retrieve(
                item["query"],
                top_k_dense=args.candidate_k,
                top_k_bm25=args.candidate_k,
                top_k_fused=args.candidate_k,
            )
            reranked = reranker.rerank(
                retrieval.query,
                retrieval.fused_results,
                candidate_limit=args.candidate_k,
                top_k=args.output_k,
            )
            expected = set(item["expected_document_ids"])
            rows.append(
                {
                    "expected": expected,
                    "before": retrieval.fused_results[:args.output_k],
                    "after": reranked.reranked_results,
                }
            )
            retrieval_latencies.append(retrieval.latency_ms)
            rerank_latencies.append(reranked.latency_ms)
            statuses[reranked.status] = statuses.get(reranked.status, 0) + 1
            traces.append(
                {
                    "id": item.get("id"),
                    "query": reranked.query.to_dict(),
                    "expected_document_ids": item["expected_document_ids"],
                    "retrieval_latency_ms": round(retrieval.latency_ms, 3),
                    "rerank_latency_ms": round(reranked.latency_ms, 3),
                    "reranker_status": reranked.status,
                    "hybrid_top": [compact_fused_hit(hit) for hit in retrieval.fused_results[:args.output_k]],
                    "reranked_top": [compact_fused_hit(hit) for hit in reranked.reranked_results],
                    "candidate_scores_before_after": [compact_fused_hit(hit) for hit in reranked.reranked_results],
                    "notes": retrieval.notes + reranked.notes,
                }
            )
        report["chunk_sizes"][str(chunk_size)] = {
            "before": {"metrics": summarize(rows, "before")},
            "after": {"metrics": summarize(rows, "after")},
            "latency_ms": {
                "retrieval_p50": percentile(retrieval_latencies, 0.50),
                "retrieval_p95": percentile(retrieval_latencies, 0.95),
                "rerank_p50": percentile(rerank_latencies, 0.50),
                "rerank_p95": percentile(rerank_latencies, 0.95),
            },
            "statuses": statuses,
            "traces": traces,
        }

    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "reranking_benchmark.json"
    markdown_path = reports_dir / "reranking_benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
