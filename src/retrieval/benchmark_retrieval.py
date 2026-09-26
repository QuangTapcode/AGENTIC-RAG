"""Benchmark dense-only, BM25-only and hybrid retrieval on a smoke set."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    from retrieval.hybrid_retriever import HybridRetriever, RetrievalHit, rrf_fuse
except ImportError:  # Allows direct execution from the repository root.
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from retrieval.hybrid_retriever import HybridRetriever, RetrievalHit, rrf_fuse  # type: ignore[no-redef]


DEFAULT_CANDIDATES = (
    {"top_k_dense": 5, "top_k_bm25": 5, "top_k_fused": 5},
    {"top_k_dense": 10, "top_k_bm25": 10, "top_k_fused": 5},
    {"top_k_dense": 20, "top_k_bm25": 20, "top_k_fused": 5},
    {"top_k_dense": 10, "top_k_bm25": 20, "top_k_fused": 5},
    {"top_k_dense": 20, "top_k_bm25": 10, "top_k_fused": 5},
)


def read_queries(path: Path) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not item.get("query") or not item.get("expected_document_ids"):
                raise ValueError(f"Invalid smoke query at {path}:{line_number}")
            queries.append(item)
    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def document_ids(results: Sequence[Any]) -> list[str]:
    return [str(result.payload.get("document_id", "")) for result in results]


def hit_at_k(results: Sequence[Any], expected: set[str], k: int) -> int:
    return int(bool(set(document_ids(results[:k])) & expected))


def reciprocal_rank(results: Sequence[Any], expected: set[str]) -> float:
    for rank, document_id in enumerate(document_ids(results), start=1):
        if document_id in expected:
            return 1.0 / rank
    return 0.0


def summarize(rows: Sequence[dict[str, Any]], result_key: str) -> dict[str, float]:
    result_lists = [row[result_key] for row in rows]
    return {
        "hit_at_1": round(statistics.mean(hit_at_k(results, row["expected"], 1) for results, row in zip(result_lists, rows)), 4),
        "hit_at_5": round(statistics.mean(hit_at_k(results, row["expected"], 5) for results, row in zip(result_lists, rows)), 4),
        "mrr": round(statistics.mean(reciprocal_rank(results, row["expected"]) for results, row in zip(result_lists, rows)), 4),
    }


def evaluate_candidate(
    rows: Sequence[dict[str, Any]],
    *,
    top_k_dense: int,
    top_k_bm25: int,
    top_k_fused: int,
) -> dict[str, Any]:
    candidate_rows: list[dict[str, Any]] = []
    for row in rows:
        dense = row["dense_results"][:top_k_dense]
        bm25 = row["bm25_results"][:top_k_bm25]
        fused = rrf_fuse(dense, bm25, top_k=top_k_fused)
        candidate_rows.append({"expected": row["expected"], "hybrid_results": fused})
    summary = summarize(candidate_rows, "hybrid_results")
    return {
        "top_k_dense": top_k_dense,
        "top_k_bm25": top_k_bm25,
        "top_k_fused": top_k_fused,
        **summary,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Retrieval Benchmark",
        "",
        f"Generated: `{report['generated_at']}`",
        "",
        "Smoke-set comparison (document-level expected source; not the final evaluation dataset):",
        "",
        "| Chunk size | Dense-only Hit@5 | BM25-only Hit@5 | Hybrid Hit@5 | Hybrid MRR |",
        "|---:|---:|---:|---:|---:|",
    ]
    for size, item in report["chunk_sizes"].items():
        baseline = item["baselines"]
        lines.append(
            f"| {size} | {baseline['dense_only']['hit_at_5']:.2f} | "
            f"{baseline['bm25_only']['hit_at_5']:.2f} | "
            f"{baseline['hybrid']['hit_at_5']:.2f} | "
            f"{baseline['hybrid']['mrr']:.2f} |"
        )
    lines.extend(["", "## Top-k candidates", "", "| Chunk size | Dense k | BM25 k | Fused k | Hit@5 | MRR |", "|---:|---:|---:|---:|---:|---:|"])
    for size, item in report["chunk_sizes"].items():
        for candidate in item["top_k_candidates"]:
            lines.append(
                f"| {size} | {candidate['top_k_dense']} | {candidate['top_k_bm25']} | "
                f"{candidate['top_k_fused']} | {candidate['hit_at_5']:.2f} | {candidate['mrr']:.2f} |"
            )
    recommendation = report["recommendation"]
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            f"Use chunk `{recommendation['chunk_size']}` with `top_k_dense={recommendation['top_k_dense']}`, "
            f"`top_k_bm25={recommendation['top_k_bm25']}` and `top_k_fused={recommendation['top_k_fused']}` "
            f"for the current smoke set (Hit@5={recommendation['hit_at_5']:.2f}, MRR={recommendation['mrr']:.2f}).",
            "This is a starting configuration; re-tune it on the official evaluation dataset in Parts 9–10.",
            "",
            "## Caveat",
            "",
            "BM25 quality for Vietnamese depends on the observable dictionary/external translation method. Dense multilingual retrieval remains the fallback when the translated query has no corpus terms.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--model", default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"))
    parser.add_argument("--queries", type=Path, default=project_root / "data" / "retrieval" / "smoke_queries.jsonl")
    parser.add_argument("--reports-dir", type=Path, default=project_root / "reports")
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root: Path = args.project_root
    query_path = args.queries if args.queries.is_absolute() else project_root / args.queries
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else project_root / args.reports_dir
    queries = read_queries(query_path)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query_file": str(query_path.relative_to(project_root)).replace("\\", "/"),
        "model": args.model,
        "chunk_sizes": {},
    }
    all_candidates: list[dict[str, Any]] = []

    for chunk_size in (300, 800):
        retriever = HybridRetriever(
            qdrant_url=args.qdrant_url,
            collection_name=f"medical_chunks_{chunk_size}",
            bm25_artifact_path=project_root / "data" / "vector_store" / f"chunks_{chunk_size}" / "bm25_index.json",
            model_name=args.model,
        )
        rows: list[dict[str, Any]] = []
        traces: list[dict[str, Any]] = []
        for item in queries:
            result = retriever.retrieve(item["query"], top_k_dense=20, top_k_bm25=20, top_k_fused=20)
            rows.append(
                {
                    "expected": set(item["expected_document_ids"]),
                    "dense_results": result.dense_results,
                    "bm25_results": result.sparse_results,
                }
            )
            traces.append(
                {
                    "id": item.get("id"),
                    "query": result.query.to_dict(),
                    "expected_document_ids": item["expected_document_ids"],
                    "latency_ms": round(result.latency_ms, 3),
                    "dense_top5": [hit.to_dict() for hit in result.dense_results[:5]],
                    "bm25_top5": [hit.to_dict() for hit in result.sparse_results[:5]],
                    "hybrid_top5": [hit.to_dict() for hit in result.fused_results[:5]],
                    "notes": result.notes,
                }
            )

        baseline_rows = []
        for row in rows:
            fused = rrf_fuse(row["dense_results"], row["bm25_results"], top_k=20)
            baseline_rows.append(
                {
                    "expected": row["expected"],
                    "dense_only": row["dense_results"],
                    "bm25_only": row["bm25_results"],
                    "hybrid": fused,
                }
            )
        candidates = [evaluate_candidate(rows, **candidate) for candidate in DEFAULT_CANDIDATES]
        for candidate in candidates:
            candidate["chunk_size"] = chunk_size
        all_candidates.extend(candidates)
        report["chunk_sizes"][str(chunk_size)] = {
            "baselines": {
                "dense_only": summarize(baseline_rows, "dense_only"),
                "bm25_only": summarize(baseline_rows, "bm25_only"),
                "hybrid": summarize(baseline_rows, "hybrid"),
            },
            "top_k_candidates": candidates,
            "traces": traces,
        }

    best = sorted(
        all_candidates,
        key=lambda item: (
            -item["hit_at_5"],
            -item["mrr"],
            item["top_k_dense"] + item["top_k_bm25"] + item["top_k_fused"],
            item["chunk_size"],
        ),
    )[0]
    report["recommendation"] = best
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "retrieval_benchmark.json"
    markdown_path = reports_dir / "retrieval_benchmark.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(markdown_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
