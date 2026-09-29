"""Aggregate cost and performance benchmark across pipeline stages.

The script reads existing artifacts that were emitted by earlier pipeline
stages:

* ``data/parsed/parse_report.json`` — per-document parsing latency.
* ``data/chunks_{300,800}/chunk_report.json`` — chunking summary and token
  counts.
* ``data/vector_store/ingestion_report.json`` — embedding + Qdrant ingest.
* ``reports/retrieval_benchmark.json`` — smoke-set retrieval latency traces.
* ``reports/reranking_benchmark.json`` — smoke-set rerank latency traces.

When Qdrant is available at ``--qdrant-url``, it also queries the collection
info to report on-disk storage. Otherwise storage is marked ``unavailable``
with a clear note.

Cost estimates for LLM calls are computed from a caller-provided pricing
table (``--pricing-json``); the default table assumes no LLM call because
the prototype's default answer path is the grounded evidence-excerpt
fallback. Multiple LLM providers can be modeled by re-running with a
different pricing file.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_TOKEN_ESTIMATE = {
    # Rough per-query estimates for a full-pipeline run. Adjust when you
    # measure real prompt sizes with a specific LLM.
    "llm_input_tokens_per_query": 1800,
    "llm_output_tokens_per_query": 220,
    "embedding_calls_per_query": 1,
    "reranker_calls_per_query": 1,
}

DEFAULT_PRICING = {
    # USD per 1K tokens. Prototype default is zero cost (evidence-excerpt fallback).
    "llm_input_usd_per_1k": 0.0,
    "llm_output_usd_per_1k": 0.0,
    "embedding_usd_per_1k": 0.0,
    # Cross-encoder is local; the per-query cost is compute time, not USD.
    "reranker_usd_per_call": 0.0,
    "notes": "Prototype default. Set explicit prices when using a paid LLM (e.g. Claude/GPT).",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def latency_summary(values: Iterable[float]) -> dict[str, float]:
    values = list(values)
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(_pct(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #


def summarize_parse(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {"available": False, "path": str(report_path.name)}
    data = read_json(report_path)
    elapsed = [r["elapsed_ms"] for r in data.get("results", []) if r.get("elapsed_ms") is not None]
    char_counts = [r.get("char_count", 0) for r in data.get("results", [])]
    return {
        "available": True,
        "path": str(report_path.name),
        "document_count": data.get("document_count"),
        "success_count": data.get("success_count"),
        "fallback_count": data.get("fallback_count"),
        "quality_flag_count": data.get("quality_flag_count"),
        "total_ms": round(sum(elapsed), 3),
        "per_document_ms": latency_summary(elapsed),
        "total_chars": sum(char_counts),
    }


def summarize_chunks(chunk_reports: Mapping[int, Path]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for size, path in chunk_reports.items():
        if not path.exists():
            entries[str(size)] = {"available": False}
            continue
        data = read_json(path)
        summary = data.get("summary", {})
        entries[str(size)] = {
            "available": True,
            "total_chunks": summary.get("total_chunks"),
            "documents_with_chunks": summary.get("documents_with_chunks"),
            "total_tokens": summary.get("total_tokens"),
            "token_statistics": summary.get("token_statistics"),
            "overlap_target_tokens": data.get("overlap_target_tokens"),
        }
    return entries


def summarize_ingestion(report_path: Path) -> dict[str, Any]:
    if not report_path.exists():
        return {"available": False}
    data = read_json(report_path)
    return {
        "available": True,
        "generated_at": data.get("generated_at"),
        "model": data.get("model"),
        "collections": [
            {
                "collection": r["collection"],
                "chunk_count": r["chunk_count"],
                "dense_dimension": r["dense_dimension"],
                "sparse_vocabulary_size": r["sparse"]["vocabulary_size"],
                "sparse_average_document_length": r["sparse"]["average_document_length"],
                "verification": r["verification"]["status"],
            }
            for r in data.get("reports", [])
        ],
    }


def summarize_retrieval(retrieval_path: Path) -> dict[str, Any]:
    if not retrieval_path.exists():
        return {"available": False}
    data = read_json(retrieval_path)
    entries: dict[str, Any] = {"available": True, "chunk_sizes": {}}
    for size, item in data.get("chunk_sizes", {}).items():
        traces = item.get("traces", [])
        latencies = [t.get("latency_ms", 0.0) for t in traces]
        entries["chunk_sizes"][size] = {
            "query_count": len(traces),
            "latency_ms": latency_summary(latencies),
        }
    return entries


def summarize_reranking(rerank_path: Path) -> dict[str, Any]:
    if not rerank_path.exists():
        return {"available": False}
    data = read_json(rerank_path)
    entries: dict[str, Any] = {"available": True, "chunk_sizes": {}}
    for size, item in data.get("chunk_sizes", {}).items():
        traces = item.get("traces", [])
        latencies = [t.get("rerank_latency_ms", 0.0) for t in traces]
        entries["chunk_sizes"][size] = {
            "query_count": len(traces),
            "latency_ms": latency_summary(latencies),
        }
    entries["model_preload_ms"] = data.get("model_preload_ms")
    return entries


def summarize_storage(qdrant_url: str) -> dict[str, Any]:
    """Ask Qdrant for collection info; return {'available': False} if unreachable."""

    try:
        from qdrant_client import QdrantClient  # local import keeps offline path clean
    except ImportError as exc:
        return {"available": False, "reason": f"qdrant_client_not_installed: {exc}"}
    try:
        client = QdrantClient(url=qdrant_url, timeout=5.0)
        entries: dict[str, Any] = {"available": True, "url": qdrant_url, "collections": {}}
        for collection_name in ("medical_chunks_300", "medical_chunks_800"):
            if not client.collection_exists(collection_name):
                entries["collections"][collection_name] = {"exists": False}
                continue
            info = client.get_collection(collection_name)
            count = client.count(collection_name=collection_name, exact=True).count
            entries["collections"][collection_name] = {
                "exists": True,
                "points_count": count,
                "status": str(info.status),
                "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                "segments_count": getattr(info, "segments_count", None),
            }
        return entries
    except Exception as exc:  # network/timeout/etc
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def compute_cost_estimate(
    token_estimate: Mapping[str, float],
    pricing: Mapping[str, float],
    query_count: int,
) -> dict[str, Any]:
    input_tokens = token_estimate["llm_input_tokens_per_query"] * query_count
    output_tokens = token_estimate["llm_output_tokens_per_query"] * query_count
    llm_cost = (
        input_tokens / 1000.0 * pricing["llm_input_usd_per_1k"]
        + output_tokens / 1000.0 * pricing["llm_output_usd_per_1k"]
    )
    embed_calls = token_estimate["embedding_calls_per_query"] * query_count
    rerank_calls = token_estimate["reranker_calls_per_query"] * query_count
    embed_cost = 0.0  # embedding tokens are small; adjust when using a paid embed API.
    rerank_cost = rerank_calls * pricing["reranker_usd_per_call"]
    total = llm_cost + embed_cost + rerank_cost
    return {
        "query_count": query_count,
        "assumed_tokens_per_query": dict(token_estimate),
        "assumed_pricing": dict(pricing),
        "llm_input_tokens_total": input_tokens,
        "llm_output_tokens_total": output_tokens,
        "estimated_llm_cost_usd": round(llm_cost, 6),
        "estimated_embedding_cost_usd": round(embed_cost, 6),
        "estimated_reranker_cost_usd": round(rerank_cost, 6),
        "estimated_total_cost_usd": round(total, 6),
        "estimated_cost_per_query_usd": round(total / max(query_count, 1), 6),
    }


# --------------------------------------------------------------------------- #
# Report rendering                                                            #
# --------------------------------------------------------------------------- #


def render_markdown(payload: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Cost & Performance Benchmark — Agentic RAG Y tế",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "This report aggregates artifacts from the parse, chunk, embed, retrieve, "
        "and rerank stages. Numbers that require a live Qdrant instance are marked "
        "explicitly when they were not available.",
        "",
    ]

    # Parsing
    parse = payload["parsing"]
    lines.extend(["## Parsing", ""])
    if parse.get("available"):
        pdms = parse["per_document_ms"]
        lines.extend(
            [
                f"- Source artifact: `{parse['path']}`",
                f"- Documents: **{parse['success_count']}/{parse['document_count']}** ok, "
                f"fallback: **{parse['fallback_count']}**, quality flags: **{parse['quality_flag_count']}**",
                f"- Total parsing time: **{parse['total_ms']:.1f} ms** for {parse['total_chars']:,} chars",
                f"- Per-document latency (ms): mean **{pdms['mean']:.2f}**, "
                f"p50 **{pdms['p50']:.2f}**, p95 **{pdms['p95']:.2f}**, max **{pdms['max']:.2f}**",
                "",
            ]
        )
    else:
        lines.extend([f"_Not available: `{parse.get('path','')}`_", ""])

    # Chunking
    lines.extend(["## Chunking", ""])
    lines.extend(["| Size | Total chunks | Total tokens | Token mean | Token max | Overlap target |",
                  "|---|---:|---:|---:|---:|---:|"])
    for size, info in payload["chunking"].items():
        if not info.get("available"):
            lines.append(f"| {size} | _unavailable_ | | | | |")
            continue
        tokstats = info["token_statistics"] or {}
        lines.append(
            f"| {size} | {info['total_chunks']} | {info['total_tokens']:,} | "
            f"{tokstats.get('mean', 0):.1f} | {tokstats.get('max', 0)} | "
            f"{info['overlap_target_tokens']} |"
        )
    lines.append("")

    # Ingestion
    ing = payload["ingestion"]
    lines.extend(["## Embedding + Ingest", ""])
    if ing.get("available"):
        lines.append(f"- Model: `{ing['model']}`  (ingested {ing.get('generated_at')})")
        lines.append("")
        lines.extend(
            [
                "| Collection | Chunks | Dense dim | Sparse vocab | Avg BM25 doc len | Verification |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for c in ing["collections"]:
            lines.append(
                f"| `{c['collection']}` | {c['chunk_count']} | {c['dense_dimension']} | "
                f"{c['sparse_vocabulary_size']:,} | {c['sparse_average_document_length']:.1f} | "
                f"{c['verification']} |"
            )
    else:
        lines.append("_Ingestion report unavailable — run `src/embedding/ingest_qdrant.py` first._")
    lines.append("")

    # Retrieval
    ret = payload["retrieval"]
    lines.extend(["## Retrieval latency (smoke set)", ""])
    if ret.get("available"):
        lines.extend(
            [
                "| Chunk size | Queries | Mean ms | p50 ms | p95 ms | Max ms |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for size, info in ret["chunk_sizes"].items():
            lat = info["latency_ms"]
            lines.append(
                f"| {size} | {info['query_count']} | {lat['mean']:.1f} | "
                f"{lat['p50']:.1f} | {lat['p95']:.1f} | {lat['max']:.1f} |"
            )
        lines.append("")
        lines.append(
            "The `max` value in the smoke set reflects a cold-start where the "
            "embedding model and Qdrant client both initialize. In production, "
            "load the model once at startup — p50 is the value to design for."
        )
    else:
        lines.append("_Not available; run `src/retrieval/benchmark_retrieval.py` first._")
    lines.append("")

    # Reranking
    rer = payload["reranking"]
    lines.extend(["## Reranking latency (smoke set)", ""])
    if rer.get("available"):
        preload = rer.get("model_preload_ms")
        if preload is not None:
            lines.append(f"- Cross-encoder preload: **{preload:.1f} ms** (once per process).")
            lines.append("")
        lines.extend(
            [
                "| Chunk size | Queries | Mean ms | p50 ms | p95 ms | Max ms |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for size, info in rer["chunk_sizes"].items():
            lat = info["latency_ms"]
            lines.append(
                f"| {size} | {info['query_count']} | {lat['mean']:.1f} | "
                f"{lat['p50']:.1f} | {lat['p95']:.1f} | {lat['max']:.1f} |"
            )
    else:
        lines.append("_Not available; run `src/reranking/benchmark_reranking.py` first._")
    lines.append("")

    # Storage
    st = payload["storage"]
    lines.extend(["## Qdrant storage", ""])
    if st.get("available"):
        for name, entry in st["collections"].items():
            if not entry.get("exists"):
                lines.append(f"- `{name}`: **not present**")
                continue
            lines.append(
                f"- `{name}`: {entry['points_count']} points, status `{entry['status']}`, "
                f"segments {entry.get('segments_count')}, "
                f"indexed vectors {entry.get('indexed_vectors_count')}"
            )
    else:
        lines.append(f"_Storage snapshot skipped ({st.get('reason', 'reason unknown')})._")
        lines.append("Bring Qdrant up with `docker compose up qdrant` and rerun this script "
                     "to populate the storage row.")
    lines.append("")

    # Cost
    cost = payload["cost_estimate"]
    lines.extend(["## Cost estimate", ""])
    lines.extend(
        [
            f"- Basis: **{cost['query_count']}** queries (evaluation set size).",
            f"- Assumed tokens/query: input **{cost['assumed_tokens_per_query']['llm_input_tokens_per_query']}**, "
            f"output **{cost['assumed_tokens_per_query']['llm_output_tokens_per_query']}**.",
            f"- Assumed pricing (USD/1K tokens): input "
            f"**${cost['assumed_pricing']['llm_input_usd_per_1k']}**, "
            f"output **${cost['assumed_pricing']['llm_output_usd_per_1k']}**.",
            f"- Estimated LLM cost: **${cost['estimated_llm_cost_usd']:.6f}** "
            f"({cost['llm_input_tokens_total']:,} in / {cost['llm_output_tokens_total']:,} out).",
            f"- Estimated per-query cost: **${cost['estimated_cost_per_query_usd']:.6f}**.",
            "",
            "Note: prototype default pricing is `0` because the answer generator "
            "falls back to grounded evidence excerpts (no LLM call). Provide a "
            "`--pricing-json` file when you wire in a paid LLM to get real numbers.",
        ]
    )
    lines.append("")

    lines.extend(
        [
            "## Repeated runs",
            "",
            "For statistical significance, run this benchmark script at least 3 times "
            "and average the p50/p95 across runs. The parse and chunk artifacts do not "
            "change between runs unless the corpus changes; retrieval/reranking do "
            "vary run-to-run and benefit from repetition.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument("--query-count", type=int, default=30, help="Number of queries for cost estimate")
    parser.add_argument("--pricing-json", type=Path, help="Optional pricing table override")
    parser.add_argument(
        "--output-name",
        default="cost_benchmark",
        help="Base filename for the report (writes .md and .json to reports/).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root: Path = args.project_root
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    pricing = dict(DEFAULT_PRICING)
    if args.pricing_json is not None:
        loaded = read_json(args.pricing_json)
        pricing.update({k: v for k, v in loaded.items() if k in pricing})

    parse = summarize_parse(root / "data" / "parsed" / "parse_report.json")
    chunks = summarize_chunks(
        {
            300: root / "data" / "chunks_300" / "chunk_report.json",
            800: root / "data" / "chunks_800" / "chunk_report.json",
        }
    )
    ingestion = summarize_ingestion(root / "data" / "vector_store" / "ingestion_report.json")
    retrieval = summarize_retrieval(root / "reports" / "retrieval_benchmark.json")
    rerank = summarize_reranking(root / "reports" / "reranking_benchmark.json")
    storage = summarize_storage(args.qdrant_url)

    cost = compute_cost_estimate(DEFAULT_TOKEN_ESTIMATE, pricing, args.query_count)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "parsing": parse,
        "chunking": chunks,
        "ingestion": ingestion,
        "retrieval": retrieval,
        "reranking": rerank,
        "storage": storage,
        "cost_estimate": cost,
    }
    output_json = reports_dir / f"{args.output_name}.json"
    output_md = reports_dir / f"{args.output_name}.md"
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {output_md.relative_to(root)} and {output_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
