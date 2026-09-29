"""Unified evaluation harness for the medical Agentic RAG pipeline.

The harness ties together the router, retrieval, reranking, and answer
generation modules against the Part 9 evaluation dataset
(``eval/questions.jsonl``). It supports three modes:

* ``router-only`` — offline router scoring (no Qdrant, no models). Always
  runnable in CI / laptops without GPU.
* ``full`` — end-to-end: router → hybrid retrieval → optional reranking →
  answer generation. Requires a running Qdrant instance and the embedding
  model. If ``ANTHROPIC_API_KEY`` or a similar hook is not present, the
  answer generator falls back to grounded evidence excerpts (still valid
  for citation, correctness proxies).
* ``from-trace`` — recompute metrics from a saved traces JSON file (no
  network/model access needed). Useful for reproducing a report from an
  archived run.

Metrics are documented in :mod:`src.evaluation.metrics`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

try:  # Allow both ``python -m src.evaluation.run_evaluation`` and direct exec.
    from evaluation.metrics import (
        citation_scores,
        context_relevance,
        evaluate_retrieval_row,
        refusal_metrics,
        router_confusion,
        router_summary,
        summarize_retrieval,
        token_f1,
    )
    from scope_router.scope_router import ScopeRouter
except ImportError:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from evaluation.metrics import (  # type: ignore[no-redef]
        citation_scores,
        context_relevance,
        evaluate_retrieval_row,
        refusal_metrics,
        router_confusion,
        router_summary,
        summarize_retrieval,
        token_f1,
    )
    from scope_router.scope_router import ScopeRouter  # type: ignore[no-redef]


ROUTER_LABELS = ("in_scope", "out_of_scope", "clarify")
DEFAULT_KS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# Dataset helpers                                                             #
# --------------------------------------------------------------------------- #


def read_questions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    if not rows:
        raise ValueError(f"No questions in {path}")
    return rows


def expected_router_label(row: Mapping[str, Any]) -> str:
    """Map the dataset category onto the router's decision space."""

    category = row.get("category")
    if category == "out_of_scope":
        return "out_of_scope"
    return "in_scope"  # answerable + in_scope_insufficient_corpus


def expected_chunk_ids(row: Mapping[str, Any]) -> set[str]:
    source = row.get("expected_source") or {}
    chunk_id = source.get("chunk_id") if isinstance(source, Mapping) else None
    return {chunk_id} if chunk_id else set()


def expected_document_ids(row: Mapping[str, Any]) -> set[str]:
    source = row.get("expected_source") or {}
    document_id = source.get("document_id") if isinstance(source, Mapping) else None
    return {document_id} if document_id else set()


# --------------------------------------------------------------------------- #
# Router-only evaluation                                                      #
# --------------------------------------------------------------------------- #


def evaluate_router(
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path | None,
    *,
    clarify_as: str = "in_scope",
) -> dict[str, Any]:
    """Score the rule-based router against the eval set.

    ``clarify_as`` decides how ``clarify`` decisions are folded into the
    binary in-scope/out-of-scope truth. The dataset does not have a
    ``clarify`` truth class; treating ``clarify`` as ``in_scope`` reflects
    the routing contract: a ``clarify`` outcome does not refuse the query,
    it asks a follow-up question, i.e. does not incur an out-of-scope
    refusal cost. Both perspectives are reported.
    """

    router = ScopeRouter(manifest_path=manifest_path)
    predictions: list[str] = []
    predictions_folded: list[str] = []
    expected: list[str] = []
    per_query: list[dict[str, Any]] = []
    for row in rows:
        classification = router.classify(row["question"])
        expected_label = expected_router_label(row)
        raw = classification.decision
        folded = clarify_as if raw == "clarify" else raw
        predictions.append(raw)
        predictions_folded.append(folded)
        expected.append(expected_label)
        per_query.append(
            {
                "id": row.get("id"),
                "question": row.get("question"),
                "language": row.get("language"),
                "category": row.get("category"),
                "expected": expected_label,
                "prediction_raw": raw,
                "prediction_folded": folded,
                "confidence": classification.confidence,
                "reasons": classification.reasons,
                "matched_medical_terms": classification.matched_medical_terms,
                "matched_out_of_scope_terms": classification.matched_out_of_scope_terms,
                "detected_language": classification.detected_language,
            }
        )

    labels_binary = ("in_scope", "out_of_scope")
    labels_three = ROUTER_LABELS
    binary_matrix = router_confusion(predictions_folded, expected, labels_binary)
    three_matrix = router_confusion(
        predictions,
        [label if label != "clarify" else "in_scope" for label in expected],
        labels_three,
    )
    summary_binary = router_summary(binary_matrix, labels_binary)
    summary_three = router_summary(three_matrix, labels_three)

    total_out_of_scope = sum(1 for row in rows if expected_router_label(row) == "out_of_scope")
    blocked = sum(
        1
        for row, pred in zip(rows, predictions_folded)
        if expected_router_label(row) == "out_of_scope" and pred == "out_of_scope"
    )
    return {
        "expected_label_source": "questions.jsonl.category",
        "clarify_folded_as": clarify_as,
        "summary_binary": summary_binary,
        "summary_three_way": summary_three,
        "safety": {
            "out_of_scope_blocked_rate": round(
                blocked / total_out_of_scope, 4
            )
            if total_out_of_scope
            else 0.0,
            "out_of_scope_total": total_out_of_scope,
            "out_of_scope_blocked": blocked,
        },
        "per_query": per_query,
    }


# --------------------------------------------------------------------------- #
# Full pipeline evaluation                                                    #
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    name: str
    chunk_size: int
    retriever_mode: str  # dense_only, bm25_only, hybrid
    use_reranker: bool = False
    top_k_dense: int = 20
    top_k_bm25: int = 20
    top_k_fused: int = 20
    top_k_output: int = 5
    rerank_candidate_k: int = 20

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "chunk_size": self.chunk_size,
            "retriever_mode": self.retriever_mode,
            "use_reranker": self.use_reranker,
            "top_k_dense": self.top_k_dense,
            "top_k_bm25": self.top_k_bm25,
            "top_k_fused": self.top_k_fused,
            "top_k_output": self.top_k_output,
            "rerank_candidate_k": self.rerank_candidate_k,
        }


DEFAULT_CONFIGS: tuple[PipelineConfig, ...] = (
    PipelineConfig("dense_chunk300", chunk_size=300, retriever_mode="dense_only"),
    PipelineConfig("dense_chunk800", chunk_size=800, retriever_mode="dense_only"),
    PipelineConfig("hybrid_chunk300", chunk_size=300, retriever_mode="hybrid"),
    PipelineConfig("hybrid_chunk800", chunk_size=800, retriever_mode="hybrid"),
    PipelineConfig(
        "hybrid_rerank_chunk300",
        chunk_size=300,
        retriever_mode="hybrid",
        use_reranker=True,
    ),
)


def _select_retrieved(
    result: Any,
    *,
    mode: str,
    top_k_output: int,
) -> list[Any]:
    if mode == "dense_only":
        hits = list(result.dense_results[:top_k_output])
    elif mode == "bm25_only":
        hits = list(result.sparse_results[:top_k_output])
    else:
        hits = list(result.fused_results[:top_k_output])
    return hits


def run_pipeline_configuration(
    config: PipelineConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    qdrant_url: str,
    embedding_model: str,
    reranker_model_name: str | None,
    llm_hook: Callable[[str, str], str] | None,
    router_manifest: Path | None,
    response_language: str = "auto",
) -> dict[str, Any]:
    """Execute one configuration against the eval set. Requires Qdrant + models."""

    # Imports happen lazily so router-only / from-trace modes stay import-clean.
    from retrieval.hybrid_retriever import HybridRetriever
    from reranking.cross_encoder_reranker import (  # noqa: WPS433
        CrossEncoderReranker,
        DEFAULT_OUTPUT_K,
    )
    from answering.answer_generator import AnswerGenerator

    bm25_artifact = (
        project_root
        / "data"
        / "vector_store"
        / f"chunks_{config.chunk_size}"
        / "bm25_index.json"
    )
    retriever = HybridRetriever(
        qdrant_url=qdrant_url,
        collection_name=f"medical_chunks_{config.chunk_size}",
        bm25_artifact_path=bm25_artifact,
        model_name=embedding_model,
    )
    reranker = None
    if config.use_reranker:
        reranker = CrossEncoderReranker(
            model_name=reranker_model_name or "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        )
        reranker.preload()
    generator = AnswerGenerator(llm=llm_hook)
    router = ScopeRouter(manifest_path=router_manifest)

    per_query_records: list[dict[str, Any]] = []
    retrieval_rows: list[dict[str, Any]] = []
    answer_rows: list[dict[str, Any]] = []
    latencies_retrieval: list[float] = []
    latencies_rerank: list[float] = []
    latencies_answer: list[float] = []

    for row in rows:
        record: dict[str, Any] = {
            "id": row.get("id"),
            "question": row.get("question"),
            "category": row.get("category"),
            "language": row.get("language"),
            "must_refuse": bool(row.get("must_refuse")),
            "expected_answer": row.get("expected_answer"),
            "expected_source": row.get("expected_source"),
        }
        classification = router.classify(row["question"])
        record["router"] = classification.to_dict()

        was_refused = False
        retrieved_ids: list[str] = []
        retrieved_documents: list[str] = []
        used_citation_chunk_ids: list[str] = []
        used_citation_documents: list[str] = []
        answer_text = ""
        answer_status = "not_run"
        failure_reason: str | None = None

        if classification.decision == "out_of_scope":
            was_refused = True
            answer_status = "router_refused"
        elif classification.decision == "clarify":
            was_refused = True
            answer_status = "router_clarify"
        else:
            retrieval_start = time.perf_counter()
            result = retriever.retrieve(
                row["question"],
                top_k_dense=config.top_k_dense,
                top_k_bm25=config.top_k_bm25,
                top_k_fused=config.top_k_fused,
            )
            latencies_retrieval.append((time.perf_counter() - retrieval_start) * 1000.0)

            hits = _select_retrieved(
                result,
                mode=config.retriever_mode,
                top_k_output=config.top_k_output,
            )
            if config.use_reranker and reranker is not None and hits:
                rerank_start = time.perf_counter()
                reranked = reranker.rerank(
                    row["question"],
                    result.fused_results,
                    candidate_limit=config.rerank_candidate_k,
                    top_k=config.top_k_output,
                )
                latencies_rerank.append((time.perf_counter() - rerank_start) * 1000.0)
                hits = reranked.reranked_results
                record["reranker"] = {
                    "status": reranked.status,
                    "notes": reranked.notes,
                    "latency_ms": reranked.latency_ms,
                }

            retrieved_ids = [str(hit.payload.get("chunk_id", hit.chunk_id)) for hit in hits]
            retrieved_documents = [
                str(hit.payload.get("document_id", "")) for hit in hits
            ]

            context_for_answer = [{"payload": hit.payload} for hit in hits]
            answer_start = time.perf_counter()
            answer = generator.generate(
                row["question"],
                context_for_answer,
                response_language=response_language,
            )
            latencies_answer.append((time.perf_counter() - answer_start) * 1000.0)
            answer_text = answer.answer
            answer_status = answer.status
            failure_reason = answer.failure_reason
            used_citation_chunk_ids = [citation.chunk_id for citation in answer.citations]
            used_citation_documents = [citation.document_id for citation in answer.citations]
            was_refused = answer.status in {"insufficient_evidence", "citation_error"}

        expected_chunks = expected_chunk_ids(row)
        expected_docs = expected_document_ids(row)
        retrieval_row = evaluate_retrieval_row(
            retrieved_ids,
            expected_chunks,
            DEFAULT_KS,
            retrieved_document_ids=retrieved_documents,
            expected_document_ids=expected_docs,
        )
        retrieval_row["context_relevance"] = context_relevance(
            retrieved_ids,
            expected_chunks,
            expected_document_ids=expected_docs,
            context_docs=retrieved_documents,
        )
        retrieval_rows.append(retrieval_row)
        citations = citation_scores(
            used_citation_chunk_ids=used_citation_chunk_ids,
            expected_chunk_id=(next(iter(expected_chunks)) if expected_chunks else None),
            expected_document_id=(next(iter(expected_docs)) if expected_docs else None),
            citation_documents=used_citation_documents,
        )

        record.update(
            {
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_documents": retrieved_documents,
                "answer": answer_text,
                "answer_status": answer_status,
                "failure_reason": failure_reason,
                "was_refused": was_refused,
                "used_citation_chunk_ids": used_citation_chunk_ids,
                "used_citation_documents": used_citation_documents,
                "retrieval_metrics": retrieval_row,
                "citation_metrics": citations,
                "answer_lexical": token_f1(answer_text, row.get("expected_answer") or ""),
            }
        )
        per_query_records.append(record)
        answer_rows.append(
            {
                "id": row.get("id"),
                "must_refuse": bool(row.get("must_refuse")),
                "was_refused": was_refused,
                "answer_status": answer_status,
                "answer_lexical": record["answer_lexical"],
                "citation_metrics": citations,
                "category": row.get("category"),
            }
        )

    # Restrict retrieval-metric aggregation to rows that have an expected chunk.
    answerable_retrieval_rows = [
        row
        for row, source in zip(retrieval_rows, rows)
        if expected_chunk_ids(source)
    ]
    retrieval_summary = summarize_retrieval(answerable_retrieval_rows, DEFAULT_KS)
    citation_summary = {
        "citation_precision_mean": round(
            statistics.fmean(
                [row["citation_metrics"]["precision"] for row in per_query_records if not row["must_refuse"]]
                or [0.0]
            ),
            4,
        ),
        "citation_recall_mean": round(
            statistics.fmean(
                [row["citation_metrics"]["recall"] for row in per_query_records if not row["must_refuse"]]
                or [0.0]
            ),
            4,
        ),
    }
    lexical_summary = {
        "answer_token_f1_mean": round(
            statistics.fmean(
                [
                    row["answer_lexical"]["f1"]
                    for row in per_query_records
                    if not row["must_refuse"]
                ]
                or [0.0]
            ),
            4,
        ),
    }
    refusal = refusal_metrics(answer_rows)
    latency_summary = {
        "retrieval_ms": _percentiles(latencies_retrieval),
        "rerank_ms": _percentiles(latencies_rerank),
        "answer_ms": _percentiles(latencies_answer),
    }
    return {
        "configuration": config.as_dict(),
        "retrieval": retrieval_summary,
        "citation": citation_summary,
        "answer_lexical": lexical_summary,
        "refusal": refusal,
        "latency": latency_summary,
        "per_query": per_query_records,
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("router-only", "full", "from-trace"),
        default="router-only",
        help="Which evaluation mode to run.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=project_root / "eval" / "questions.jsonl",
    )
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "manifest.json")
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
    )
    parser.add_argument(
        "--reranker-model",
        default=os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"),
    )
    parser.add_argument(
        "--local-qwen",
        action="store_true",
        help="Use local Qwen through Ollama for generated answers in full mode",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
    )
    parser.add_argument(
        "--answer-language",
        choices=("auto", "vi", "en", "bilingual"),
        default="auto",
        help="Answer language used in full evaluation",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=project_root / "reports",
    )
    parser.add_argument(
        "--trace-input",
        type=Path,
        help="Trace JSON file to reload for --mode from-trace",
    )
    parser.add_argument(
        "--output-name",
        default="evaluation_report",
        help="Base name for the emitted report files (.md and .json).",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args(argv)


def _markdown_router(section: dict[str, Any]) -> list[str]:
    binary = section["summary_binary"]
    three = section["summary_three_way"]
    safety = section["safety"]
    lines = [
        "## Router",
        "",
        f"Expected labels are derived from `questions.jsonl.category` "
        f"(`out_of_scope` → out; everything else → in-scope; `clarify` folded as `{section['clarify_folded_as']}`).",
        "",
        "### Binary (in-scope / out-of-scope)",
        "",
        f"- Accuracy: **{binary['accuracy']:.2%}**",
        f"- Macro F1: **{binary['macro_f1']:.4f}**",
        f"- Weighted F1: **{binary['weighted_f1']:.4f}**",
        f"- Out-of-scope blocked: **{safety['out_of_scope_blocked']}/{safety['out_of_scope_total']}** "
        f"({safety['out_of_scope_blocked_rate']:.0%})",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, stats in binary["per_class"].items():
        support = binary["support"].get(label, 0)
        lines.append(
            f"| {label} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {support} |"
        )
    lines.extend(
        [
            "",
            "### Three-way (with `clarify` visible)",
            "",
            f"- Accuracy: **{three['accuracy']:.2%}**",
            f"- Macro F1: **{three['macro_f1']:.4f}**",
            "",
            "| Class | Precision | Recall | F1 | Support |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label, stats in three["per_class"].items():
        support = three["support"].get(label, 0)
        lines.append(
            f"| {label} | {stats['precision']:.4f} | {stats['recall']:.4f} | {stats['f1']:.4f} | {support} |"
        )
    return lines


def _markdown_config_comparison(configs: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "## Configuration comparison — document level",
        "",
        "Document-level Recall/MRR is the fair cross-config metric: `expected_source.chunk_id` in the eval set is anchored on chunks_300, so chunk-level Recall on chunks_800 is not comparable. Document-level uses `expected_source.document_id`.",
        "",
        "| Config | Doc Recall@5 | Doc Recall@10 | Doc MRR | Doc NDCG@5 | Citation P | Citation R | Ans token F1 | Refuse (must) | Over-refuse |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in configs:
        r = entry["retrieval"]
        c = entry["citation"]
        a = entry["answer_lexical"]
        f = entry["refusal"]
        lines.append(
            f"| `{entry['configuration']['name']}` | "
            f"{r.get('document_recall@5', 0):.2f} | {r.get('document_recall@10', 0):.2f} | "
            f"{r.get('document_mrr', 0):.2f} | {r.get('document_ndcg@5', 0):.2f} | "
            f"{c.get('citation_precision_mean', 0):.2f} | {c.get('citation_recall_mean', 0):.2f} | "
            f"{a.get('answer_token_f1_mean', 0):.2f} | "
            f"{f.get('refusal_recall_on_must_refuse', 0):.2f} | "
            f"{f.get('over_refusal_rate', 0):.2f} |"
        )
    lines.extend([
        "",
        "### Chunk-level (chunks_300 configs only — strict)",
        "",
        "| Config | Recall@5 | Recall@10 | MRR | NDCG@5 |",
        "|---|---:|---:|---:|---:|",
    ])
    for entry in configs:
        if entry["configuration"]["chunk_size"] != 300:
            continue
        r = entry["retrieval"]
        lines.append(
            f"| `{entry['configuration']['name']}` | "
            f"{r.get('recall@5', 0):.2f} | {r.get('recall@10', 0):.2f} | "
            f"{r.get('mrr', 0):.2f} | {r.get('ndcg@5', 0):.2f} |"
        )
    return lines


def _markdown_latency(configs: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "## Latency (ms) — per query",
        "",
        "| Config | Retrieval p50 | Retrieval p95 | Rerank p50 | Rerank p95 | Answer p50 | Answer p95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in configs:
        lat = entry["latency"]
        lines.append(
            f"| `{entry['configuration']['name']}` | "
            f"{lat['retrieval_ms']['p50']:.1f} | {lat['retrieval_ms']['p95']:.1f} | "
            f"{lat['rerank_ms']['p50']:.1f} | {lat['rerank_ms']['p95']:.1f} | "
            f"{lat['answer_ms']['p50']:.1f} | {lat['answer_ms']['p95']:.1f} |"
        )
    return lines


def render_markdown(
    router_section: dict[str, Any] | None,
    configs: Sequence[dict[str, Any]],
    *,
    dataset_path: Path,
    generated_at: str,
    mode: str,
    notes: Sequence[str],
) -> str:
    lines: list[str] = [
        "# Evaluation Report — Agentic RAG Y tế",
        "",
        f"Generated: `{generated_at}` — mode: `{mode}` — dataset: `{dataset_path.name}`",
        "",
    ]
    if notes:
        lines.append("## Notes")
        lines.append("")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")
    if router_section is not None:
        lines.extend(_markdown_router(router_section))
        lines.append("")
    if configs:
        lines.extend(_markdown_config_comparison(configs))
        lines.append("")
        lines.extend(_markdown_latency(configs))
        lines.append("")
        lines.append("## Configurations")
        lines.append("")
        for entry in configs:
            config = entry["configuration"]
            lines.append(f"- `{config['name']}`: chunk={config['chunk_size']}, "
                         f"mode={config['retriever_mode']}, rerank={config['use_reranker']}, "
                         f"top_k_output={config['top_k_output']}")
        lines.append("")
    lines.append("## Metric definitions")
    lines.append("")
    lines.extend(
        [
            "- **Recall@k / Precision@k**: chunk-level, based on `expected_source.chunk_id`. Aggregated over answerable questions only (`in_scope_answerable`).",
            "- **MRR / NDCG@k**: same expected-chunk basis. NDCG uses binary relevance.",
            "- **Citation precision**: fraction of shown citation documents equal to expected document.",
            "- **Citation recall**: 1 if expected chunk (or fallback: expected document) appears in shown citations; 0 otherwise.",
            "- **Answer token F1**: bag-of-token F1 between generated answer and `expected_answer` (Vietnamese/English NFKC casefold). Lexical proxy for correctness — for a full correctness/faithfulness reading, an LLM judge should be added.",
            "- **Refusal recall on must-refuse**: fraction of `must_refuse=true` questions that the pipeline actually refused (out-of-scope or insufficient-evidence).",
            "- **Over-refusal rate**: fraction of answerable questions the pipeline mistakenly refused.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root: Path = args.project_root
    dataset_path = args.dataset if args.dataset.is_absolute() else project_root / args.dataset
    manifest_path = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else project_root / args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows = read_questions(dataset_path)

    generated_at = datetime.now(timezone.utc).isoformat()
    notes: list[str] = []
    router_section: dict[str, Any] | None = None
    configs: list[dict[str, Any]] = []
    llm_hook: Callable[[str, str], str] | None = None

    if args.local_qwen:
        from answering.local_qwen import OllamaQwenClient

        llm_hook = OllamaQwenClient(
            base_url=args.ollama_url,
            model=args.qwen_model,
        )
        notes.append(f"Generated answers use local Ollama model `{args.qwen_model}`.")

    if args.mode in {"router-only", "full"}:
        router_section = evaluate_router(rows, manifest_path)

    if args.mode == "full":
        for config in DEFAULT_CONFIGS:
            configs.append(
                run_pipeline_configuration(
                    config,
                    rows,
                    project_root=project_root,
                    qdrant_url=args.qdrant_url,
                    embedding_model=args.embedding_model,
                    reranker_model_name=args.reranker_model,
                    llm_hook=llm_hook,
                    response_language=args.answer_language,
                    router_manifest=manifest_path,
                )
            )
    elif args.mode == "from-trace":
        if not args.trace_input:
            raise SystemExit("--trace-input is required for --mode from-trace")
        payload = json.loads(args.trace_input.read_text(encoding="utf-8"))
        router_section = payload.get("router")
        configs = payload.get("configurations", [])
        notes.append(f"Loaded trace from {args.trace_input.name}")

    if args.mode == "router-only":
        notes.append("Router-only mode: retrieval/rerank/answer metrics are not populated. "
                     "Run with --mode full when Qdrant + models are available.")

    output_json = reports_dir / f"{args.output_name}.json"
    output_md = reports_dir / f"{args.output_name}.md"
    payload = {
        "generated_at": generated_at,
        "mode": args.mode,
        "dataset": str(dataset_path.relative_to(project_root)).replace("\\", "/"),
        "router": router_section,
        "configurations": configs,
        "notes": notes,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=list) + "\n", encoding="utf-8")
    output_md.write_text(
        render_markdown(
            router_section,
            configs,
            dataset_path=dataset_path,
            generated_at=generated_at,
            mode=args.mode,
            notes=notes,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {output_md.relative_to(project_root)} and {output_json.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
