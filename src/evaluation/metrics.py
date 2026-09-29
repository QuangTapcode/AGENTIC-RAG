"""Metric implementations for router, retrieval, answer and citation evaluation.

Each metric here is deliberately deterministic and does not depend on an LLM
judge. When an LLM judge is added later it can plug in through
``AnswerQualityJudge``; the lexical proxies below let the report be reproduced
in an offline environment (no network, no LLM key).
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence


TOKEN_RE = re.compile(r"(?u)\w+")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def tokenize(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(_normalize(text))]


# --------------------------------------------------------------------------- #
# Router metrics                                                              #
# --------------------------------------------------------------------------- #


def router_confusion(
    predictions: Sequence[str],
    expected: Sequence[str],
    labels: Sequence[str],
) -> dict[str, dict[str, int]]:
    """Confusion matrix keyed as ``matrix[expected_label][predicted_label]``."""

    if len(predictions) != len(expected):
        raise ValueError("predictions and expected must be the same length")
    matrix = {label: {other: 0 for other in labels} for label in labels}
    for pred, true in zip(predictions, expected):
        if true not in matrix:
            raise ValueError(f"unknown expected label: {true!r}")
        if pred not in matrix[true]:
            matrix[true][pred] = 0
        matrix[true][pred] = matrix[true].get(pred, 0) + 1
    return matrix


def per_class_prf(
    matrix: Mapping[str, Mapping[str, int]],
    label: str,
) -> dict[str, float]:
    """Precision / recall / F1 for a single class from a confusion matrix."""

    tp = matrix.get(label, {}).get(label, 0)
    fn = sum(count for other, count in matrix.get(label, {}).items() if other != label)
    fp = sum(
        counts.get(label, 0)
        for other_label, counts in matrix.items()
        if other_label != label
    )
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def router_summary(
    matrix: Mapping[str, Mapping[str, int]],
    labels: Sequence[str],
) -> dict[str, Any]:
    per_class = {label: per_class_prf(matrix, label) for label in labels}
    total = sum(sum(counts.values()) for counts in matrix.values())
    correct = sum(matrix[label].get(label, 0) for label in labels)
    accuracy = correct / total if total else 0.0
    macro_f1 = sum(per_class[label]["f1"] for label in labels) / max(len(labels), 1)
    supports = {label: sum(matrix[label].values()) for label in labels}
    weighted_f1 = (
        sum(per_class[label]["f1"] * supports[label] for label in labels) / total
        if total
        else 0.0
    )
    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class": per_class,
        "support": supports,
        "confusion": {label: dict(counts) for label, counts in matrix.items()},
    }


# --------------------------------------------------------------------------- #
# Retrieval metrics                                                           #
# --------------------------------------------------------------------------- #


def recall_at_k(retrieved_ids: Sequence[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids[:k] if cid in expected_ids)
    return hits / len(expected_ids)


def precision_at_k(retrieved_ids: Sequence[str], expected_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not retrieved_ids:
        return 0.0
    top = retrieved_ids[:k]
    hits = sum(1 for cid in top if cid in expected_ids)
    return hits / len(top)


def reciprocal_rank(retrieved_ids: Sequence[str], expected_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in expected_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[str], expected_ids: set[str], k: int) -> float:
    if not expected_ids:
        return 0.0
    dcg = 0.0
    for rank, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in expected_ids:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(expected_ids), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def summarize_retrieval(
    per_query: Sequence[Mapping[str, Any]],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for k in ks:
        recall_values = [row["recall_at_k"].get(k, 0.0) for row in per_query]
        precision_values = [row["precision_at_k"].get(k, 0.0) for row in per_query]
        ndcg_values = [row["ndcg_at_k"].get(k, 0.0) for row in per_query]
        metrics[f"recall@{k}"] = round(_mean(recall_values), 4)
        metrics[f"precision@{k}"] = round(_mean(precision_values), 4)
        metrics[f"ndcg@{k}"] = round(_mean(ndcg_values), 4)
    metrics["mrr"] = round(_mean([row["mrr"] for row in per_query]), 4)

    has_document = any("document_recall_at_k" in row for row in per_query)
    if has_document:
        for k in ks:
            doc_recall = [row.get("document_recall_at_k", {}).get(k, 0.0) for row in per_query]
            doc_precision = [row.get("document_precision_at_k", {}).get(k, 0.0) for row in per_query]
            doc_ndcg = [row.get("document_ndcg_at_k", {}).get(k, 0.0) for row in per_query]
            metrics[f"document_recall@{k}"] = round(_mean(doc_recall), 4)
            metrics[f"document_precision@{k}"] = round(_mean(doc_precision), 4)
            metrics[f"document_ndcg@{k}"] = round(_mean(doc_ndcg), 4)
        metrics["document_mrr"] = round(
            _mean([row.get("document_mrr", 0.0) for row in per_query]),
            4,
        )
    return metrics


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def evaluate_retrieval_row(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: set[str],
    ks: Sequence[int] = (1, 3, 5, 10),
    *,
    retrieved_document_ids: Sequence[str] | None = None,
    expected_document_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compute retrieval metrics at chunk level and (optionally) document level.

    Chunk-level metrics are strict but only meaningful when the retrieved
    chunk size matches the eval-set's anchor chunk size. Document-level
    metrics let two different chunk sizes be compared fairly, so we report
    both when the caller supplies the document ids.
    """

    row: dict[str, Any] = {
        "recall_at_k": {k: recall_at_k(retrieved_chunk_ids, expected_chunk_ids, k) for k in ks},
        "precision_at_k": {k: precision_at_k(retrieved_chunk_ids, expected_chunk_ids, k) for k in ks},
        "ndcg_at_k": {k: ndcg_at_k(retrieved_chunk_ids, expected_chunk_ids, k) for k in ks},
        "mrr": reciprocal_rank(retrieved_chunk_ids, expected_chunk_ids),
    }
    if retrieved_document_ids is not None and expected_document_ids is not None:
        # Retrieved documents may repeat (many chunks per doc); dedupe preserving
        # rank order so document-level recall is a proper hit-rate, not the
        # duplicate-inflated chunk count.
        unique_documents: list[str] = []
        seen: set[str] = set()
        for did in retrieved_document_ids:
            if did and did not in seen:
                unique_documents.append(did)
                seen.add(did)
        row["document_recall_at_k"] = {
            k: recall_at_k(unique_documents, expected_document_ids, k) for k in ks
        }
        row["document_precision_at_k"] = {
            k: precision_at_k(unique_documents, expected_document_ids, k) for k in ks
        }
        row["document_ndcg_at_k"] = {
            k: ndcg_at_k(unique_documents, expected_document_ids, k) for k in ks
        }
        row["document_mrr"] = reciprocal_rank(unique_documents, expected_document_ids)
    return row


# --------------------------------------------------------------------------- #
# Answer quality metrics (lexical proxies)                                    #
# --------------------------------------------------------------------------- #


def token_f1(prediction: str, reference: str) -> dict[str, float]:
    """Token-level F1, precision, recall between prediction and reference.

    This is a lexical proxy. A real "answer correctness" measurement should be
    done by an LLM judge; the harness allows plugging one in later. The proxy
    still catches gross failures like empty answers, off-topic answers, or
    answers that hallucinate keywords absent from the reference.
    """

    pred_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)
    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = Counter(pred_tokens) & Counter(ref_tokens)
    matches = sum(common.values())
    if matches == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def context_relevance(
    retrieved_chunk_ids: Sequence[str],
    expected_chunk_ids: set[str],
    expected_document_ids: set[str] | None = None,
    context_docs: Sequence[str] | None = None,
) -> dict[str, float]:
    """Fraction of retrieved context that is expected-chunk or expected-doc level."""

    if not retrieved_chunk_ids:
        return {"chunk_precision": 0.0, "document_precision": 0.0}
    chunk_hits = sum(1 for cid in retrieved_chunk_ids if cid in expected_chunk_ids)
    doc_hits = 0
    if expected_document_ids is not None and context_docs is not None:
        doc_hits = sum(1 for did in context_docs if did in expected_document_ids)
    return {
        "chunk_precision": round(chunk_hits / len(retrieved_chunk_ids), 4),
        "document_precision": round(doc_hits / len(retrieved_chunk_ids), 4) if context_docs else 0.0,
    }


def citation_scores(
    used_citation_chunk_ids: Sequence[str],
    expected_chunk_id: str | None,
    expected_document_id: str | None,
    citation_documents: Sequence[str] | None = None,
) -> dict[str, float]:
    """Citation precision and recall against the expected source anchor.

    Precision: fraction of shown citations that match the expected document.
    Recall: whether the expected chunk (or, if missing, expected document) is
    among the citations shown to the user.
    """

    if not used_citation_chunk_ids:
        return {"precision": 0.0, "recall": 0.0}
    documents = list(citation_documents or [])
    if not documents:
        return {"precision": 0.0, "recall": 0.0}
    if expected_document_id is None:
        return {"precision": 0.0, "recall": 0.0}
    precision_hits = sum(1 for did in documents if did == expected_document_id)
    precision = precision_hits / len(documents)
    if expected_chunk_id and expected_chunk_id in used_citation_chunk_ids:
        recall = 1.0
    elif expected_document_id in documents:
        recall = 1.0
    else:
        recall = 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4)}


def refusal_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute correctness of refusals for must_refuse=True vs answered rows."""

    total_must_refuse = sum(1 for row in rows if row.get("must_refuse"))
    refused_correctly = sum(
        1
        for row in rows
        if row.get("must_refuse") and row.get("was_refused")
    )
    total_answerable = sum(1 for row in rows if not row.get("must_refuse"))
    wrongly_refused = sum(
        1
        for row in rows
        if not row.get("must_refuse") and row.get("was_refused")
    )
    return {
        "refusal_recall_on_must_refuse": round(
            refused_correctly / total_must_refuse, 4
        )
        if total_must_refuse
        else 0.0,
        "over_refusal_rate": round(
            wrongly_refused / total_answerable, 4
        )
        if total_answerable
        else 0.0,
        "counts": {
            "must_refuse": total_must_refuse,
            "answerable": total_answerable,
            "refused_correctly": refused_correctly,
            "wrongly_refused": wrongly_refused,
        },
    }
