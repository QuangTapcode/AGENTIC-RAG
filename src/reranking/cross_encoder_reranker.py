"""Cross-lingual cross-encoder reranking for hybrid retrieval candidates.

The reranker consumes the original user query and the English WHO chunk text.
This intentionally avoids translating the query a second time: the selected
model is multilingual, and preserving the original query prevents a bad
translation from silently changing the user's intent or a medicine name.

The normal path is hybrid top-20 -> cross-encoder -> top-5.  If model loading
or inference fails, the class returns the original hybrid order with an
explicit fallback status so the caller can observe the failure.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from sentence_transformers import CrossEncoder

from retrieval.hybrid_retriever import FusedHit, QueryBundle, build_query_bundle


RERANKER_MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
DEFAULT_CANDIDATE_K = 20
DEFAULT_OUTPUT_K = 5
DEFAULT_MAX_LENGTH = 512
DEFAULT_LATENCY_BUDGET_MS = 5000.0


@dataclass
class RerankedHit:
    """A candidate with both its pre-rerank and cross-encoder evidence."""

    chunk_id: str
    rank: int
    rerank_score: float | None
    before_rank: int
    before_rrf_score: float
    payload: dict[str, Any]
    dense_rank: int | None = None
    dense_score: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RerankingResult:
    query: QueryBundle
    model_name: str
    status: str
    candidate_count: int
    candidate_limit: int
    output_k: int
    max_length: int
    latency_budget_ms: float | None
    latency_ms: float
    candidates_before: list[FusedHit]
    reranked_results: list[RerankedHit]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query.to_dict(),
            "model_name": self.model_name,
            "status": self.status,
            "configuration": {
                "candidate_count": self.candidate_count,
                "candidate_limit": self.candidate_limit,
                "output_k": self.output_k,
                "max_length": self.max_length,
                "latency_budget_ms": self.latency_budget_ms,
            },
            "latency_ms": round(self.latency_ms, 3),
            "candidates_before": [candidate.to_dict() for candidate in self.candidates_before],
            "reranked_results": [candidate.to_dict() for candidate in self.reranked_results],
            "notes": self.notes,
        }


class CrossEncoderReranker:
    """Rerank hybrid candidates with a multilingual SentenceTransformers model."""

    def __init__(
        self,
        *,
        model_name: str = RERANKER_MODEL_NAME,
        model: Any | None = None,
        batch_size: int = 8,
        max_length: int = DEFAULT_MAX_LENGTH,
        latency_budget_ms: float | None = DEFAULT_LATENCY_BUDGET_MS,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if latency_budget_ms is not None and latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive or None")
        self.model_name = model_name
        self.model = model
        self.batch_size = batch_size
        self.max_length = max_length
        self.latency_budget_ms = latency_budget_ms

    def _get_model(self) -> Any:
        if self.model is None:
            self.model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self.model

    def preload(self) -> None:
        """Load model during application startup, outside per-query latency."""

        self._get_model()

    @staticmethod
    def _from_fused(candidate: FusedHit, *, rank: int, score: float | None) -> RerankedHit:
        return RerankedHit(
            chunk_id=candidate.chunk_id,
            rank=rank,
            rerank_score=score,
            before_rank=candidate.rank,
            before_rrf_score=candidate.rrf_score,
            payload=candidate.payload,
            dense_rank=candidate.dense_rank,
            dense_score=candidate.dense_score,
            bm25_rank=candidate.bm25_rank,
            bm25_score=candidate.bm25_score,
            sources=list(candidate.sources),
        )

    def _fallback(
        self,
        candidates: Sequence[FusedHit],
        *,
        top_k: int,
    ) -> list[RerankedHit]:
        return [
            self._from_fused(candidate, rank=rank, score=None)
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]

    def rerank(
        self,
        query: str | QueryBundle,
        candidates: Sequence[FusedHit],
        *,
        candidate_limit: int = DEFAULT_CANDIDATE_K,
        top_k: int = DEFAULT_OUTPUT_K,
    ) -> RerankingResult:
        if candidate_limit < 1:
            raise ValueError("candidate_limit must be positive")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        query_bundle = query if isinstance(query, QueryBundle) else build_query_bundle(query)
        selected_candidates = list(candidates[:candidate_limit])
        started = time.perf_counter()
        notes = [
            "reranker_query_source=original_query",
            f"cross_encoder_max_length={self.max_length}",
        ]
        if not selected_candidates:
            return RerankingResult(
                query=query_bundle,
                model_name=self.model_name,
                status="empty",
                candidate_count=0,
                candidate_limit=candidate_limit,
                output_k=top_k,
                max_length=self.max_length,
                latency_budget_ms=self.latency_budget_ms,
                latency_ms=0.0,
                candidates_before=[],
                reranked_results=[],
                notes=notes,
            )

        pairs = [
            (query_bundle.original_query, str(candidate.payload.get("text", "")))
            for candidate in selected_candidates
        ]
        try:
            scores = self._get_model().predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            score_values = [float(score) for score in scores]
            if len(score_values) != len(selected_candidates):
                raise ValueError(
                    f"reranker returned {len(score_values)} scores for {len(selected_candidates)} candidates"
                )
        except Exception as exc:  # Model/network/runtime failures must not stop retrieval.
            notes.append(f"reranker_failure={type(exc).__name__}")
            notes.append("fallback_to_hybrid_order")
            return RerankingResult(
                query=query_bundle,
                model_name=self.model_name,
                status="fallback",
                candidate_count=len(selected_candidates),
                candidate_limit=candidate_limit,
                output_k=top_k,
                max_length=self.max_length,
                latency_budget_ms=self.latency_budget_ms,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                candidates_before=selected_candidates,
                reranked_results=self._fallback(selected_candidates, top_k=top_k),
                notes=notes,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if self.latency_budget_ms is not None and elapsed_ms > self.latency_budget_ms:
            notes.append(
                f"reranker_latency_budget_exceeded={self.latency_budget_ms:g}ms"
            )
            notes.append("fallback_to_hybrid_order")
            return RerankingResult(
                query=query_bundle,
                model_name=self.model_name,
                status="fallback",
                candidate_count=len(selected_candidates),
                candidate_limit=candidate_limit,
                output_k=top_k,
                max_length=self.max_length,
                latency_budget_ms=self.latency_budget_ms,
                latency_ms=elapsed_ms,
                candidates_before=selected_candidates,
                reranked_results=self._fallback(selected_candidates, top_k=top_k),
                notes=notes,
            )

        scored = list(zip(selected_candidates, score_values))
        scored.sort(key=lambda pair: (-pair[1], pair[0].rank, pair[0].chunk_id))
        reranked = [
            self._from_fused(candidate, rank=rank, score=score)
            for rank, (candidate, score) in enumerate(scored[:top_k], start=1)
        ]
        return RerankingResult(
            query=query_bundle,
            model_name=self.model_name,
            status="ok",
            candidate_count=len(selected_candidates),
            candidate_limit=candidate_limit,
            output_k=top_k,
            max_length=self.max_length,
            latency_budget_ms=self.latency_budget_ms,
            latency_ms=elapsed_ms,
            candidates_before=selected_candidates,
            reranked_results=reranked,
            notes=notes,
        )
