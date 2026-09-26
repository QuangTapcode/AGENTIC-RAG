from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_retriever import FusedHit  # noqa: E402
from reranking.cross_encoder_reranker import CrossEncoderReranker  # noqa: E402


def fused(chunk_id: str, rank: int, text: str, document_id: str) -> FusedHit:
    return FusedHit(
        chunk_id=chunk_id,
        rank=rank,
        rrf_score=1.0 / rank,
        payload={"chunk_id": chunk_id, "document_id": document_id, "text": text},
        dense_rank=rank,
        dense_score=1.0 / rank,
        sources=["dense", "bm25"],
    )


class FakeCrossEncoder:
    def __init__(self, scores: list[float] | None = None, error: Exception | None = None) -> None:
        self.scores = scores or []
        self.error = error
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, **kwargs):
        self.pairs.extend(pairs)
        if self.error is not None:
            raise self.error
        return self.scores


class RerankingTests(unittest.TestCase):
    def test_reranker_promotes_cross_encoder_score_and_keeps_before_fields(self) -> None:
        model = FakeCrossEncoder([0.2, 0.95, 0.5])
        reranker = CrossEncoderReranker(model=model)
        candidates = [
            fused("a", 1, "A", "doc-a"),
            fused("b", 2, "B", "doc-b"),
            fused("c", 3, "C", "doc-c"),
        ]
        result = reranker.rerank("Bệnh tiểu đường", candidates, candidate_limit=3, top_k=2)
        self.assertEqual(result.status, "ok")
        self.assertEqual([hit.chunk_id for hit in result.reranked_results], ["b", "c"])
        self.assertEqual(result.reranked_results[0].before_rank, 2)
        self.assertEqual(result.reranked_results[0].before_rrf_score, 0.5)
        self.assertEqual(result.reranked_results[0].rerank_score, 0.95)
        self.assertEqual(model.pairs[0][0], "Bệnh tiểu đường")

    def test_model_failure_falls_back_to_hybrid_order(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder(error=RuntimeError("model unavailable")))
        candidates = [fused("a", 1, "A", "doc-a"), fused("b", 2, "B", "doc-b")]
        result = reranker.rerank("What is diabetes?", candidates, candidate_limit=2, top_k=2)
        self.assertEqual(result.status, "fallback")
        self.assertEqual([hit.chunk_id for hit in result.reranked_results], ["a", "b"])
        self.assertIsNone(result.reranked_results[0].rerank_score)
        self.assertIn("fallback_to_hybrid_order", result.notes)

    def test_latency_budget_falls_back_to_hybrid_order(self) -> None:
        reranker = CrossEncoderReranker(
            model=FakeCrossEncoder([0.9]),
            latency_budget_ms=1e-9,
        )
        result = reranker.rerank(
            "What is diabetes?",
            [fused("a", 1, "A", "doc-a")],
        )
        self.assertEqual(result.status, "fallback")
        self.assertIn("reranker_latency_budget_exceeded", " ".join(result.notes))

    def test_original_query_and_medicine_text_are_not_translated_or_mutated(self) -> None:
        model = FakeCrossEncoder([0.9])
        reranker = CrossEncoderReranker(model=model)
        medicine_text = "Metformin and SGLT-2 inhibitors are discussed here."
        result = reranker.rerank(
            "Tác dụng phụ của metformin và SGLT-2 là gì?",
            [fused("medicine", 1, medicine_text, "who_diabetes")],
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(model.pairs[0][0], "Tác dụng phụ của metformin và SGLT-2 là gì?")
        self.assertEqual(model.pairs[0][1], medicine_text)
        self.assertEqual(result.reranked_results[0].payload["text"], medicine_text)

    def test_empty_candidates_are_handled(self) -> None:
        reranker = CrossEncoderReranker(model=FakeCrossEncoder())
        result = reranker.rerank("What is diabetes?", [], top_k=5)
        self.assertEqual(result.status, "empty")
        self.assertEqual(result.reranked_results, [])


if __name__ == "__main__":
    unittest.main()
