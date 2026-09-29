from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from retrieval.hybrid_retriever import (  # noqa: E402
    DiseaseCatalog,
    MedicalQueryTranslator,
    RetrievalHit,
    build_query_bundle,
    rrf_fuse,
)


class HybridRetrievalTests(unittest.TestCase):
    def test_disease_anchor_is_extracted_before_keyword_retrieval(self) -> None:
        catalog = DiseaseCatalog.from_manifest(PROJECT_ROOT / "data" / "manifest.json")
        bundle = build_query_bundle(
            "Bệnh tiểu đường có những triệu chứng gì?",
            disease_catalog=catalog,
        )

        self.assertEqual(bundle.disease_document_ids, ["who_diabetes"])
        self.assertEqual(bundle.disease_titles, ["Diabetes"])
        self.assertIn("diabetes", bundle.disease_terms)

    def test_disease_scope_filter_targets_only_matched_documents(self) -> None:
        catalog = DiseaseCatalog.from_manifest(PROJECT_ROOT / "data" / "manifest.json")
        bundle = build_query_bundle(
            "What are the symptoms of diabetes?",
            disease_catalog=catalog,
        )

        self.assertEqual(bundle.disease_document_ids, ["who_diabetes"])
        self.assertEqual(
            bundle.document_scope_filter,
            {"document_id": {"$in": ["who_diabetes"]}},
        )

    def test_vietnamese_query_bundle_keeps_original_and_translates_terms(self) -> None:
        bundle = build_query_bundle("Bệnh tiểu đường có triệu chứng gì?")
        self.assertEqual(bundle.original_query, "Bệnh tiểu đường có triệu chứng gì?")
        self.assertEqual(bundle.detected_language, "vi")
        self.assertEqual(bundle.translation_method, "dictionary")
        self.assertIn("diabetes", bundle.translated_query_en)
        self.assertIn("symptoms", bundle.translated_query_en)

    def test_english_query_is_not_translated(self) -> None:
        bundle = build_query_bundle("What are symptoms of asthma?")
        self.assertEqual(bundle.detected_language, "en")
        self.assertEqual(bundle.translation_method, "identity_en")
        self.assertEqual(bundle.translated_query_en, "what are symptoms of asthma?")

    def test_external_translator_is_observable(self) -> None:
        translator = MedicalQueryTranslator(external_translator=lambda _: "diabetes symptoms")
        bundle = build_query_bundle("Một câu hỏi y tế", translator=translator)
        self.assertEqual(bundle.translation_method, "external_translator")
        self.assertEqual(bundle.translated_query_en, "diabetes symptoms")

    def test_rrf_preserves_ranks_scores_and_promotes_overlap(self) -> None:
        dense = [
            RetrievalHit("chunk-a", 1, 0.90, "dense", {"document_id": "doc-a"}),
            RetrievalHit("chunk-b", 2, 0.80, "dense", {"document_id": "doc-b"}),
        ]
        sparse = [
            RetrievalHit("chunk-b", 1, 3.20, "bm25", {"document_id": "doc-b"}),
            RetrievalHit("chunk-c", 2, 2.10, "bm25", {"document_id": "doc-c"}),
        ]
        fused = rrf_fuse(dense, sparse, top_k=3, rrf_k=60)
        self.assertEqual([item.chunk_id for item in fused], ["chunk-b", "chunk-a", "chunk-c"])
        self.assertEqual(fused[0].dense_rank, 2)
        self.assertEqual(fused[0].bm25_rank, 1)
        self.assertEqual(fused[0].dense_score, 0.80)
        self.assertEqual(fused[0].bm25_score, 3.20)
        self.assertEqual(fused[0].sources, ["dense", "bm25"])
        self.assertEqual([item.rank for item in fused], [1, 2, 3])

    def test_rrf_rejects_invalid_top_k(self) -> None:
        with self.assertRaises(ValueError):
            rrf_fuse([], [], top_k=0)


if __name__ == "__main__":
    unittest.main()
