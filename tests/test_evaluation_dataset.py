from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.validate_questions import validate_questions  # noqa: E402


class EvaluationDatasetTests(unittest.TestCase):
    def test_questions_have_required_coverage_and_valid_source_anchors(self) -> None:
        report = validate_questions(
            PROJECT_ROOT / "eval" / "questions.jsonl",
            PROJECT_ROOT / "data" / "chunks_300" / "chunks.jsonl",
        )
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["question_count"], 30)
        self.assertEqual(report["category_counts"]["in_scope_answerable"], 20)
        self.assertEqual(report["category_counts"]["in_scope_insufficient_corpus"], 5)
        self.assertEqual(report["category_counts"]["out_of_scope"], 5)
        self.assertEqual(report["answerable_language_counts"], {"vi": 15, "en": 5})
        self.assertEqual(report["source_anchors_validated"], 20)


if __name__ == "__main__":
    unittest.main()
