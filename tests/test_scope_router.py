from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scope_router.scope_router import ScopeRouter  # noqa: E402


class ScopeRouterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.router = ScopeRouter(manifest_path=PROJECT_ROOT / "data" / "manifest.json")

    def test_vietnamese_medical_query_is_in_scope(self) -> None:
        result = self.router.route("Bệnh tiểu đường có những triệu chứng gì?")
        self.assertEqual(result["action"], "retrieve")
        self.assertFalse(result["terminal"])
        self.assertEqual(result["classification"]["detected_language"], "vi")

    def test_english_medical_query_is_in_scope(self) -> None:
        result = self.router.route("What are the symptoms of asthma?")
        self.assertEqual(result["action"], "retrieve")

    def test_out_of_scope_query_calls_terminal_tool(self) -> None:
        result = self.router.route("Viết code Python để sắp xếp danh sách")
        self.assertEqual(result["action"], "tool_call")
        self.assertTrue(result["terminal"])
        self.assertEqual(result["tool_result"]["tool_name"], "reject_out_of_scope")

    def test_ambiguous_query_requests_clarification(self) -> None:
        result = self.router.route("Bạn có thể giúp tôi không?")
        self.assertEqual(result["action"], "clarify")
        self.assertTrue(result["terminal"])

    def test_optional_classifier_can_resolve_ambiguous_query(self) -> None:
        result = self.router.route(
            "Tôi muốn hỏi về vấn đề này",
            llm_classifier=lambda _: {"decision": "in_scope", "confidence": 0.91},
        )
        self.assertEqual(result["action"], "retrieve")
        self.assertTrue(result["classification"]["classifier_used"])

    def test_feedback_logs_only_misclassification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "router.jsonl"
            router = ScopeRouter(log_path=log_path)
            self.assertTrue(
                router.log_feedback(
                    "Viết code Python",
                    expected_decision="in_scope",
                    root_cause="test fixture",
                    fix="review medical allowlist",
                    lesson_learned="Do not let a generic programming query enter retrieval.",
                )
            )
            events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[0]["event"], "router_misclassification")
            self.assertEqual(events[0]["actual_decision"], "out_of_scope")


if __name__ == "__main__":
    unittest.main()
