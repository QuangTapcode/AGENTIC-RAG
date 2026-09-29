from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from answering.local_qwen import OllamaQwenClient  # noqa: E402


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LocalQwenTests(unittest.TestCase):
    def test_chat_returns_final_content_and_disables_thinking(self) -> None:
        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: float) -> _FakeResponse:
            captured["request"] = request
            captured["timeout"] = timeout
            return _FakeResponse(
                {"message": {"role": "assistant", "content": "Answer [CITATION:chunk-1]"}}
            )

        with patch("answering.local_qwen.urllib.request.urlopen", fake_urlopen):
            answer = OllamaQwenClient(model="qwen3:4b-instruct", timeout_s=12)("system", "user")

        request = captured["request"]
        body = json.loads(request.data.decode("utf-8"))  # type: ignore[attr-defined]
        self.assertEqual(answer, "Answer [CITATION:chunk-1]")
        self.assertEqual(body["model"], "qwen3:4b-instruct")
        self.assertFalse(body["think"])
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(captured["timeout"], 12)

    def test_thinking_tags_are_removed_before_citation_validation(self) -> None:
        with patch(
            "answering.local_qwen.urllib.request.urlopen",
            return_value=_FakeResponse(
                {
                    "message": {
                        "content": "<think>private reasoning</think>Final [CITATION:chunk-1]"
                    }
                }
            ),
        ):
            answer = OllamaQwenClient()("system", "user")

        self.assertEqual(answer, "Final [CITATION:chunk-1]")

    def test_empty_response_is_an_error(self) -> None:
        with patch(
            "answering.local_qwen.urllib.request.urlopen",
            return_value=_FakeResponse({"message": {"content": ""}}),
        ):
            with self.assertRaises(RuntimeError):
                OllamaQwenClient()("system", "user")


if __name__ == "__main__":
    unittest.main()
