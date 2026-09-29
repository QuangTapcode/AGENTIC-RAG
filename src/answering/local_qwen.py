"""Local Qwen answer generation through the Ollama HTTP API.

The client deliberately returns plain text and leaves grounding/citation
validation to :class:`answering.answer_generator.AnswerGenerator`. If Ollama
is unavailable or returns an invalid payload, the caller's existing fallback
path remains responsible for the user-facing response.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_QWEN_MODEL = "qwen3:4b-instruct"
DEFAULT_TIMEOUT_S = 120.0
DEFAULT_TEMPERATURE = 0.1
# Bilingual answers need room for both concise sections. The prompt still
# limits each section to five bullets so this is a ceiling, not a target.
DEFAULT_NUM_PREDICT = 768

THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def _remove_thinking_blocks(text: str) -> str:
    """Remove Qwen reasoning tags before AnswerGenerator validates citations."""

    return THINK_BLOCK_PATTERN.sub("", text).strip()


class OllamaQwenClient:
    """Callable ``(system_prompt, user_prompt) -> answer`` adapter for Qwen."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OLLAMA_BASE_URL,
        model: str = DEFAULT_QWEN_MODEL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        temperature: float = DEFAULT_TEMPERATURE,
        num_predict: int = DEFAULT_NUM_PREDICT,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url must not be empty")
        if not model.strip():
            raise ValueError("model must not be empty")
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if num_predict < 1:
            raise ValueError("num_predict must be positive")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.num_predict = num_predict

    def __call__(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Qwen3 supports a thinking mode; disable it so only the grounded
            # answer is passed into citation validation and rendered to users.
            "think": False,
            "keep_alive": "10m",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Ollama is unavailable at {self.base_url}: {exc.reason}") from exc

        try:
            result: dict[str, Any] = json.loads(raw)
            content = str((result.get("message") or {}).get("content") or "")
        except (TypeError, ValueError, AttributeError) as exc:
            raise RuntimeError("Ollama returned invalid JSON") from exc
        content = _remove_thinking_blocks(content)
        if not content:
            raise RuntimeError("Ollama returned an empty answer")
        return content
