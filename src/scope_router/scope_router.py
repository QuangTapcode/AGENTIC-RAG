"""Scope classifier and routing contract for the medical Agentic RAG pipeline."""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .scope_tools import reject_out_of_scope
except ImportError:  # Allows ``python src/scope_router/scope_router.py ...``.
    from scope_tools import reject_out_of_scope


Decision = str
LLMClassifier = Callable[[str], Mapping[str, Any]]
VALID_DECISIONS = {"in_scope", "out_of_scope", "clarify"}

DEFAULT_MEDICAL_TERMS = {
    "bệnh",
    "bệnh lý",
    "triệu chứng",
    "dấu hiệu",
    "thuốc",
    "dược phẩm",
    "liều",
    "liều lượng",
    "tác dụng phụ",
    "chống chỉ định",
    "tương tác thuốc",
    "điều trị",
    "chẩn đoán",
    "xét nghiệm",
    "phòng ngừa",
    "phòng bệnh",
    "vắc xin",
    "vaccine",
    "sức khỏe",
    "y tế",
    "bác sĩ",
    "health",
    "medical",
    "medicine",
    "medication",
    "medications",
    "drug",
    "symptom",
    "symptoms",
    "signs",
    "disease",
    "diseases",
    "disorder",
    "diagnosis",
    "treatment",
    "therapy",
    "prevention",
    "vaccine",
    "dosage",
    "dose",
    "side effect",
    "contraindication",
    "public health",
    "who",
    # Common inflection of the manifest title "Heart attack".
    "heart attacks",
}

DEFAULT_OUT_OF_SCOPE_TERMS = {
    "lập trình",
    "viết code",
    "python",
    "javascript",
    "programming",
    "software",
    "thời tiết",
    "weather",
    "chính trị",
    "politics",
    "bóng đá",
    "football",
    "sports",
    "thể thao",
    "chứng khoán",
    "stock market",
    "tài chính",
    "finance",
    "du lịch",
    "travel",
    "công thức nấu ăn",
    "recipe",
    "phim",
    "movie",
    "âm nhạc",
    "music",
    "trò chơi",
    "game",
    "toán học",
    "mathematics",
}

VIETNAMESE_MARKERS = {
    "bệnh",
    "thuốc",
    "triệu",
    "chứng",
    "điều",
    "trị",
    "phòng",
    "ngừa",
    "tôi",
    "có",
    "gì",
    "như",
    "thế",
    "nào",
}

TERM_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def normalise_query(query: str) -> str:
    return unicodedata.normalize("NFKC", query).casefold().strip()


def detect_language(query: str) -> str:
    """Detect the response language conservatively without translating the query."""

    normalised = normalise_query(query)
    if any(character in normalised for character in "ăâđêôơưáàảãạấầẩẫậắằẳẵặ"):
        return "vi"
    words = set(re.findall(r"(?u)\b\w+\b", normalised))
    if words & VIETNAMESE_MARKERS:
        return "vi"
    if re.search(r"[a-z]", normalised):
        return "en"
    return "unknown"


def _term_pattern(term: str) -> re.Pattern[str]:
    pattern = TERM_PATTERN_CACHE.get(term)
    if pattern is None:
        pattern = re.compile(rf"(?<!\w){re.escape(term)}(?!\w)", re.IGNORECASE)
        TERM_PATTERN_CACHE[term] = pattern
    return pattern


def matching_terms(text: str, terms: set[str]) -> list[str]:
    return sorted(term for term in terms if _term_pattern(term).search(text))


def _term_weight(term: str) -> int:
    return 3 if " " in term or len(term) >= 8 else 1


@dataclass
class ScopeClassification:
    decision: Decision
    confidence: float
    detected_language: str
    medical_score: int
    out_of_scope_score: int
    matched_medical_terms: list[str] = field(default_factory=list)
    matched_out_of_scope_terms: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    classifier_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScopeRouter:
    """Route a query before retrieval, with an optional LLM override hook."""

    def __init__(
        self,
        *,
        manifest_path: Path | None = None,
        log_path: Path | None = None,
        medical_terms: set[str] | None = None,
        out_of_scope_terms: set[str] | None = None,
    ) -> None:
        self.log_path = log_path
        self.medical_terms = set(medical_terms or DEFAULT_MEDICAL_TERMS)
        self.out_of_scope_terms = set(out_of_scope_terms or DEFAULT_OUT_OF_SCOPE_TERMS)
        if manifest_path is not None:
            self._add_manifest_terms(manifest_path)

    def _add_manifest_terms(self, manifest_path: Path) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for document in manifest.get("documents", []):
            for field_name in ("title", "topic"):
                value = str(document.get(field_name, "")).strip().casefold()
                if value:
                    self.medical_terms.add(value)

    def classify(
        self,
        query: str,
        *,
        llm_classifier: LLMClassifier | None = None,
    ) -> ScopeClassification:
        if not query or not query.strip():
            return ScopeClassification(
                decision="clarify",
                confidence=0.99,
                detected_language="unknown",
                medical_score=0,
                out_of_scope_score=0,
                reasons=["empty_query"],
            )

        text = normalise_query(query)
        medical_matches = matching_terms(text, self.medical_terms)
        out_matches = matching_terms(text, self.out_of_scope_terms)
        medical_score = sum(_term_weight(term) for term in medical_matches)
        out_score = sum(_term_weight(term) for term in out_matches)
        language = detect_language(text)

        if medical_score >= 2 and medical_score > out_score:
            decision = "in_scope"
            confidence = min(0.99, 0.66 + 0.06 * medical_score)
            reasons = ["medical_terms_detected"]
        elif out_score >= 2 and out_score > medical_score:
            decision = "out_of_scope"
            confidence = min(0.99, 0.76 + 0.05 * out_score)
            reasons = ["out_of_scope_terms_detected"]
        elif medical_score and out_score:
            decision = "clarify"
            confidence = 0.42
            reasons = ["mixed_domain_signals"]
        else:
            decision = "clarify"
            confidence = 0.35
            reasons = ["insufficient_domain_signal"]

        classifier_used = False
        if llm_classifier is not None and (decision == "clarify" or confidence < 0.78):
            try:
                suggestion = llm_classifier(query)
                suggested_decision = str(
                    suggestion.get("decision", suggestion.get("label", ""))
                ).strip().casefold()
                suggested_confidence = float(suggestion.get("confidence", 0.0))
                if suggested_decision in VALID_DECISIONS and suggested_confidence >= 0.60:
                    decision = suggested_decision
                    confidence = min(0.99, suggested_confidence)
                    reasons.append("classifier_override")
                    classifier_used = True
            except (TypeError, ValueError, KeyError):
                reasons.append("classifier_error_fallback_to_rules")

        return ScopeClassification(
            decision=decision,
            confidence=round(confidence, 4),
            detected_language=language,
            medical_score=medical_score,
            out_of_scope_score=out_score,
            matched_medical_terms=medical_matches,
            matched_out_of_scope_terms=out_matches,
            reasons=reasons,
            classifier_used=classifier_used,
        )

    def route(
        self,
        query: str,
        *,
        llm_classifier: LLMClassifier | None = None,
    ) -> dict[str, Any]:
        classification = self.classify(query, llm_classifier=llm_classifier)
        if classification.decision == "out_of_scope":
            tool_result = reject_out_of_scope(
                query,
                reason=classification.reasons[0],
                detected_language=classification.detected_language,
            )
            result = {
                "action": "tool_call",
                "terminal": True,
                "classification": classification.to_dict(),
                "tool_result": tool_result,
            }
        elif classification.decision == "clarify":
            result = {
                "action": "clarify",
                "terminal": True,
                "classification": classification.to_dict(),
                "message": self._clarification_message(classification.detected_language),
            }
        else:
            result = {
                "action": "retrieve",
                "terminal": False,
                "classification": classification.to_dict(),
            }
        self._log({"event": "route", "query": query, **result})
        return result

    @staticmethod
    def _clarification_message(language: str) -> str:
        if language == "en":
            return "Please clarify which disease, symptom, medicine, or health topic you are asking about."
        return "Bạn hãy làm rõ bệnh, triệu chứng, thuốc hoặc chủ đề sức khỏe muốn hỏi."

    def log_feedback(
        self,
        query: str,
        *,
        expected_decision: Decision,
        actual_decision: Decision | None = None,
        root_cause: str = "",
        fix: str = "",
        lesson_learned: str = "",
    ) -> bool:
        actual = actual_decision or self.classify(query).decision
        misclassified = actual != expected_decision
        if misclassified:
            self._log(
                {
                    "event": "router_misclassification",
                    "query": query,
                    "expected_decision": expected_decision,
                    "actual_decision": actual,
                    "root_cause": root_cause,
                    "fix": fix,
                    "lesson_learned": lesson_learned,
                }
            )
        return misclassified

    def _log(self, event: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event,
        }
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "manifest.json")
    parser.add_argument("--log-file", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    router = ScopeRouter(manifest_path=args.manifest, log_path=args.log_file)
    print(json.dumps(router.route(args.query), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
