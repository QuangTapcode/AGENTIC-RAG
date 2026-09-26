"""Tools that can terminate the RAG pipeline before retrieval."""

from __future__ import annotations

from typing import Any


def reject_out_of_scope(
    query: str,
    *,
    reason: str = "no_medical_signal",
    detected_language: str = "vi",
) -> dict[str, Any]:
    """Return a terminal, user-facing tool result for an unrelated query."""

    if detected_language == "en":
        message = (
            "This question is outside the scope of the medical knowledge base. "
            "Please ask about a disease, symptom, medicine, diagnosis, treatment, "
            "prevention, or public-health topic."
        )
    else:
        message = (
            "Câu hỏi này không thuộc phạm vi cơ sở tri thức y tế. "
            "Bạn hãy hỏi về bệnh, triệu chứng, thuốc, chẩn đoán, điều trị, "
            "phòng ngừa hoặc chủ đề sức khỏe cộng đồng."
        )
    return {
        "tool_name": "reject_out_of_scope",
        "status": "rejected",
        "terminal": True,
        "query": query,
        "reason": reason,
        "detected_language": detected_language,
        "message": message,
    }
