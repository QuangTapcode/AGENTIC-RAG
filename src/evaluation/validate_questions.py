"""Validate the Part 9 evaluation dataset and its expected source anchors."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_CATEGORY_COUNTS = {
    "in_scope_answerable": 20,
    "in_scope_insufficient_corpus": 5,
    "out_of_scope": 5,
}
REQUIRED_FIELDS = {
    "id",
    "question",
    "language",
    "detected_language",
    "category",
    "expected_answer",
    "expected_page",
    "expected_source",
    "must_refuse",
    "generated_answer",
    "retrieved_chunks",
    "result",
    "failure_type",
}


class DatasetValidationError(ValueError):
    """Raised when the evaluation dataset violates its schema or coverage."""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetValidationError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise DatasetValidationError(f"Expected object at {path}:{line_number}")
            rows.append(item)
    return rows


def read_chunks(path: Path) -> dict[str, dict[str, Any]]:
    chunks: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(path):
        chunk_id = str(item.get("chunk_id", ""))
        if not chunk_id:
            raise DatasetValidationError(f"Chunk without chunk_id in {path}")
        chunks[chunk_id] = item
    return chunks


def validate_questions(dataset_path: Path, chunks_path: Path) -> dict[str, Any]:
    rows = read_jsonl(dataset_path)
    chunks = read_chunks(chunks_path)
    errors: list[str] = []
    ids = [str(row.get("id", "")) for row in rows]
    if len(rows) != 30:
        errors.append(f"expected 30 questions, found {len(rows)}")
    if len(set(ids)) != len(ids):
        errors.append("question ids must be unique")

    category_counts = Counter(row.get("category") for row in rows)
    for category, expected_count in EXPECTED_CATEGORY_COUNTS.items():
        if category_counts.get(category, 0) != expected_count:
            errors.append(
                f"category {category!r}: expected {expected_count}, found {category_counts.get(category, 0)}"
            )

    answerable = [row for row in rows if row.get("category") == "in_scope_answerable"]
    if sum(row.get("language") == "vi" for row in answerable) != 15:
        errors.append("answerable Vietnamese coverage must be exactly 15")
    if sum(row.get("language") == "en" for row in answerable) != 5:
        errors.append("answerable English coverage must be exactly 5")

    for index, row in enumerate(rows, start=1):
        missing = sorted(REQUIRED_FIELDS - set(row))
        if missing:
            errors.append(f"row {index} missing fields: {missing}")
            continue
        if row["language"] not in {"vi", "en"}:
            errors.append(f"{row['id']}: unsupported language {row['language']!r}")
        if row["detected_language"] != row["language"]:
            errors.append(f"{row['id']}: detected_language does not match language label")
        if not str(row["question"]).strip() or not str(row["expected_answer"]).strip():
            errors.append(f"{row['id']}: question and expected_answer must be non-empty")
        if not isinstance(row["must_refuse"], bool):
            errors.append(f"{row['id']}: must_refuse must be boolean")
        if row["generated_answer"] is not None:
            errors.append(f"{row['id']}: generated_answer must start as null")
        if row["retrieved_chunks"] != []:
            errors.append(f"{row['id']}: retrieved_chunks must start empty")
        if row["result"] is not None or row["failure_type"] is not None:
            errors.append(f"{row['id']}: result/failure_type must start null")

        is_answerable = row["category"] == "in_scope_answerable"
        source = row["expected_source"]
        if is_answerable:
            if not isinstance(source, dict):
                errors.append(f"{row['id']}: answerable query needs expected_source")
                continue
            chunk_id = source.get("chunk_id")
            chunk = chunks.get(chunk_id)
            if chunk is None:
                errors.append(f"{row['id']}: expected chunk not found: {chunk_id}")
                continue
            for field_name in ("document_id", "section", "source_url"):
                if source.get(field_name) != chunk.get(field_name):
                    errors.append(
                        f"{row['id']}: expected_source.{field_name} does not match chunk {chunk_id}"
                    )
            expected_lines = source.get("expected_source_lines")
            actual_lines = [chunk.get("source_line_start"), chunk.get("source_line_end")]
            if expected_lines != actual_lines:
                errors.append(f"{row['id']}: expected source lines {expected_lines} != {actual_lines}")
            if row["expected_page"] != chunk.get("page_start"):
                errors.append(f"{row['id']}: expected_page does not match chunk page_start")
            if row["must_refuse"]:
                errors.append(f"{row['id']}: answerable query must not be marked must_refuse")
        else:
            if source is not None:
                errors.append(f"{row['id']}: unsupported/out-of-scope query must have expected_source=null")
            if not row["must_refuse"]:
                errors.append(f"{row['id']}: unsupported/out-of-scope query must be must_refuse=true")

    if errors:
        raise DatasetValidationError("\n".join(errors))
    return {
        "question_count": len(rows),
        "category_counts": dict(category_counts),
        "answerable_language_counts": dict(
            Counter(row["language"] for row in answerable)
        ),
        "source_anchors_validated": len(answerable),
        "status": "ok",
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=project_root / "eval" / "questions.jsonl")
    parser.add_argument("--chunks", type=Path, default=project_root / "data" / "chunks_300" / "chunks.jsonl")
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root: Path = args.project_root
    dataset = args.dataset if args.dataset.is_absolute() else project_root / args.dataset
    chunks = args.chunks if args.chunks.is_absolute() else project_root / args.chunks
    report = validate_questions(dataset, chunks)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
