"""Parse the medical corpus with Kreuzberg and emit structured JSON records."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kreuzberg import ExtractionConfig, OutputFormat, extract_file_sync


def jsonable(value: Any) -> Any:
    """Convert Kreuzberg values to JSON-safe values without losing diagnostics."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return jsonable(vars(value))
    return str(value)


def extract_headings(text: str) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.match(r"^\s*(#{1,6})\s+(.+?)\s*$", line)
        if match:
            headings.append(
                {
                    "level": len(match.group(1)),
                    "text": match.group(2).strip(),
                    "line": line_number,
                }
            )
    return headings


def extract_markdown_tables(text: str) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    current: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            tables.append({"rows": [row.strip() for row in current]})
            current = []

    for line in text.splitlines():
        if re.match(r"^\s*\|.*\|\s*$", line):
            current.append(line.strip())
        else:
            flush()
    flush()
    return tables


def extract_source_headers(text: str) -> dict[str, str]:
    headers: dict[str, str] = {}
    patterns = {
        "title": r"^Title:\s*(.+?)\s*$",
        "url_source": r"^URL Source:\s*(.+?)\s*$",
        "published_time": r"^Published Time:\s*(.+?)\s*$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            headers[key] = match.group(1).strip()
    return headers


def fallback_read(path: Path) -> tuple[str, str, str | None, list[Any], dict[str, Any]]:
    """Keep the document usable if Kreuzberg cannot parse one input."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(encoding="utf-8", errors="replace")
    return text, "fallback_raw_text", "Kreuzberg extraction failed; raw UTF-8 text was preserved.", [], {}


def parse_document(document: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    input_path = Path(document["raw_file"])
    started = time.perf_counter()
    error: str | None = None

    try:
        result = extract_file_sync(
            str(input_path),
            config=ExtractionConfig(output_format=OutputFormat.MARKDOWN),
        )
        content = result.content or ""
        parser_name = "kreuzberg"
        tables = jsonable(result.tables)
        parser_metadata = jsonable(result.metadata)
    except Exception as exc:  # noqa: BLE001 - batch parsing must continue per document
        content, parser_name, error, tables, parser_metadata = fallback_read(input_path)
        error = f"{error} Original error: {type(exc).__name__}: {exc}"

    quality_flags: list[str] = []
    if not content.strip():
        quality_flags.append("empty_content")
    if len(content.strip()) < 500:
        quality_flags.append("short_content")
    if "\ufffd" in content:
        quality_flags.append("replacement_character_detected")
    if parser_name != "kreuzberg":
        quality_flags.append("fallback_parser_used")

    record = {
        "document_id": document["document_id"],
        "title": document["title"],
        "topic": document["topic"],
        "source_url": document["source_url"],
        "source_language": "en",
        "published_date": document.get("published_date"),
        "input_file": str(input_path).replace("\\", "/"),
        "parser": parser_name,
        "parser_version": "4.10.4" if parser_name == "kreuzberg" else None,
        "output_format": "markdown" if parser_name == "kreuzberg" else "raw_text",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "char_count": len(content),
        "word_count": len(re.findall(r"\b\w+\b", content, flags=re.UNICODE)),
        "headings": extract_headings(content),
        "tables": tables,
        "parser_metadata": parser_metadata,
        "source_headers": extract_source_headers(content),
        "quality_flags": quality_flags,
        "error": error,
    }

    output_path = output_dir / f"{document['document_id']}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "document_id": document["document_id"],
        "status": "ok" if not error else "fallback",
        "parser": parser_name,
        "char_count": record["char_count"],
        "word_count": record["word_count"],
        "heading_count": len(record["headings"]),
        "table_count": len(tables),
        "quality_flags": quality_flags,
        "error": error,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "output_file": str(output_path).replace("\\", "/"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/parsed"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [parse_document(document, args.output_dir) for document in manifest["documents"]]
    report = {
        "dataset_id": manifest["dataset_id"],
        "parser": "kreuzberg",
        "parser_version": "4.10.4",
        "parsed_at": datetime.now(timezone.utc).isoformat(),
        "document_count": len(results),
        "success_count": sum(result["status"] == "ok" for result in results),
        "fallback_count": sum(result["status"] == "fallback" for result in results),
        "quality_flag_count": sum(bool(result["quality_flags"]) for result in results),
        "results": results,
    }
    report_path = args.output_dir / "parse_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({key: report[key] for key in (
        "document_count", "success_count", "fallback_count", "quality_flag_count"
    )}, ensure_ascii=False))
    return 0 if report["fallback_count"] == 0 and report["quality_flag_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
