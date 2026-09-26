"""Create section-aware 300-token and 800-token chunk datasets.

The input is the structured JSON produced by ``src/parser/parse_documents.py``.
Chunks are first grouped by Markdown heading/section and then packed greedily
using a fixed tokenizer. Tables, fenced blocks, and consecutive lists are
treated as atomic blocks whenever they fit inside the configured budget.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import tiktoken


TOKENIZER_NAME = "cl100k_base"
HEADING_RE = re.compile(r"^(?P<marks>#{1,6})[ \t]+(?P<title>.*?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")
LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


@dataclass(frozen=True)
class Block:
    """An indivisible Markdown unit unless it exceeds the token budget."""

    text: str
    kind: str
    line_start: int
    line_end: int


@dataclass(frozen=True)
class Section:
    """A heading section and its source line range."""

    name: str
    heading: str | None
    heading_level: int
    heading_line: int | None
    lines: tuple[tuple[int, str], ...]


def token_count(encoder: tiktoken.Encoding, text: str) -> int:
    """Count tokens without interpreting special strings as control tokens."""

    return len(encoder.encode(text, disallowed_special=()))


def _is_blank(line: str) -> bool:
    return not line.strip()


def _is_heading(line: str) -> bool:
    return bool(HEADING_RE.match(line))


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.count("|") >= 2


def _is_list_start(line: str) -> bool:
    return bool(LIST_RE.match(line))


def _normalise_heading(match: re.Match[str]) -> str:
    title = match.group("title").strip()
    return f"{match.group('marks')} {title}".rstrip()


def split_sections(content: str) -> list[Section]:
    """Split Markdown into heading sections while preserving source lines."""

    sections: list[Section] = []
    current_lines: list[tuple[int, str]] = []
    current_name = "Preamble"
    current_heading: str | None = None
    current_level = 0
    current_heading_line: int | None = None
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal current_lines
        if current_lines or current_heading is not None:
            sections.append(
                Section(
                    name=current_name,
                    heading=current_heading,
                    heading_level=current_level,
                    heading_line=current_heading_line,
                    lines=tuple(current_lines),
                )
            )
        current_lines = []

    for line_number, line in enumerate(content.splitlines(), start=1):
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group("marks"))
            title = match.group("title").strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append((level, title))
            current_name = " > ".join(title for _, title in heading_stack)
            current_heading = _normalise_heading(match)
            current_level = level
            current_heading_line = line_number
            continue
        current_lines.append((line_number, line))

    flush()
    return sections


def split_blocks(lines: Iterable[tuple[int, str]]) -> list[Block]:
    """Extract paragraphs, lists, tables, and fenced blocks from a section."""

    items = list(lines)
    blocks: list[Block] = []
    index = 0

    while index < len(items):
        line_number, line = items[index]
        if _is_blank(line):
            index += 1
            continue

        fence = FENCE_RE.match(line)
        if fence:
            start = index
            marker = fence.group(1)[0]
            index += 1
            while index < len(items):
                if items[index][1].lstrip().startswith(marker * 3):
                    index += 1
                    break
                index += 1
            block_lines = [value for _, value in items[start:index]]
            blocks.append(Block("\n".join(block_lines).strip(), "fenced_block", line_number, items[index - 1][0]))
            continue

        if _is_table_line(line):
            start = index
            index += 1
            while index < len(items) and (_is_table_line(items[index][1]) or _is_blank(items[index][1])):
                index += 1
            selected = [(number, value) for number, value in items[start:index] if not _is_blank(value)]
            blocks.append(
                Block(
                    "\n".join(value for _, value in selected).strip(),
                    "table",
                    selected[0][0],
                    selected[-1][0],
                )
            )
            continue

        if _is_list_start(line):
            start = index
            index += 1
            while index < len(items):
                next_line = items[index][1]
                if _is_blank(next_line) or _is_heading(next_line) or FENCE_RE.match(next_line) or _is_table_line(next_line):
                    break
                index += 1
            block_lines = [value for _, value in items[start:index]]
            blocks.append(Block("\n".join(block_lines).strip(), "list", line_number, items[index - 1][0]))
            continue

        start = index
        index += 1
        while index < len(items):
            next_line = items[index][1]
            if (
                _is_blank(next_line)
                or _is_heading(next_line)
                or FENCE_RE.match(next_line)
                or _is_table_line(next_line)
                or _is_list_start(next_line)
            ):
                break
            index += 1
        block_lines = [value for _, value in items[start:index]]
        blocks.append(Block("\n".join(block_lines).strip(), "paragraph", line_number, items[index - 1][0]))

    return [block for block in blocks if block.text]


def split_list_items(block: Block) -> list[Block]:
    """Split a list block at item boundaries, preserving continuation lines."""

    lines = block.text.splitlines()
    item_groups: list[list[str]] = []
    for line in lines:
        if _is_list_start(line):
            item_groups.append([line])
        elif item_groups:
            item_groups[-1].append(line)
        else:
            item_groups.append([line])
    if len(item_groups) <= 1:
        return [block]
    return [
        Block("\n".join(group).strip(), "list_item", block.line_start, block.line_end)
        for group in item_groups
        if "\n".join(group).strip()
    ]


def render_text(heading_prefix: str, blocks: list[Block]) -> str:
    """Render a chunk exactly as it will be written to JSONL."""

    return "\n\n".join(part for part in (heading_prefix, *(block.text for block in blocks)) if part)


def split_block_to_fit(
    encoder: tiktoken.Encoding,
    text: str,
    *,
    heading_prefix: str,
    chunk_size: int,
    overlap_target: int,
) -> list[tuple[str, int]]:
    """Split an atomic block using the exact rendered chunk budget."""

    tokens = encoder.encode(text, disallowed_special=())
    parts: list[tuple[str, int]] = []
    start = 0
    previous_end: int | None = None
    while start < len(tokens):
        low, high = 1, len(tokens) - start
        best = 0
        while low <= high:
            middle = (low + high) // 2
            candidate = encoder.decode(tokens[start : start + middle]).strip()
            candidate_tokens = token_count(encoder, render_text(heading_prefix, [Block(candidate, "split", 1, 1)]))
            if candidate_tokens <= chunk_size:
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == 0:
            raise ValueError("Unable to split an atomic block within the chunk budget")
        end = start + best
        overlap_tokens = 0 if previous_end is None else previous_end - start
        parts.append((encoder.decode(tokens[start:end]).strip(), overlap_tokens))
        if end >= len(tokens):
            break
        previous_end = end
        start = max(start + 1, end - overlap_target)
    return parts


def _suffix_for_overlap(
    encoder: tiktoken.Encoding,
    blocks: list[Block],
    overlap_target: int,
    heading_prefix: str,
    chunk_size: int,
) -> tuple[list[Block], int]:
    """Return the largest contiguous suffix that fits the overlap target."""

    suffix: list[Block] = []
    total = 0
    for block in reversed(blocks):
        block_tokens = token_count(encoder, block.text)
        if total + block_tokens > overlap_target:
            break
        suffix.insert(0, block)
        total += block_tokens
    while suffix and token_count(encoder, render_text(heading_prefix, suffix)) > chunk_size:
        removed = suffix.pop(0)
        total -= token_count(encoder, removed.text)
    return suffix, total


def _build_chunk(
    *,
    document: dict[str, Any],
    section: Section,
    blocks: list[Block],
    heading_prefix: str,
    chunk_size: int,
    overlap_target: int,
    overlap_tokens: int,
    quality_flags: set[str],
    encoder: tiktoken.Encoding,
) -> dict[str, Any]:
    text = render_text(heading_prefix, blocks)
    source_start = section.heading_line or min(block.line_start for block in blocks)
    source_end = max(block.line_end for block in blocks)
    return {
        "document_id": document["document_id"],
        "title": document["title"],
        "topic": document["topic"],
        "source_url": document["source_url"],
        "source_language": document["source_language"],
        "published_date": document.get("published_date"),
        "chunk_size": chunk_size,
        "overlap_target_tokens": overlap_target,
        "overlap_tokens": overlap_tokens,
        "token_count": token_count(encoder, text),
        "section": section.name,
        "heading_level": section.heading_level,
        "page_start": None,
        "page_end": None,
        "source_line_start": source_start,
        "source_line_end": source_end,
        "block_kinds": list(dict.fromkeys(block.kind for block in blocks)),
        "quality_flags": sorted(quality_flags),
        "text": text,
    }


def chunk_section(
    document: dict[str, Any],
    section: Section,
    *,
    chunk_size: int,
    overlap_target: int,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    """Pack blocks in one section, carrying a bounded suffix as overlap."""

    blocks = split_blocks(section.lines)
    if not blocks:
        return []

    heading_prefix = section.heading or ""
    chunks: list[dict[str, Any]] = []
    current: list[Block] = []
    current_has_new_content = False
    current_overlap_tokens = 0

    def emit_current() -> None:
        nonlocal current, current_has_new_content, current_overlap_tokens
        if not current or not current_has_new_content:
            return
        chunks.append(
            _build_chunk(
                document=document,
                section=section,
                blocks=current,
                heading_prefix=heading_prefix,
                chunk_size=chunk_size,
                overlap_target=overlap_target,
                overlap_tokens=current_overlap_tokens,
                quality_flags=set(),
                encoder=encoder,
            )
        )
        suffix, suffix_tokens = _suffix_for_overlap(
            encoder, current, overlap_target, heading_prefix, chunk_size
        )
        current = suffix
        current_overlap_tokens = suffix_tokens
        current_has_new_content = False

    pending_blocks = list(blocks)
    while pending_blocks:
        block = pending_blocks.pop(0)
        if token_count(encoder, render_text(heading_prefix, [block])) > chunk_size:
            if block.kind == "list":
                item_blocks = split_list_items(block)
                if len(item_blocks) > 1:
                    pending_blocks = item_blocks + pending_blocks
                    continue
            emit_current()
            # A carried suffix that cannot hold this block is not emitted by itself.
            current = []
            current_overlap_tokens = 0
            current_has_new_content = False
            for part, split_overlap_tokens in split_block_to_fit(
                encoder,
                block.text,
                heading_prefix=heading_prefix,
                chunk_size=chunk_size,
                overlap_target=overlap_target,
            ):
                split_block = Block(part, f"{block.kind}_split", block.line_start, block.line_end)
                chunks.append(
                    _build_chunk(
                        document=document,
                        section=section,
                        blocks=[split_block],
                        heading_prefix=heading_prefix,
                        chunk_size=chunk_size,
                        overlap_target=overlap_target,
                        overlap_tokens=min(split_overlap_tokens, overlap_target),
                        quality_flags={"oversized_atomic_block"},
                        encoder=encoder,
                    )
                )
            continue

        if current and token_count(encoder, render_text(heading_prefix, [*current, block])) > chunk_size:
            if current_has_new_content:
                emit_current()
            else:
                # Do not create a chunk containing overlap only.
                current = []
                current_overlap_tokens = 0
                current_has_new_content = False

        current.append(block)
        current_has_new_content = True

    emit_current()
    return chunks


def chunk_document(
    document: dict[str, Any],
    *,
    chunk_size: int,
    overlap_target: int,
    encoder: tiktoken.Encoding,
) -> list[dict[str, Any]]:
    sections = split_sections(document["content"])
    chunks: list[dict[str, Any]] = []
    for section in sections:
        chunks.extend(
            chunk_section(
                document,
                section,
                chunk_size=chunk_size,
                overlap_target=overlap_target,
                encoder=encoder,
            )
        )
    return chunks


def load_document(manifest_item: dict[str, Any], parsed_dir: Path) -> dict[str, Any]:
    parsed_path = parsed_dir / f"{manifest_item['document_id']}.json"
    parsed = json.loads(parsed_path.read_text(encoding="utf-8"))
    return {
        "document_id": manifest_item["document_id"],
        "title": manifest_item["title"],
        "topic": manifest_item["topic"],
        "source_url": manifest_item["source_url"],
        "source_language": parsed.get("source_language", "en"),
        "published_date": manifest_item.get("published_date"),
        "content": parsed["content"],
    }


def _statistics(values: list[int]) -> dict[str, float | int]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0, "median": 0}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 2),
        "median": median(values),
    }


def build_dataset(
    *,
    manifest_path: Path,
    parsed_dir: Path,
    output_root: Path,
    chunk_size: int,
    overlap_target: int,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoder = tiktoken.get_encoding(TOKENIZER_NAME)
    all_chunks: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for item in manifest["documents"]:
        try:
            document = load_document(item, parsed_dir)
            all_chunks.extend(
                chunk_document(
                    document,
                    chunk_size=chunk_size,
                    overlap_target=overlap_target,
                    encoder=encoder,
                )
            )
        except Exception as exc:  # pragma: no cover - surfaced in the report
            failures.append({"document_id": item["document_id"], "error": str(exc)})

    output_dir = output_root / f"chunks_{chunk_size}"
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, chunk in enumerate(all_chunks, start=1):
        chunk["chunk_id"] = f"{chunk['document_id']}__{chunk_size}__{index:05d}"

    jsonl_path = output_dir / "chunks.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in all_chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False, separators=(",", ":")) + "\n")

    token_values = [chunk["token_count"] for chunk in all_chunks]
    overlap_values = [chunk["overlap_tokens"] for chunk in all_chunks if chunk["overlap_tokens"] > 0]
    flag_counts = Counter(flag for chunk in all_chunks for flag in chunk["quality_flags"])
    document_stats: dict[str, Any] = {}
    for document_id in sorted({chunk["document_id"] for chunk in all_chunks}):
        doc_chunks = [chunk for chunk in all_chunks if chunk["document_id"] == document_id]
        document_stats[document_id] = {
            "chunk_count": len(doc_chunks),
            "token_count": sum(chunk["token_count"] for chunk in doc_chunks),
            "token_statistics": _statistics([chunk["token_count"] for chunk in doc_chunks]),
        }

    report = {
        "dataset_id": manifest.get("dataset_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "chunk_size": chunk_size,
        "overlap_target_tokens": overlap_target,
        "tokenizer": TOKENIZER_NAME,
        "input": {
            "manifest": str(manifest_path).replace("\\", "/"),
            "parsed_directory": str(parsed_dir).replace("\\", "/"),
        },
        "policy": {
            "section_first": True,
            "atomic_blocks": ["table", "fenced_block", "list", "paragraph"],
            "oversized_atomic_block_policy": "token_split_with_quality_flag",
            "page_metadata": "null_for_markdown_inputs",
        },
        "lessons_learned": [
            "Token count must include heading and Markdown separators; checking only the sum of block tokens can exceed the budget.",
            "Long lists should be split at item boundaries before tokenizer splitting so medical instruction items remain intact.",
            "Section-first chunking preserves context but can produce short chunks when a section is smaller than the target budget.",
        ],
        "summary": {
            "expected_documents": len(manifest["documents"]),
            "documents_with_chunks": len(document_stats),
            "total_chunks": len(all_chunks),
            "total_tokens": sum(token_values),
            "token_statistics": _statistics(token_values),
            "chunks_with_overlap": len(overlap_values),
            "overlap_statistics": _statistics(overlap_values),
            "section_count": len({(chunk["document_id"], chunk["section"]) for chunk in all_chunks}),
            "quality_flag_counts": dict(sorted(flag_counts.items())),
            "over_limit_chunks": sum(chunk["token_count"] > chunk_size for chunk in all_chunks),
            "failure_count": len(failures),
        },
        "documents": document_stats,
        "failures": failures,
    }
    (output_dir / "chunk_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=project_root / "data" / "manifest.json")
    parser.add_argument("--parsed-dir", type=Path, default=project_root / "data" / "parsed")
    parser.add_argument("--output-root", type=Path, default=project_root / "data")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configurations = ((300, 60), (800, 120))
    summaries: dict[str, Any] = {}
    for chunk_size, overlap_target in configurations:
        summaries[str(chunk_size)] = build_dataset(
            manifest_path=args.manifest,
            parsed_dir=args.parsed_dir,
            output_root=args.output_root,
            chunk_size=chunk_size,
            overlap_target=overlap_target,
        )["summary"]
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    return 0 if all(summary["over_limit_chunks"] == 0 and summary["failure_count"] == 0 for summary in summaries.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
