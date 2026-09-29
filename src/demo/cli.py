"""Interactive CLI demo for the medical Agentic RAG prototype.

Usage examples::

    # Interactive mode (asks for a query line by line)
    python src/demo/cli.py

    # Single query
    python src/demo/cli.py --query "Bệnh tiểu đường có triệu chứng gì?"

    # Scripted demo: run the three showcase queries in one go
    python src/demo/cli.py --scenario all

The demo shows a full trace: router decision, retrieval hits with score,
optional rerank, and the grounded answer with citations. It runs even when
Qdrant / models are not available — in that case retrieval and rerank are
reported as ``skipped`` and only the router path is shown, which is enough to
demonstrate the out-of-scope refusal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


try:
    from scope_router.scope_router import ScopeRouter
except ImportError:
    source_root = Path(__file__).resolve().parents[1]
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    from scope_router.scope_router import ScopeRouter  # type: ignore[no-redef]


SCENARIOS = {
    "in_scope": (
        "Bệnh tiểu đường có những triệu chứng gì?",
        "In-scope: expect answered with WHO diabetes citation.",
    ),
    "out_of_scope": (
        "Hôm nay thời tiết Hà Nội thế nào?",
        "Out-of-scope: expect router to call reject_out_of_scope and stop.",
    ),
    "insufficient": (
        "Tôi bị dị ứng thuốc X hiếm, có nên đổi sang thuốc Y không?",
        "Insufficient evidence: no chunk covers this; expect refusal with warning.",
    ),
}


@dataclass
class Colors:
    header = "\033[95m"
    ok = "\033[92m"
    warn = "\033[93m"
    err = "\033[91m"
    dim = "\033[2m"
    bold = "\033[1m"
    end = "\033[0m"

    @classmethod
    def strip(cls) -> None:
        for name in ("header", "ok", "warn", "err", "dim", "bold", "end"):
            setattr(cls, name, "")


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print_header(title: str) -> None:
    line = "=" * max(len(title), 40)
    print(f"\n{Colors.header}{Colors.bold}{title}{Colors.end}")
    print(f"{Colors.dim}{line}{Colors.end}")


def _print_router(classification: Any) -> None:
    print(f"{Colors.bold}Router{Colors.end}: decision={classification.decision} "
          f"confidence={classification.confidence:.2f} "
          f"lang={classification.detected_language}")
    if classification.matched_medical_terms:
        print(f"  {Colors.dim}medical_terms:{Colors.end} "
              f"{', '.join(classification.matched_medical_terms)}")
    if classification.matched_out_of_scope_terms:
        print(f"  {Colors.dim}out_of_scope_terms:{Colors.end} "
              f"{', '.join(classification.matched_out_of_scope_terms)}")
    if classification.reasons:
        print(f"  {Colors.dim}reasons:{Colors.end} {', '.join(classification.reasons)}")


def _try_run_pipeline(
    query: str,
    *,
    chunk_size: int,
    qdrant_url: str,
    embedding_model: str,
    use_reranker: bool,
    top_k_fused: int,
    top_k_output: int,
    use_local_qwen: bool,
    ollama_url: str,
    qwen_model: str,
    response_language: str,
    project_root: Path,
) -> tuple[bool, dict[str, Any]]:
    """Run retrieval + optional rerank + answer. Returns (ok, payload)."""

    try:
        from retrieval.hybrid_retriever import HybridRetriever
        from answering.answer_generator import AnswerGenerator
    except ImportError as exc:
        return False, {"reason": f"module_missing: {exc}"}
    try:
        retriever = HybridRetriever(
            qdrant_url=qdrant_url,
            collection_name=f"medical_chunks_{chunk_size}",
            bm25_artifact_path=project_root / "data" / "vector_store"
            / f"chunks_{chunk_size}" / "bm25_index.json",
            model_name=embedding_model,
        )
        result = retriever.retrieve(
            query,
            top_k_dense=20,
            top_k_bm25=20,
            top_k_fused=top_k_fused,
        )
    except Exception as exc:
        return False, {"reason": f"{type(exc).__name__}: {exc}"}

    hits = list(result.fused_results[:top_k_output])
    rerank_note = None
    if use_reranker and hits:
        try:
            from reranking.cross_encoder_reranker import CrossEncoderReranker

            reranker = CrossEncoderReranker()
            # Load the model outside the per-query latency budget; otherwise
            # the first call trips the 5s budget on cold start and falls back.
            reranker.preload()
            reranked = reranker.rerank(query, result.fused_results, top_k=top_k_output)
            hits = reranked.reranked_results
            rerank_note = reranked.status
        except Exception as exc:
            rerank_note = f"rerank_failed:{type(exc).__name__}"

    if use_local_qwen:
        from answering.local_qwen import OllamaQwenClient

        generator = AnswerGenerator(
            llm=OllamaQwenClient(base_url=ollama_url, model=qwen_model)
        )
    else:
        generator = AnswerGenerator(llm=None)  # fallback path is grounded excerpts
    answer = generator.generate(
        query,
        [{"payload": hit.payload} for hit in hits],
        response_language=response_language,
    )
    return True, {
        "hits": [
            {
                "chunk_id": str(hit.payload.get("chunk_id", getattr(hit, "chunk_id", ""))),
                "document_id": str(hit.payload.get("document_id", "")),
                "section": str(hit.payload.get("section", "")),
                "score": getattr(hit, "rrf_score", getattr(hit, "rerank_score", None)),
            }
            for hit in hits
        ],
        "rerank_status": rerank_note,
        "answer": answer.to_dict(),
        "retrieval_latency_ms": result.latency_ms,
    }


def _print_pipeline(payload: dict[str, Any]) -> None:
    hits = payload.get("hits", [])
    if hits:
        print(f"\n{Colors.bold}Retrieved{Colors.end} (top {len(hits)}):")
        for i, hit in enumerate(hits, start=1):
            score = hit.get("score")
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
            print(f"  {i}. {hit['document_id']} · {hit['section'] or 'unspecified'} "
                  f"(chunk={hit['chunk_id']}, score={score_str})")
    if payload.get("rerank_status"):
        print(f"  {Colors.dim}rerank status:{Colors.end} {payload['rerank_status']}")

    answer = payload.get("answer") or {}
    print(f"\n{Colors.bold}Answer{Colors.end}: status={answer.get('status')} "
          f"used_llm={answer.get('used_llm')} context_count={answer.get('context_count')}")
    if answer.get("failure_reason"):
        print(f"  {Colors.warn}failure_reason:{Colors.end} {answer['failure_reason']}")
    body = answer.get("answer") or ""
    for line in body.splitlines():
        print(f"  {line}")
    if answer.get("citations"):
        print(f"\n{Colors.bold}Citations{Colors.end}:")
        for citation in answer["citations"]:
            print(f"  {citation['display_label']}")


def run_query(
    query: str,
    *,
    args: argparse.Namespace,
    project_root: Path,
) -> dict[str, Any]:
    manifest = project_root / "data" / "manifest.json"
    router = ScopeRouter(manifest_path=manifest if manifest.exists() else None)
    classification = router.classify(query)
    _print_router(classification)

    payload: dict[str, Any] = {
        "query": query,
        "router": classification.to_dict(),
    }

    if classification.decision == "out_of_scope":
        from scope_router.scope_tools import reject_out_of_scope

        tool_result = reject_out_of_scope(
            query,
            reason=classification.reasons[0] if classification.reasons else "out_of_scope",
            detected_language=classification.detected_language,
        )
        print(f"\n{Colors.warn}Tool call:{Colors.end} reject_out_of_scope")
        print(f"  message: {tool_result.get('message')}")
        payload["tool_result"] = tool_result
        return payload

    if classification.decision == "clarify":
        message = (
            "Bạn hãy làm rõ bệnh, triệu chứng, thuốc hoặc chủ đề sức khỏe muốn hỏi."
            if classification.detected_language != "en"
            else "Please clarify which disease, symptom, medicine, or health topic you are asking about."
        )
        print(f"\n{Colors.warn}Clarify:{Colors.end} {message}")
        payload["clarify_message"] = message
        if not args.force_pipeline:
            return payload

    ok, pipeline_payload = _try_run_pipeline(
        query,
        chunk_size=args.chunk_size,
        qdrant_url=args.qdrant_url,
        embedding_model=args.embedding_model,
        use_reranker=args.rerank,
        top_k_fused=args.top_k_fused,
        top_k_output=args.top_k_output,
        use_local_qwen=args.local_qwen,
        ollama_url=args.ollama_url,
        qwen_model=args.qwen_model,
        response_language=args.answer_language,
        project_root=project_root,
    )
    if not ok:
        print(f"\n{Colors.err}Pipeline skipped:{Colors.end} {pipeline_payload.get('reason')}")
        print(f"  {Colors.dim}Bring Qdrant up with `docker compose up qdrant` and install "
              f"requirements.txt to enable retrieval + answering.{Colors.end}")
        payload["pipeline"] = {"status": "skipped", **pipeline_payload}
        return payload

    _print_pipeline(pipeline_payload)
    payload["pipeline"] = {"status": "ok", **pipeline_payload}
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = _project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="Single query to run and exit")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS) + ["all"],
        help="Run one of the built-in showcase scenarios and exit",
    )
    parser.add_argument("--chunk-size", type=int, choices=(300, 800), default=300)
    parser.add_argument("--qdrant-url", default=os.getenv("QDRANT_URL", "http://localhost:6333"))
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
    )
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument(
        "--local-qwen",
        action="store_true",
        help="Generate the grounded answer with a local Qwen model served by Ollama",
    )
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    )
    parser.add_argument(
        "--qwen-model",
        default=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
    )
    parser.add_argument(
        "--answer-language",
        choices=("auto", "vi", "en", "bilingual"),
        default="auto",
        help="Answer language: auto, vi, en, or bilingual",
    )
    parser.add_argument("--top-k-fused", type=int, default=20)
    parser.add_argument("--top-k-output", type=int, default=5)
    parser.add_argument("--json-out", type=Path, help="Also write structured trace as JSON")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument(
        "--force-pipeline",
        action="store_true",
        help="Run retrieval even when the router says clarify (useful for debugging)",
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.no_color or os.environ.get("NO_COLOR"):
        Colors.strip()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    project_root: Path = args.project_root
    traces: list[dict[str, Any]] = []

    if args.scenario:
        scenarios = SCENARIOS.items() if args.scenario == "all" else [(args.scenario, SCENARIOS[args.scenario])]
        for name, (query, description) in scenarios:
            _print_header(f"Scenario: {name}")
            print(f"{Colors.dim}{description}{Colors.end}")
            print(f"{Colors.bold}Query{Colors.end}: {query}")
            traces.append({"scenario": name, **run_query(query, args=args, project_root=project_root)})
    elif args.query:
        _print_header("Query")
        print(f"{Colors.bold}Query{Colors.end}: {args.query}")
        traces.append(run_query(args.query, args=args, project_root=project_root))
    else:
        print(f"{Colors.dim}Interactive mode — type 'quit' to exit.{Colors.end}")
        while True:
            try:
                query = input(f"\n{Colors.bold}query> {Colors.end}").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not query or query.lower() in {"quit", "exit"}:
                break
            _print_header("Query")
            print(f"{Colors.bold}Query{Colors.end}: {query}")
            traces.append(run_query(query, args=args, project_root=project_root))

    if args.json_out and traces:
        args.json_out.write_text(
            json.dumps(traces, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(f"\nTrace saved to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
