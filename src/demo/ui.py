"""Streamlit web UI for the medical Agentic RAG prototype.

Run with::

    py -3 -m streamlit run src/demo/ui.py

The UI wraps the same pipeline as ``src/demo/cli.py``: router → hybrid
retrieval → optional rerank → grounded answer with citations. Router
runs entirely offline; retrieval + rerank require Qdrant + models. When
those are missing the UI still shows the router path with an inline
notice — this matches the CLI's ``pipeline skipped`` behavior.
"""

from __future__ import annotations

import json
import sys
import time
import os
from pathlib import Path
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scope_router.scope_router import ScopeRouter  # noqa: E402
from scope_router.scope_tools import reject_out_of_scope  # noqa: E402


SCENARIOS = {
    "In-scope (triệu chứng tiểu đường, VI)": "Bệnh tiểu đường có những triệu chứng gì?",
    "In-scope (điều trị hen suyễn, EN)": "What treatments are available for asthma?",
    "Experiment (rerank đổi thứ hạng, EN)": "What are the causes of a heart attack?",
    "Experiment (chunk + rerank, EN)": "What are common symptoms of depression?",
    "Out-of-scope (thời tiết)": "Hôm nay thời tiết Hà Nội thế nào?",
    "Insufficient / clarify": "Tôi bị dị ứng thuốc X hiếm, có nên đổi sang thuốc Y không?",
}


def _manifest_scenarios() -> dict[str, str]:
    """Create one sample question for every document in the corpus."""

    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scenarios: dict[str, str] = {}
    for document in manifest.get("documents", []):
        document_id = str(document.get("document_id", "")).strip()
        title = str(document.get("title", "")).strip()
        if not document_id or not title:
            continue
        scenarios[f"WHO — {title} (key facts, EN)"] = (
            f"What are the key facts and prevention guidance about {title}?"
        )
    return scenarios


# Replace the two original disease-specific examples with the complete
# manifest-driven list while keeping the experiment and refusal examples.
SCENARIOS = {
    **_manifest_scenarios(),
    **{
        label: query
        for label, query in SCENARIOS.items()
        if not label.startswith("In-scope")
    },
}


st.set_page_config(
    page_title="Agentic RAG Y tế",
    page_icon="🩺",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def get_router() -> ScopeRouter:
    manifest = PROJECT_ROOT / "data" / "manifest.json"
    return ScopeRouter(manifest_path=manifest if manifest.exists() else None)


@st.cache_resource(show_spinner="Đang load embedding model + Qdrant client...")
def get_retriever(chunk_size: int, qdrant_url: str, embedding_model: str):
    from retrieval.hybrid_retriever import HybridRetriever

    return HybridRetriever(
        qdrant_url=qdrant_url,
        collection_name=f"medical_chunks_{chunk_size}",
        bm25_artifact_path=PROJECT_ROOT / "data" / "vector_store"
        / f"chunks_{chunk_size}" / "bm25_index.json",
        model_name=embedding_model,
    )


@st.cache_resource(show_spinner="Đang load cross-encoder reranker...")
def get_reranker():
    from reranking.cross_encoder_reranker import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    reranker.preload()
    return reranker


@st.cache_resource(show_spinner=False)
def get_generator(use_local_qwen: bool, ollama_url: str, qwen_model: str):
    from answering.answer_generator import AnswerGenerator

    if not use_local_qwen:
        return AnswerGenerator(llm=None)

    from answering.local_qwen import OllamaQwenClient

    return AnswerGenerator(
        llm=OllamaQwenClient(
            base_url=ollama_url,
            model=qwen_model,
        )
    )


def render_router(classification: Any) -> None:
    color_map = {"in_scope": "green", "out_of_scope": "red", "clarify": "orange"}
    decision = classification.decision
    color = color_map.get(decision, "gray")
    st.markdown(
        f"### Router → **:{color}[{decision}]** "
        f"(confidence {classification.confidence:.2f}, "
        f"ngôn ngữ `{classification.detected_language}`)"
    )
    cols = st.columns(2)
    with cols[0]:
        if classification.matched_medical_terms:
            st.caption("Medical terms đã match")
            st.write(", ".join(classification.matched_medical_terms))
    with cols[1]:
        if classification.matched_out_of_scope_terms:
            st.caption("Out-of-scope terms đã match")
            st.write(", ".join(classification.matched_out_of_scope_terms))
    if classification.reasons:
        st.caption(f"Lý do: {', '.join(classification.reasons)}")


def render_hits(hits, elapsed_ms: float | None = None) -> None:
    st.markdown(f"### Context đã retrieve — top {len(hits)}")
    if elapsed_ms is not None:
        st.caption(f"Retrieval latency: {elapsed_ms:.1f} ms")
    for i, hit in enumerate(hits, start=1):
        payload = hit.payload
        chunk_id = payload.get("chunk_id", getattr(hit, "chunk_id", ""))
        document = payload.get("document_id", "unknown")
        section = payload.get("section", "unspecified")
        rerank_score = getattr(hit, "rerank_score", None)
        score = rerank_score if rerank_score is not None else getattr(hit, "rrf_score", None)
        score_str = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        before_rank = getattr(hit, "before_rank", None)
        rank_note = f" · hybrid rank {before_rank}" if before_rank is not None else ""
        with st.expander(f"{i}. **{document}** — {section}  ·  score {score_str}{rank_note}"):
            st.caption(f"chunk_id: `{chunk_id}`")
            text = str(payload.get("text", ""))
            st.write(text[:1500] + ("…" if len(text) > 1500 else ""))


def _hit_chunk_id(hit: Any) -> str:
    payload = getattr(hit, "payload", {}) or {}
    return str(payload.get("chunk_id", getattr(hit, "chunk_id", "")))


def render_experiment_trace(result: Any, hybrid_hits: list[Any], final_hits: list[Any], rerank_result: Any | None) -> None:
    """Make retrieval/reranking changes observable even when answer text is similar."""

    output_hybrid_ids = [_hit_chunk_id(hit) for hit in hybrid_hits]
    output_final_ids = [_hit_chunk_id(hit) for hit in final_hits]
    output_moved_positions = sum(
        before_id != after_id
        for before_id, after_id in zip(output_hybrid_ids, output_final_ids)
    )
    candidate_k = min(result.top_k_fused, max(len(hybrid_hits), 5))
    candidate_hybrid_hits = list(result.fused_results[:candidate_k])
    candidate_final_hits = (
        list(rerank_result.reranked_results[:candidate_k])
        if rerank_result is not None
        else candidate_hybrid_hits
    )
    candidate_hybrid_ids = [_hit_chunk_id(hit) for hit in candidate_hybrid_hits]
    candidate_final_ids = [_hit_chunk_id(hit) for hit in candidate_final_hits]
    candidate_moved_positions = sum(
        before_id != after_id
        for before_id, after_id in zip(candidate_hybrid_ids, candidate_final_ids)
    )
    hybrid_document = (
        str((getattr(hybrid_hits[0], "payload", {}) or {}).get("document_id", ""))
        if hybrid_hits
        else ""
    )
    final_document = (
        str((getattr(final_hits[0], "payload", {}) or {}).get("document_id", ""))
        if final_hits
        else ""
    )

    st.markdown("### Experiment trace")
    st.caption(
        f"collection=`{result.collection_name}` · chunk size=`{result.collection_name.rsplit('_', 1)[-1]}` · "
        f"RRF candidates={result.top_k_fused}"
    )
    disease_titles = list(getattr(result.query, "disease_titles", []) or [])
    disease_document_ids = list(getattr(result.query, "disease_document_ids", []) or [])
    if disease_document_ids:
        st.success(
            "Disease anchor: "
            + ", ".join(disease_titles)
            + " | document filter: "
            + ", ".join(disease_document_ids)
        )
    else:
        st.info("No exact disease anchor detected; hybrid retrieval uses the full corpus.")
    if result.notes:
        st.caption("Retrieval notes: " + " | ".join(str(note) for note in result.notes))

    columns = st.columns(4)
    columns[0].metric("Hybrid top-1", hybrid_document or "—")
    columns[1].metric("Final top-1", final_document or "—")
    columns[2].metric("Output đổi vị trí", output_moved_positions)
    columns[3].metric(f"Candidate đổi vị trí (top {candidate_k})", candidate_moved_positions)

    if rerank_result is not None:
        st.caption(
            f"reranker status=`{rerank_result.status}` · candidates={rerank_result.candidate_count} · "
            f"latency={rerank_result.latency_ms:.1f} ms"
        )
    if hybrid_document and hybrid_document == final_document:
        st.info(
            "Hybrid và final vẫn cùng chọn document top-1. Với answer fallback, hệ thống trích excerpt "
            "từ cùng document nên câu chữ có thể gần giống; hãy dùng chunk ID, thứ tự, citation và metric "
            "retrieval để đánh giá khác biệt."
        )

    with st.expander("Xem thứ tự chunk trước/sau"):
        st.write(
            {
                "output_hybrid_order": output_hybrid_ids,
                "output_final_order": output_final_ids,
                "candidate_hybrid_order": candidate_hybrid_ids,
                "candidate_final_order": candidate_final_ids,
            }
        )


def render_answer(answer_dict: dict) -> None:
    status = answer_dict.get("status")
    status_color = {
        "answered": "green",
        "answered_fallback": "blue",
        "insufficient_evidence": "orange",
        "citation_error": "red",
    }.get(status, "gray")

    st.markdown(f"### Câu trả lời — :{status_color}[{status}]")
    st.caption(
        f"used_llm={answer_dict.get('used_llm')} · "
        f"context_count={answer_dict.get('context_count')} · "
        f"response_language={answer_dict.get('response_language', 'auto')}"
    )
    if answer_dict.get("failure_reason"):
        st.warning(f"LLM fallback: `{answer_dict['failure_reason']}`")
    notes = answer_dict.get("notes") or []
    if notes:
        st.caption(" · ".join(str(note) for note in notes))
    body = answer_dict.get("answer") or ""
    st.markdown(body.replace("\n", "\n\n"))

    citations = answer_dict.get("citations") or []
    if citations:
        st.markdown("### Citations (nguồn trích dẫn)")
        for citation in citations:
            with st.container(border=True):
                st.markdown(
                    f"**[{citation['source_id']}] {citation['title']}** — "
                    f"{citation['page_label']} — section *{citation['section'] or 'unspecified'}*"
                )
                if citation.get("source_url"):
                    st.markdown(f"[Xem nguồn]({citation['source_url']})")
                if citation.get("source_line_start") is not None:
                    st.caption(
                        f"chunk_id `{citation['chunk_id']}` · source lines "
                        f"{citation['source_line_start']}–{citation.get('source_line_end') or citation['source_line_start']}"
                    )


def run_full_pipeline(
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
) -> None:
    router = get_router()
    classification = router.classify(query)
    render_router(classification)

    if classification.decision == "out_of_scope":
        tool_result = reject_out_of_scope(
            query,
            reason=classification.reasons[0] if classification.reasons else "out_of_scope",
            detected_language=classification.detected_language,
        )
        st.warning(f"🚫 **Đã gọi tool**: `reject_out_of_scope`\n\n{tool_result.get('message')}")
        return

    if classification.decision == "clarify":
        message = (
            "Bạn hãy làm rõ bệnh, triệu chứng, thuốc hoặc chủ đề sức khỏe muốn hỏi."
            if classification.detected_language != "en"
            else "Please clarify which disease, symptom, medicine, or health topic you are asking about."
        )
        st.info(f"❓ **Cần làm rõ**: {message}")
        st.caption("Router không đủ tín hiệu domain. Bỏ qua retrieval.")
        return

    try:
        retriever = get_retriever(chunk_size, qdrant_url, embedding_model)
    except Exception as exc:
        st.error(
            f"⚠️ Bỏ qua pipeline — không kết nối được Qdrant hoặc không load được embedding model.\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            f"Chạy `docker compose up qdrant` rồi thử lại."
        )
        return

    with st.spinner("Đang retrieve từ Qdrant..."):
        started = time.perf_counter()
        try:
            result = retriever.retrieve(
                query,
                top_k_dense=20,
                top_k_bm25=20,
                top_k_fused=top_k_fused,
            )
        except Exception as exc:
            st.error(f"Retrieval lỗi: `{type(exc).__name__}: {exc}`")
            return
        elapsed = (time.perf_counter() - started) * 1000.0

    hybrid_hits = list(result.fused_results[:top_k_output])
    hits = hybrid_hits
    rerank_result = None
    rerank_status = None
    if use_reranker and hits:
        try:
            reranker = get_reranker()
        except Exception as exc:
            st.warning(f"Reranker không dùng được, giữ nguyên hybrid order. `{type(exc).__name__}: {exc}`")
        else:
            with st.spinner("Đang rerank với cross-encoder..."):
                reranked = reranker.rerank(
                    query,
                    result.fused_results,
                    top_k=max(top_k_output, 5),
                )
                rerank_result = reranked
                hits = reranked.reranked_results[:top_k_output]
                rerank_status = reranked.status

    if rerank_status:
        st.caption(f"Rerank status: `{rerank_status}`")

    render_experiment_trace(result, hybrid_hits, hits, rerank_result)
    render_hits(hits, elapsed_ms=elapsed)

    generator = get_generator(use_local_qwen, ollama_url, qwen_model)
    answer = generator.generate(
        query,
        [{"payload": hit.payload} for hit in hits],
        response_language=response_language,
    )
    render_answer(answer.to_dict())


# --------------------------------------------------------------------------- #
# Sidebar controls                                                            #
# --------------------------------------------------------------------------- #

st.sidebar.title("🩺 Agentic RAG Y tế")
st.sidebar.caption("Hỏi đáp y tế đa ngôn ngữ có citation grounding — corpus WHO (30 fact sheets).")

with st.sidebar:
    st.subheader("Pipeline")
    chunk_size = st.selectbox("Chunk size", (300, 800), index=0)
    use_reranker = st.toggle("Bật cross-encoder reranker", value=True)
    top_k_output = st.slider("Số chunk context (top-K)", min_value=1, max_value=10, value=5)
    top_k_fused = st.slider("Số candidate cho RRF fusion", min_value=5, max_value=50, value=20)
    st.subheader("Endpoints")
    qdrant_url = st.text_input("Qdrant URL", value="http://localhost:6333")
    embedding_model = st.text_input(
        "Embedding model",
        value="intfloat/multilingual-e5-small",
        help="HuggingFace model id. Dense encoder đa ngôn ngữ.",
    )
    use_local_qwen = st.toggle("Dùng Qwen local qua Ollama", value=True)
    ollama_url = st.text_input(
        "Ollama URL",
        value=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        disabled=not use_local_qwen,
    )
    qwen_model = st.text_input(
        "Qwen model",
        value=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct"),
        disabled=not use_local_qwen,
        help="Model phải có sẵn trong Ollama, ví dụ qwen3:4b-instruct.",
    )
    response_language = st.selectbox(
        "Ngôn ngữ trả lời",
        options=("auto", "vi", "en", "bilingual"),
        format_func={
            "auto": "Tự động theo câu hỏi",
            "vi": "Tiếng Việt",
            "en": "English",
            "bilingual": "Song ngữ EN + VN",
        }.get,
    )
    st.subheader("Kịch bản mẫu")
    scenario_label = st.selectbox(
        "Chọn câu hỏi mẫu",
        options=["(chọn một)"] + list(SCENARIOS.keys()),
    )
    if scenario_label != "(chọn một)":
        st.session_state["prefill_query"] = SCENARIOS[scenario_label]
    st.divider()
    st.caption(
        "Qwen local chỉ được phép dùng retrieved context và phải trích citation; "
        "nếu Ollama lỗi, hệ thống fallback về evidence excerpts."
    )
    st.caption(
        "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn từ bác sĩ."
    )


# --------------------------------------------------------------------------- #
# Main pane                                                                   #
# --------------------------------------------------------------------------- #

st.title("Agentic RAG Y tế")
st.markdown(
    "Hỏi câu về **bệnh, triệu chứng, thuốc, hướng dẫn điều trị**. "
    "Câu ngoài phạm vi sẽ bị router từ chối; câu thiếu bằng chứng sẽ được refuse với cảnh báo."
)

default_query = st.session_state.pop("prefill_query", "")
query = st.text_area(
    "Câu hỏi",
    value=default_query,
    height=100,
    placeholder="Ví dụ: Bệnh tiểu đường có triệu chứng gì?",
)

ask = st.button("🔍 Hỏi", type="primary", disabled=not query.strip())

if ask and query.strip():
    with st.status("Đang chạy pipeline…", expanded=True) as status:
        run_full_pipeline(
            query.strip(),
            chunk_size=chunk_size,
            qdrant_url=qdrant_url,
            embedding_model=embedding_model,
            use_reranker=use_reranker,
            top_k_fused=top_k_fused,
            top_k_output=top_k_output,
            use_local_qwen=use_local_qwen,
            ollama_url=ollama_url,
            qwen_model=qwen_model,
            response_language=response_language,
        )
        status.update(label="Xong", state="complete", expanded=True)

st.divider()
with st.expander("Pipeline này làm gì? — Các bước xử lý"):
    st.markdown(
        """
        1. **ScopeRouter** — rule-based classifier + LLM classifier tuỳ chọn.
           Chặn các query out-of-scope bằng tool `reject_out_of_scope` và yêu
           cầu người dùng làm rõ khi tín hiệu domain yếu.
        2. **Hybrid retrieval** — dense đa ngôn ngữ (E5) chạy trên original query
           + BM25 chạy trên translated English query, fuse bằng Reciprocal Rank Fusion.
        3. **Cross-encoder reranking (tuỳ chọn)** — sắp xếp lại top-20 với
           `mmarco-mMiniLMv2-L12`. Nếu lỗi thì fallback về hybrid order.
        4. **Answer generation** — có grounding và citation. Từ chối khi context
           rỗng hoặc khi LLM cite một chunk không nằm trong retrieved context.
        """
    )
