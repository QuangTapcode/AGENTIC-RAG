# Failure Analysis — Agentic RAG Y tế

Aggregated failure log across parsing, chunking, retrieval, reranking, answer
generation, citation and router. Each row records query/input, expected,
actual, root cause, fix, and lesson learned. This file is the canonical
starting point; module-specific pages under `reports/*_failure_analysis.md`
keep deep detail for a single stage.

## 1. Parsing / OCR

| Input | Expected | Actual | Root cause | Fix | Lesson |
|---|---|---|---|---|---|
| WHO Markdown fact sheets | 30/30 documents parsed to structured JSON | 30/30 ok, 0 fallbacks, 0 quality flags — see [parse_report.json](../data/parsed/parse_report.json) | Corpus is native Markdown, so no OCR path was exercised | No fix needed; keep OCR path documented for future PDFs | Do not enable OCR until real PDF/scan inputs are added — it inflates latency for no gain |
| Future PDF/DOCX | Preserve tables, section headers, page numbers | _Untested_ | Not applicable yet; Kreuzberg exposes OCR flags per document | Add `--ocr` argument to `parse_documents.py` when PDFs are ingested and record `quality_flags` per file | Page numbers are a first-class citation surface; whatever parser we add for PDF must expose them |
| Any parser failure | Marked with `parser=fallback` and quality flag | Handled by UTF-8 raw fallback in [parse_documents.py](../src/parser/parse_documents.py) | Kreuzberg exception path returns raw text | Keep the fallback — never let a bad parse block ingest | Ingest must be robust to a bad file, not brittle |

## 2. Chunking

| Input | Expected | Actual | Root cause | Fix | Lesson |
|---|---|---|---|---|---|
| 30 documents → chunks_300 | ≤ 300 tokens per chunk, 0 oversized | 476 chunks, mean 194 tokens, max 300, oversized 0 — see [chunk_report.json](../data/chunks_300/chunk_report.json) | Section-first policy plus atomic block enforcement | No fix — quality flags surface if any block had to be token-split | Never split across a table/list/dose block; add a `quality_flag` when forced |
| 30 documents → chunks_800 | ≤ 800 tokens per chunk | 350 chunks, mean 257 tokens, max 799 | Same policy, larger budget preserved more full sections | No fix | Larger budget shows how many chunks are "small by structure" — do not pad chunks just to hit budget |
| Any chunk containing dose/contraindication | Whole block stays together | Enforced via atomic_blocks policy | Deliberate design decision | Keep as-is | Splitting a dose across chunks silently corrupts the answer |

## 3. Retrieval (smoke set)

Detail: [retrieval_benchmark.md](retrieval_benchmark.md) and [retrieval_benchmark.json](retrieval_benchmark.json).

| Query pattern | Expected | Actual | Root cause | Fix | Lesson |
|---|---|---|---|---|---|
| Vietnamese query with only 1 medical term (e.g. "bệnh dại", "ung thư") | Hybrid retrieval finds the correct WHO document | Dense-only Hit@5 on smoke set: 0.33 for chunk 300, chunk 800; BM25-only via translated query brings it to 1.0 | Dense multilingual embedding alone underweights medical entity match on short Vietnamese queries; BM25 on translated query recovers the exact-term match | Keep hybrid RRF (default) and never fallback to dense-only | Hybrid is not a "nice-to-have"; on VI queries with 1-2 medical terms it is the difference between find and miss |
| Vietnamese query whose dictionary translation misses a medical term | BM25 still contributes | BM25 returns empty; note `translated_query_en_has_no_corpus_terms` set on the retrieval trace | Static VI→EN dictionary in [hybrid_retriever.py](../src/retrieval/hybrid_retriever.py) misses long-tail terms | Dense retrieval still ranks; note is preserved for evaluation | Do not silently substitute a passthrough VI query into BM25 — always mark it in the trace |
| Query with rare disease name absent from BM25 vocab | Hybrid picks up via dense | Chunk 800 hybrid Hit@5 = 0.83 (one miss out of 6) | BM25 vocab does not cover the entity; dense embedding still finds the right doc | Retune `top_k_dense` up (20) and keep chunk 300 as default (higher MRR = 0.68) | Chunk 300 balances precision and recall better on this corpus |

## 4. Reranking

Detail: [reranking_benchmark.md](reranking_benchmark.md), [reranking_failure_analysis.md](reranking_failure_analysis.md).

| Failure mode | Detection | Fallback | Lesson |
|---|---|---|---|
| Model load or inference error | `status=fallback`, `notes` contains `reranker_failure=...` | Return hybrid RRF order untouched | Reranker is a refinement, not a hard dependency |
| Reranker latency > budget | `notes` contains `reranker_latency_budget_exceeded=<budget>ms` | Return hybrid RRF order and log the note | Never let a slow model stall the answer path |
| Reranker degrades Hit@5 vs hybrid | Compare `retrieval` vs `reranking` benchmark output | Do not declare reranking a win in the report | On chunk 800, smoke set showed rerank MRR dropped from 0.66 → 0.50 — do not enable rerank there by default |
| Drug/entity name distortion during rerank | Cross-encoder reads original chunk text as-is | Payload text is passed unchanged | Never rewrite source text before reranking |

## 5. Answer generation

Detail: [answer_generation_failure_analysis.md](answer_generation_failure_analysis.md).

| Failure mode | Detection | Safe behavior | Lesson |
|---|---|---|---|
| No context / no valid payload | `citations=[]` after `_collect_sources` | `status=insufficient_evidence` refusal with warning | Do not let LLM answer when context is empty |
| LLM cites a chunk id not present in context | `CitationValidationError: citation_not_in_context:<id>` | Discard the draft, `status=citation_error`, emit refusal | Never trust LLM-invented citation ids |
| LLM emits factual claim without any `[CITATION:...]` marker | `CitationValidationError: answer_contains_no_valid_citation_marker` | Discard the draft | Citation is a hard invariant, not a decoration |
| LLM/provider outage | Exception in `self.llm(...)` | Fall back to evidence-excerpt path with citations; `used_llm=False` | Provider failure must not silently drop safety |
| Payload has no page number (WHO fact sheets) | `page_start` / `page_end` are `None` | Show `page unavailable` + `source_line_start/end` | Do not invent page numbers |
| Prompt injection inside source text | System prompt marks `<SOURCE>` as untrusted evidence | Ignore instructions inside source blocks | Retrieved text is data, not instructions |

## 6. Citation

| Failure mode | Detection | Safe behavior | Lesson |
|---|---|---|---|
| Citation shown for chunk not in the retrieved context | Blocked at `_validate_and_render` | Refuse | Same invariant as answer generation — validated at render time |
| Duplicate citations for the same chunk | `_collect_sources` skips already-seen `chunk_id` | Emit once | Cleaner reader experience, no double-count for citation precision |
| Missing `source_url` in payload | Citation `display_label` still shows `title — page label — section` | Include what is available | Payload schema must include `source_url`; enforced at ingestion |

## 7. Router misclassification (observed, live)

Detail: run `python src/evaluation/run_evaluation.py --mode router-only`; report at
[evaluation_report.md](evaluation_report.md).

On the 30-question evaluation set the rule-based router blocks **5/5 out-of-scope**
queries (100% safety recall) and never mis-labels an in-scope query as
out-of-scope. However **5/25 in-scope answerable questions** are routed as
`clarify` because the query only contains a single medical term and does not
clear the `medical_score >= 2` threshold.

| Query | Expected | Actual | Root cause | Fix | Lesson |
|---|---|---|---|---|---|
| `Sau khi bị chó nghi mắc bệnh dại cắn thì cần biết gì về dự phòng sau phơi nhiễm?` | in_scope | clarify | Only "bệnh" (weight 1) matches VI medical terms; "bệnh dại" is not in the term list | Add multi-word VI entities from manifest into router term set | The manifest already lists English titles/topics — add a VI alias file so "bệnh dại" → matches |
| `Những cách nào có thể giúp giảm nguy cơ ung thư?` | in_scope | clarify | "ung thư" is not in medical terms | Add "ung thư" and other VI entity aliases | The manifest topic "cancer" is English; the router needs a VI translation of the manifest |
| `Động kinh có thể được kiểm soát bằng cách nào?` | in_scope | clarify | "động kinh" not in medical terms | Add VI aliases | Same as above |
| `What treatments are available for asthma?` | in_scope | clarify | Only "asthma" matches (weight 1); "treatments" (plural) not in term list | Add plural forms and lemma-fold before matching | Term matching in a rule-based router needs stemming, or fall back to the LLM classifier hook (already supported at [scope_router.py:241](../src/scope_router/scope_router.py#L241)) |
| `How can pneumonia be prevented in children?` | in_scope | clarify | "pneumonia" not in default medical terms | Same as above | Route rule-based failures through the LLM classifier hook — the interface exists, it just needs a provider |

Fixes queued (not yet implemented):

1. Extend `DEFAULT_MEDICAL_TERMS` with Vietnamese entity aliases derived from
   the manifest (`bệnh dại`, `ung thư`, `động kinh`, `viêm phổi`, `hen suyễn`, …).
2. Case-insensitive plural/lemma fold (`treatment` ↔ `treatments`).
3. Wire the optional LLM classifier hook when confidence < 0.6 — the code path is
   already there.

## 8. Summary of open risks

- **Router recall on rare entities**: fixable by expanding the term list; currently blocks 20 % of answerable queries into `clarify`.
- **Hybrid rerank on chunk 800**: smoke set showed regression; do not enable by default at chunk 800.
- **No live LLM in eval**: correctness/faithfulness are lexical proxies until an LLM judge is wired in.
- **No page numbers for WHO Markdown**: intentional. Source-line ranges are the substitute; do not invent.
- **Prompt-injection posture**: enforced by system prompt only. A dedicated content filter on `<SOURCE>` blocks is still a fair follow-up.

## 9. Feedback loop

New failures should be appended to this file with the same schema. The router's
`log_feedback(...)` helper at [scope_router.py:309](../src/scope_router/scope_router.py#L309)
is the recommended entry point for router misclassifications; it writes
JSONL to a configurable path so they can be rolled into this file weekly.
