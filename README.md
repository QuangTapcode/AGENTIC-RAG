# Agentic RAG Y tế — Medical Question Answering with Grounded Citations

A bilingual (Vietnamese/English) Retrieval-Augmented Generation prototype that
answers questions about **diseases, symptoms, medicines, and treatment
guidelines** using a curated corpus of 30 WHO fact sheets. Every answer is
either grounded in retrieved evidence with a validated citation, or the system
refuses.

## Problem

Health information available online mixes reliable sources with unverified
content. A patient-facing assistant that answers without citing its evidence
risks giving false confidence to advice that is off-corpus, out-of-date, or
outright hallucinated. This project builds a small, auditable prototype that
demonstrates the safety machinery — router, hybrid retrieval, reranking,
citation validation, refusal — end-to-end, so those components can be scaled
onto a bigger corpus later.

## Dataset

- **30 WHO fact sheets** collected on 2026-09-27, English source language.
  Metadata is in [data/manifest.json](data/manifest.json).
- Parsed to structured JSON via Kreuzberg 4.10.4; parse report at
  [data/parsed/parse_report.json](data/parsed/parse_report.json). Success:
  **30/30**, no fallbacks, no quality flags.
- Chunked at both **300** and **800** tokens (`tiktoken` `cl100k_base`);
  section-first, atomic-block-safe policy. See
  [data/chunks_300/chunk_report.json](data/chunks_300/chunk_report.json) and
  [data/chunks_800/chunk_report.json](data/chunks_800/chunk_report.json).

Rights and attribution: [docs/data_rights.md](docs/data_rights.md).
Scope, MVP criteria, refusal contract: [docs/scope.md](docs/scope.md).

## Architecture

```
                      ┌──────────────────────┐
    query ───────────►│  ScopeRouter (rules  │──► reject_out_of_scope(query)
                      │  + optional LLM      │
                      │  hook)               │──► clarify (ask user)
                      └──────────┬───────────┘
                                 │ in_scope
                                 ▼
        ┌─────────────────────────────────────────────┐
        │  HybridRetriever                            │
        │  - dense: multilingual-e5-small (original)  │
        │  - sparse: BM25 (translated_query_en)       │
        │  - RRF fusion                               │
        └──────────────┬──────────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────────┐
        │  CrossEncoderReranker                       │
        │  (mmarco-mMiniLMv2-L12) — optional          │
        │  Fallback to hybrid order on failure.       │
        └──────────────┬──────────────────────────────┘
                       ▼
        ┌─────────────────────────────────────────────┐
        │  AnswerGenerator                            │
        │  - LLM hook enforced by system prompt       │
        │  - CitationValidationError on drift         │
        │  - Fallback: grounded evidence-excerpts     │
        │  - Insufficient-evidence refusal            │
        └─────────────────────────────────────────────┘
```

Module code lives under [src/](src/): `parser/`, `chunking/`, `embedding/`,
`scope_router/`, `retrieval/`, `reranking/`, `answering/`, `evaluation/`,
`demo/`. Tests are in [tests/](tests/).

## Setup

Prerequisites:

- Python **3.10+** (developed against 3.12/3.14).
- Docker (for local Qdrant) and `docker compose`.
- Ollama with a local Qwen model. The default is `qwen3:4b-instruct`.
- A cloud LLM API key is optional. If Qwen/Ollama is unavailable, the answer
  generator falls back to grounded evidence excerpts — still valid and cited,
  just terser.

```bash
# 1. Clone and enter the repo
git clone https://github.com/<your-account>/<your-repository>.git
cd <your-repository>

# 2. Create the virtualenv (see requirements.txt)
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell / cmd
# source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt

# 3. Optional: copy the environment reference file
# Windows PowerShell: Copy-Item .env.example .env
# macOS/Linux:
# cp .env.example .env

# 4. Verify the local Qwen model (Ollama)
ollama list
# If it is not installed yet:
ollama pull qwen3:4b-instruct

# 5. Bring up Qdrant
docker compose up -d qdrant

# 6. (One-off) ingest chunks into Qdrant
python src/embedding/ingest_qdrant.py --recreate

# 7. Run the CLI with local Qwen
python src/demo/cli.py --scenario in_scope --local-qwen

# 8. Run the Streamlit UI; Qwen local is enabled by default
python -m streamlit run src/demo/ui.py
```

On Windows, if `python` is not found, use the Python launcher instead:
`py -3 -m venv .venv`, activate `.venv\Scripts\Activate.ps1`, and replace
`python` with `py -3` in the commands above. The checked-in `.env.example` is
an environment reference; the prototype also has safe local defaults, and
shell environment variables are read when explicitly set.

The local model is called through Ollama at
`http://127.0.0.1:11434`. The UI allows changing the Ollama URL and model name
from the sidebar. The CLI uses `--local-qwen`; the full evaluator uses the same
flag. No medical document is sent to a cloud provider by this path.

## Answer language modes

The answer generator supports four response modes:

| Mode | Behavior |
|---|---|
| `auto` | Uses the language detected from the user's question. |
| `vi` | Returns a Vietnamese answer. |
| `en` | Returns an English answer. |
| `bilingual` | Returns two sections: `English` and `Tiếng Việt`. Both sections must use the same retrieved evidence. |

The language mode changes the presentation layer, not the evidence source:

```
Question
   ↓
ScopeRouter
   ↓
Hybrid retrieval (dense + BM25 + RRF)
   ↓
Optional cross-encoder reranking
   ↓
Retrieved <SOURCE> blocks with short citation IDs (S1, S2, ...)
   ↓
Local Qwen via Ollama
   ├── auto       → language of the question
   ├── vi         → Vietnamese
   ├── en         → English
   └── bilingual  → English + Tiếng Việt
   ↓
Citation normalization and validation
   ↓
Answer + validated citations + medical warning
```

In the Streamlit sidebar, select `Ngôn ngữ trả lời`. From the CLI:

```bash
python src/demo/cli.py --query "Bệnh tiểu đường có triệu chứng gì?" \
  --local-qwen --answer-language bilingual
```

The model may cite either the short source ID (`[CITATION:S1]`) or the full
chunk ID. The backend maps the citation to the real chunk and rejects unknown
references. In bilingual mode, the backend requires both `English` and
`Tiếng Việt` sections. If Qwen stops after English, it makes one grounded
Vietnamese-repair call; if that also fails, the incomplete answer is rejected
instead of being shown as a valid bilingual answer. If Ollama is unavailable,
the system falls back to grounded evidence excerpts.

## Disease-first retrieval

When a question contains a recognizable disease name, the pipeline resolves it
against the document manifest before searching for keywords. For example,
`Bệnh tiểu đường có triệu chứng gì?` is resolved to `who_diabetes`; the dense
and BM25 searches are then filtered by that `document_id`. This prevents a
generic word such as `symptoms` from bringing in a chunk about another disease.

```text
Question
   ↓
DiseaseCatalog: identify disease name
   ↓
Qdrant document_id filter
   ↓
Dense + BM25 → RRF → optional reranker
   ↓
Qwen receives only the scoped evidence
```

If no exact disease anchor is found, retrieval keeps the full corpus and the
UI shows that no document-level filter was applied. If multiple diseases are
explicitly named, the filter contains their union rather than silently picking
one.

## What does “RRF candidate count” mean?

Dense retrieval and BM25 each return a ranked list. RRF (Reciprocal Rank
Fusion) merges those lists and keeps `top_k_fused` candidates before the
optional cross-encoder reranker. `top_k_output` is the number of final chunks
sent to the answer generator.

For the query `Bệnh hen suyễn có triệu chứng gì?`:

```text
Dense top-20 + BM25 top-20
              ↓ RRF
top_k_fused candidates
              ↓ reranker
top_k_output chunks → Qwen
```

### `top_k_fused = 5`

Only RRF ranks 1–5 survive. If the best `Symptoms` chunk is rank 8 after
fusion, it is discarded and the reranker cannot recover it. This is faster but
has lower recall.

### `top_k_fused = 50`

The system keeps a wider candidate pool. In the current UI, dense and BM25
each retrieve 20 items, so the union can contain at most about 40 unique
chunks even if the slider is set to 50. A relevant chunk at fused rank 8 or
35 can still reach the reranker, which may promote it into the final top 5.
The trade-off is higher reranking latency and more candidate noise.

Recommended starting point for this prototype:

```text
top_k_fused = 20
top_k_output = 5
```

Changing `top_k_fused` changes the candidate pool; changing `top_k_output`
changes how much evidence Qwen actually receives. `rrf_k=60` is a separate
rank-decay constant and is not the candidate count.

## Running the pipeline

Individual stages are runnable directly and idempotent:

| Stage | Command | Output |
|---|---|---|
| Parse | `python src/parser/parse_documents.py` | `data/parsed/*.json`, `parse_report.json` |
| Chunk (300) | `python src/chunking/chunk_documents.py --chunk-size 300` | `data/chunks_300/chunks.jsonl` |
| Chunk (800) | `python src/chunking/chunk_documents.py --chunk-size 800` | `data/chunks_800/chunks.jsonl` |
| Embed + Ingest | `python src/embedding/ingest_qdrant.py --recreate` | Qdrant collections + BM25 artifacts |
| Tests | `python -m unittest discover -s tests -p "test_*.py"` | deterministic unit/regression checks |
| Retrieve (smoke) | `python src/retrieval/benchmark_retrieval.py` | `reports/retrieval_benchmark.{md,json}` |
| Rerank (smoke) | `python src/reranking/benchmark_reranking.py` | `reports/reranking_benchmark.{md,json}` |
| Evaluate router | `python src/evaluation/run_evaluation.py --mode router-only` | `reports/evaluation_report.{md,json}` |
| Evaluate full | `python src/evaluation/run_evaluation.py --mode full --local-qwen` | full pipeline metrics with local Qwen (requires Qdrant + Ollama) |
| Cost benchmark | `python src/evaluation/benchmark_pipeline.py` | `reports/cost_benchmark.{md,json}` |

## Reports (checked in)

- **Scope & MVP**: [docs/scope.md](docs/scope.md)
- **Data rights**: [docs/data_rights.md](docs/data_rights.md)
- **Multilingual retrieval design**: [docs/multilingual-retrieval.md](docs/multilingual-retrieval.md)
- **Retrieval benchmark (smoke)**: [reports/retrieval_benchmark.md](reports/retrieval_benchmark.md)
- **Reranking benchmark (smoke)**: [reports/reranking_benchmark.md](reports/reranking_benchmark.md)
- **Evaluation report**: [reports/evaluation_report.md](reports/evaluation_report.md) — router: 100 % binary accuracy, 5/5 out-of-scope blocked
- **Cost & performance benchmark**: [reports/cost_benchmark.md](reports/cost_benchmark.md)
- **Failure analysis (canonical)**: [reports/failure_analysis.md](reports/failure_analysis.md)
- **AI worklog**: [AI_WORKLOG.md](AI_WORKLOG.md)

### How to interpret the experiment

The Streamlit demo uses a grounded evidence-excerpt fallback when no LLM is
configured. The answer is therefore generated from the final retrieved context;
it is not an independent paragraph written from general model knowledge.

#### Why chunk size changes the result

- **Chunk 300** creates smaller, more focused passages. A passage is more likely
  to contain one section such as `Symptoms`, but important context may be split
  across multiple chunks.
- **Chunk 800** creates longer passages containing more surrounding context.
  This can improve context coverage, but unrelated sentences can dilute the
  embedding/BM25 signal.

Changing the chunk size changes the text being embedded and searched. Therefore
the top chunk, chunk ID, excerpt length, citation coverage, and retrieval metrics
may change even when both configurations identify the same WHO document.

#### Why reranking changes the result

Hybrid retrieval uses dense similarity plus BM25/RRF to produce candidates. The
cross-encoder reranker then reads each `(query, chunk text)` pair and reorders
the candidates according to direct query–passage relevance. For example, a
hybrid result may initially rank `Key facts` above `Symptoms`; the reranker can
move `Symptoms` to rank 1 because it is a more direct answer to a symptoms
question.

Reranking is not guaranteed to change the final answer. If hybrid retrieval has
already placed the correct chunk at rank 1, reranking may preserve it. This is
what happens for the diabetes-symptoms example: lower candidates can be
reordered while the top-1 context, citation, and fallback answer remain the
same. That is an agreement case, not evidence that the reranker was skipped.

The UI distinguishes `Output changed positions` (chunks actually supplied to
the answer generator) from `Candidate changed positions` (reordering anywhere
in the inspected candidate pool).

For the submission, compare configurations on the full evaluation set rather
than on one answer alone. Report Recall@k/MRR, citation precision/recall,
answer quality, and latency. Use document-level metrics when comparing chunk
sizes; the evaluation anchors expected chunk IDs on `chunks_300`, so raw
chunk-level IDs are not directly comparable with `chunks_800`.

For a visible demo, choose the built-in scenario
`Experiment (chunk + rerank, EN)` with the query `What are common symptoms of
depression?`, set output top-K to `1`, and compare chunk `300` vs `800`, then
toggle the reranker. This query changes both the top chunk and the excerpt
length in the current corpus. The heart-attack scenario is a second reranking
example.

## Demo scenarios

```bash
python src/demo/cli.py --scenario in_scope        # answer with citation
python src/demo/cli.py --scenario out_of_scope    # refuse via reject_out_of_scope
python src/demo/cli.py --scenario insufficient    # refuse due to weak evidence
python src/demo/cli.py --scenario all             # run all three
python src/demo/cli.py --query "..." --rerank     # try your own query
```

The demo prints router decision → retrieval trace → optional rerank → Qwen
grounded answer with citations. Qwen receives only the retrieved `<SOURCE>`
blocks and must cite their source IDs or chunk IDs. Citation validation rejects
a citation that is not present in the retrieved context. If Ollama fails, the
pipeline falls back to grounded excerpts and records the failure reason in the
UI/trace.

## Troubleshooting a fresh clone

- **`localhost:6333` is unavailable:** run `docker compose up -d qdrant`, then
  retry. If the collection does not exist, run
  `python src/embedding/ingest_qdrant.py --recreate` once.
- **Ollama is unavailable:** start Ollama with `ollama serve` if it is not
  already running, then run `ollama pull qwen3:4b-instruct`. Alternatively,
  turn off `Dùng Qwen local qua Ollama` in the UI; the system will show grounded
  evidence excerpts instead.
- **The first request is slow:** the embedding and reranker models may be
  downloaded from Hugging Face on first use. `HF_TOKEN` is optional but can
  increase download limits.
- **The UI opens but retrieval is skipped:** the Streamlit shell can run
  without services, but answering requires both Qdrant and the embedding
  model. Check the error shown under the retrieval trace.
- **Bilingual output is incomplete:** use `response_language=bilingual` (or
  `Song ngữ EN + VN` in the UI). The backend retries a missing Vietnamese
  section once and rejects an incomplete bilingual answer rather than silently
  presenting English-only output.

## Limitations

- **Corpus is small** (30 WHO fact sheets). Broader questions may be routed to
  `clarify` or refused; that is by design.
- **No page numbers** in the WHO Markdown corpus. Citations fall back to
  `source_line_start/end`. The renderer intentionally displays "page
  unavailable" rather than inventing a number.
- **Answer correctness is measured with a lexical proxy** until an LLM judge is
  wired into [src/evaluation/run_evaluation.py](src/evaluation/run_evaluation.py).
- **Local Qwen is a 4B quantized model**: it improves answer synthesis and
  readability over excerpt-only fallback, but can still make unsupported or
  malformed claims. The grounding prompt, required citations, and validation
  layer remain mandatory; local generation is not treated as automatically
  correct.
- **Vietnamese router recall depends on term list**: the manifest currently
  supplies English topic/title terms, so 5/25 answerable Vietnamese queries
  route to `clarify`. See
  [reports/failure_analysis.md](reports/failure_analysis.md) §7 for the queued
  fix.
- **Rerank on chunk 800** regressed MRR on the smoke set — do not enable it
  there by default.
- **BM25 for Vietnamese** relies on a static VI→EN medical dictionary; missing
  entries fall back to Vietnamese passthrough and are reported in the
  retrieval trace note.

## Repository layout

```
Agentic RAG/
├── data/                     # raw, parsed, chunked, vector artifacts, manifest
├── docs/                     # scope, data rights, multilingual notes
├── eval/                     # evaluation dataset (30 questions)
├── reports/                  # generated benchmarks and analyses
├── src/
│   ├── parser/               # Kreuzberg-based parsing
│   ├── chunking/             # section-first, tiktoken-verified
│   ├── embedding/            # e5-small + BM25 ingest to Qdrant
│   ├── scope_router/         # rule + optional LLM classifier
│   ├── retrieval/            # dense + sparse + RRF hybrid
│   ├── reranking/            # multilingual cross-encoder
│   ├── answering/            # grounded, cited answer generator
│   ├── evaluation/           # metrics + harness + cost benchmark
│   └── demo/                 # CLI showcase
├── tests/                    # pytest smoke tests per stage
├── docker-compose.yml        # Qdrant service (pinned digest)
├── requirements.txt
└── TODO.md                   # per-phase checklist
```
