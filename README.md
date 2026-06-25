# notion-second-brain

A local-first "second brain" agent. Ingests Notion exports (and if needed live Notion pages), optionally crawls linked pages (not implemented yet), processes PDFs/images into markdown, indexes everything into Qdrant, and serves RAG queries via CLI (without memory) or in Streamlit GUI (with memory within abd between runs). All inference runs locally through Ollama — no cloud APIs. Designed to run on native Windows.

## Why this exists

Demonstrates a complete local RAG stack with: hybrid dense+sparse retrieval, cross-encoder reranking, sentence-aware chunking (with atomic code/table handling), file-based persistent memory, an anchored-rubric evaluation harness, and (optionally) Phoenix observability — all wired together with `pydantic-ai`. Built to run optimally on a single 12 GB VRAM / 32 GB RAM system. Does not include potentially significant llama.cpp optimizations.

## What it does

1. **Ingests** Notion exports + crawled pages (not implemented yet) + PDFs + scanned images
2. **Processes** via OCR (marker-pdf), quality filtering, sentence-aware chunking, hybrid embedding (dense + sparse BM25)
3. **Indexes** into Qdrant with optional cross-encoder reranking
4. **Serves** via CLI REPL and Streamlit GUI
5. **Remembers** conversation context across turns via file-based memory
6. **Evaluates** quality via hand-rolled runner + `pydantic_evals` runner (with example cases and rubrics)

## Requirements

- **Python**: 3.12+
- **Ollama**: running locally — pulls models on demand
- **Docker**: for Qdrant + (optionally) Phoenix containers
- **Hardware baseline**: 12 GB VRAM, 32 GB RAM (tested on Windows; cross-platform)
- **Disk**: ~30 GB for Ollama models + Qdrant persistence

For lower-end hardware, see [Hardware sizing](#hardware-sizing).

## Install

Single venv. One conflict: openai version swap to move between RAG/agent mode and marker mode (see [OpenAI conflict](#openai-conflict-marker-pdf-vs-pydantic-ai)).

### With pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### With uv (alternative)

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

The fully project-managed `uv` flow (with `pyproject.toml` + `uv.lock`) is the theoretically more optimal path, but encountered setup issues on this stack — `uv pip` or simply `pip` is the recommended path until those are resolved.

## Notion setup

You need **two separate Notion integrations** because they serve different code paths:

- `NOTION_TO_MD_AUTH_TOKEN` — used by `scripts/run_etl.py` to bulk-fetch pages as markdown during ETL.
- `NOTION_ASSISTANT_AUTH_TOKEN` — used by the agent's `fetch_notion_page` tool to live-fetch a single page on demand (slow, re-runs markdown extraction on the spot).

If you don't want or need the live-fetch tool, skip the second integration and only set `NOTION_TO_MD_AUTH_TOKEN`.

### Creating the integrations

1. Open https://www.notion.so/profile/integrations
2. Click **+ New integration** → name it e.g. `Second Brain — Notion-to-MD` → submit.
3. Capabilities: **Read content** (read-only permissions are enough).
4. Copy the **Internal Integration Secret** (starts with `ntn_` in newer Notion formats).
5. Paste it into your `.env`: NOTION_TO_MD_AUTH_TOKEN=ntn_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
6. **Optional**: create a second integration named `Second Brain — Live Assistant` (same read-only capabilities) and paste its token into `NOTION_ASSISTANT_AUTH_TOKEN` only (used by the live Notion tool, established if re-indexing entire DB is too painful).

### Granting page access

For each integration:
- Open each page you want ingested in Notion.
- Click the **⋯** menu → **Connections** → add the integration.
- For workspace-wide access: open Settings → **Connections** → add the integration at the workspace level.

Without this step the integration returns empty results for every page.

## Configure `.env`

Copy the block below into `.env` and fill in the two Notion tokens from above. Comments explain each knob.

```bash
# ── Notion (paste both tokens from setup above) ─────────────────────────────
NOTION_ASSISTANT_AUTH_TOKEN=ntn_...
NOTION_TO_MD_AUTH_TOKEN=ntn_...

# ── ETL mode ────────────────────────────────────────────────────────────────
LOAD_MODE="notion"            # "files" or "notion"

ETL_PAGE_NAME="AI/ML/Data Science"
# For testing per-page extraction and avoiding Notion rate limits, set to a
# specific page name. Otherwise, leave blank to load all pages at once.
# ETL_PAGE_NAME="AI/ML/Data Science"

# ── Marker (PDF / image OCR) ─────────────────────────────────────────────────
MARKER_USE_LLM=false          # leave false for fast local OCR; set true to
                              # improve accuracy via LLM cleanup (requires
                              # marker.services.ollama.OllamaService config)
MARKER_FORCE_OCR=true         # force OCR on all pages even if digital text
                              # is present; also formats inline math
MARKER_WORKERS=2              # parallel workers; raise to 4+ if CPU-rich
MARKER_DISABLE_IMAGES=false   # set true to skip image extraction (faster)

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:32768    # matches docker-compose port mapping
FORCE_REINDEX=true                   # set true on first run, or when
                                     # chunker/embedder changes — drops the
                                     # old collection so v3+v4 chunks don't
                                     # coexist
ENABLE_RERANK=true                   # cross-encoder reranking on top of
                                     # hybrid retrieval

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:latest
# USE GEMMA, NOT QWEN, FOR THE AGENT — Qwen has a large KV-cache offload that
# spills to CPU and tanks throughput (see test_agent_time.py).
# You CAN use qwen for chunking — fine there.
# MUST ensure Ollama only uses GPU for inference — check with `ollama ps` command.

# ── Memory layer ────────────────────────────────────────────────────────────
ENABLE_MEMORY=true
RECENT_LOG_DAYS=2

# ── Evaluations ─────────────────────────────────────────────────────────────
JUDGE_MODEL=qwen3:8b
```

### Verify GPU offload

After Ollama pulls a model, run:

```bash
ollama ps
```

Expected output: `100% GPU`. If you see `35%/65% CPU/GPU` or any CPU percentage, the model is offloading. Common fixes:
- Set `OLLAMA_NUM_GPU=99` in your shell env (ignored when set in `.env` on some Ollama versions — use shell-level export).
- Create a `Modelfile` with `num_gpu 99` and re-create the model: `ollama create gemma4-100gpu -f Modelfile`.
- Use a smaller model with little to no GPU offload.

## Data layout

```
data/
├── raw/                           # Notion markdown + downloaded images
│   ├── documents/<page-name>/*    # Raw Notion exports
│   ├── images/<page-name>/*       # downloaded page images             
│   └── raw_md                     # Raw Notion text exports
├── crawled/                       # (not currently used) - web crawler output (JSON)
├── clean/                         # processed markdown
│   ├── pdfs_md/                   # PDF → md via marker
│   ├── images_md/                 # image OCR via marker
│   └── clean_md/                  # raw_md → cleaned md
├── pages.txt                      # page list for live Notion fetch
└── qdrant_storage/                # Qdrant persistence volume
memory/                            # persistent conversation memory 
├── YYYY-MM-DD.md                  # Session-level memory
└── MEMORY.md                      # User-level memory

```

Format of `data/pages.txt`:
```
Page Name | https://www.notion.so/...
Another Page | https://www.notion.so/...
```

## Pipeline stages

### 1. ETL (Notion / files → raw markdown)

```bash
python scripts/run_etl.py
```

Reads `data/raw/*.md` if `LOAD_MODE="files"`, or fetches live from Notion if `LOAD_MODE="notion"`. Writes raw markdown under `data/raw/raw_md/`.

**First-run recommendation:** set `ETL_PAGE_NAME="Some Page"` to test the Notion connection on a single page before pulling your whole workspace. Notion rate limits are aggressive on large workspaces and one bad token can waste 20 minutes.

### 2. Cleaning (raw → clean markdown)

```bash
python scripts/run_clean_md.py
```

LLM-based cleanup of `data/raw/raw_md/` → `data/clean/clean_md/`. Runs after ETL and (optionally, no need to re-process) Marker.

### 3. Marker (PDF + image → markdown)

```bash
python scripts/run_marker.py
```

Converts PDFs (`data/raw/pdfs/`) and images (`data/raw/images/`) into markdown under `data/clean/pdfs_md/` and `data/clean/images_md/`. Independent of ETL — only run when you have new source PDFs/images.

Useful env vars (in addition to the `.env` block):
- `MARKER_STEP` — `pdfs`, `images`, or `all`
- `MARKER_TEST_SUBDIR` — process a subset for debugging

For higher accuracy at the cost of speed, set `MARKER_USE_LLM=true` and configure `marker.services.ollama.OllamaService` per marker-pdf docs. (Default `false` is what we ship — skips marker's LLM dependency.)

**Alternatives with more involved setup paths for native Windows (some not yet supported), but which might avoid the `open-ai` conflict:** PyMuPDF4LLM (lightest, no OCR), MinerU (heavier, complex install), Kreuzberg, Docling.

### 4. RAG indexing (clean markdown → Qdrant)

```bash
python scripts/run_rag.py
```

Reads from `data/clean/` (all subfolders), applies:
- **Sentence-aware chunking** (`pipelines/rag/chunker.py`, `chunker_version=4`) — atomic handling of fenced code blocks and tables
- **Hybrid embeddings** (`pipelines/rag/embeddings.py`) — dense (Ollama `nomic-embed-text`) + sparse BM25 (`fastembed`)
- **Cross-encoder reranking** (`pipelines/rag/reranker.py`) — `BAAI/bge-reranker-v2-m3`, gated by `ENABLE_RERANK`
- **Qdrant upload** with native dense+sparse vectors and RRF fusion

Indexes into collection `second_brain` at `localhost:32768` (mapped from Qdrant's internal 6333). Visualized through http://localhost:32768/dashboard#/collections.

Useful env vars:
- `FORCE_REINDEX=true` — drop and recreate the collection (use when more data is needs to be added or chunker/embedder changes)
- `RETRIEVAL_TOP_N=10`, `RERANK_TOP_K=3` — optimization knobs

> **Qdrant indexing note:** Qdrant may report `indexed_vectors_count: 0` while `points_count > 0` because the optimizer defers HNSW indexing until unindexed data exceeds `indexing_threshold` (kB). For immediate indexing, call `/collections/<name>/indexes/optimizer` `optimize` or lower `indexing_threshold` in `pipelines/rag/indexer.py`.

### 4. Agent (query your second brain)

#### CLI REPL (NO MEMORY)
```bash
python -m assistant.cli
```

Interactive:
```
> What are my current goals?
> Summarize my goals for the next quarter
> exit
```

Single query:
```bash
python -m assistant.cli "What should I focus on?"
```

#### Streamlit GUI (WITH MEMORY)
```bash
streamlit run assistant.app
```

Opens a small UI for Q&A + conversation history. Always run from the repo root.

## Memory (persistent context)

File-based memory at `memory/`:
- `memory/MEMORY.md`     — distilled long-term context (curated facts, preferences)
- `memory/YYYY-MM-DD.md` — daily conversation logs

Each turn is appended as:
```
### HH:MM
[IN-CONVERSATION CONTEXT — not knowledge base]
User: ...
Assistant: ...
```

The agent loads `MEMORY.md` + the most recent `RECENT_LOG_DAYS` daily logs (default 2) into its system prompt per call. Implemented via per-call `Agent` instantiation in `assistant/agent.py` (pydantic-ai 1.x freezes `agent.system_prompt` after construction, so the dynamic memory block is injected by rebuilding the agent each turn).

Toggle: `ENABLE_MEMORY=true|false`.

## Evaluations

Two eval runners — both use the same anchored rubric and golden test set, so you can cross-validate.

| Script | Framework | Purpose |
|---|---|---|
| `python -m evals.run_evals` | Hand-rolled | Canonical. 4-criterion anchored rubric (relevance, correctness, citation_quality, safety), LLM-as-judge via Ollama. |
| `python -m evals.run_pydantic_evals` | pydantic_evals | Same dataset + rubric, using `Evaluator` + `EvaluationReason` natively. Default runs all 3 tiers. |

Each writes a JSON artifact to `evals/results/*.json` for trend analysis.

Set the judge model via `.env` e.g.: `JUDGE_MODEL=qwen3:8b`.

Required local Ollama models e.g.: `gemma4:latest` (agent), `qwen3:8b` (judge), `nomic-embed-text` (embeddings).

### Eval design methodology

The eval design follows the frameworks documented in https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/10-evaluation/docs/en.md:
- `extras/llm-eval-patterns.md` — methodology for moving from "vibes-based" to statistically significant evaluation (anchored scales, pointwise vs pairwise, test suite sizing, CI gates).
- `extras/prompt-eval-designer.md` — the protocol used to design the rubric + test cases in `evals/cases.py` and `evals/rubrics.py` (analyze → criteria → anchored rubric → 3-tier test suite → judge prompt → decision framework).

Both kept in `extras/` for reference.

## Observability (Phoenix)

`extras/run_phoenix.py` wires Phoenix OTel tracing. Traces land in the Phoenix UI at `http://localhost:6006`.

The Phoenix container is already defined in `docker-compose.yml`. Start it with:

```bash
docker compose up -d phoenix
```

Then visit `http://localhost:6006/projects`, select the `default` project (or rename appropriately), and run the agent via CLI/Streamlit — every `agent.run(...)`, LLM call, and tool call appears as a span.

## OpenAI conflict (marker-pdf vs pydantic-ai)

The single biggest setup pain point. From `pip check`:

> ```
> marker-pdf 1.10.2 requires openai<2.0.0,>=1.65.2, but you have openai 2.41.0 which is incompatible.
> ```

Or in reverse when running marker first:

> ```
> pydantic-ai 1.x requires openai>=2.0.0, but you have openai 1.99.9 which is incompatible.
> ```

- `marker-pdf 1.10.2` pins `openai<2.0.0` (v1.x)
- `pydantic-ai 1.105.0` ships with `openai==2.41.0` (v2.x)
- These are **incompatible**.

**Recommended: The install path uses one venv.** Switch between modes by swapping `openai` version:

```bash
# Going into marker mode
pip install "openai<2.0.0,>=1.65.2"

# Going into agent mode
pip install "openai>=2.0.0"
```

Cost: ~10 seconds for the swap + a few hundred KB download. Needed if extra documents or images have been added to the knowledge base that need to be processed to markdown.

### Decision tree

New PDFs / images need OCR?       → marker mode (pip install openai<2, run_marker.py)
Just want to re-index cleaned md? → agent mode (pip install openai>=2, run_rag.py — does NOT need marker)
Want to chat / develop agent?     → agent mode
Want to run evals?                → agent mode (eval judge is local Ollama, doesn't care about openai)

### Troubleshooting
- **Notion returns nothing**: integration created but pages aren't shared. Open each page → ⋯ → Connections → add the integration. See [Notion setup](#notion-setup-required-for-live-ingestion).
- **Notion rate limits**: aggressive on large workspaces. Use `ETL_PAGE_NAME="Single Page"` for first-run validation. Notion returns 429s after ~3 requests/sec sustained; ETL retries but a 10k-page workspace can take hours.
- **GPU offload**: `ollama ps` after first agent run. If CPU %, set `OLLAMA_NUM_GPU=99` in shell env or recreate the model via Modelfile with `num_gpu 99`.
- **Streamlit from wrong CWD**: app inserts `sys.path` defensively but `cd` to repo root first.
- **Eval JSON missing**: all runners write to `evals/results/*.json` by default; pass `--output PATH` to override.
- **Memory not used**: check `ENABLE_MEMORY=true` and that `memory/MEMORY.md` exists (auto-created on first run).
- **Schema change**: set `FORCE_REINDEX=true` to drop the Qdrant collection and rebuild (otherwise v3 + v4 chunks coexist and retrieval gets noisy).

The base `requirements.txt` pulls `torch+cu130` — about 2.5 GB. Expect first install to download ~3-4 GB of wheels.

## Project layout

```
notion-second-brain/
├── assistant/                  # Agent runtime (CLI, Streamlit, programmatic)
│   ├── agent.py                # pydantic-ai Agent + per-call instantiation
│   ├── app.py                  # Streamlit GUI
│   ├── cli.py                  # CLI REPL
│   ├── memory.py               # file-based memory (MEMORY.md + daily logs)
│   └── tools.py                # hybrid_retrieve, fetch_notion_page
├── pipelines/
│   ├── etl/                    # raw markdown → clean markdown
│   │   ├── crawler.py
│   │   ├── image_extractor.py
│   │   ├── notion_loader.py    # NOTION_TO_MD_AUTH_TOKEN
│   │   └── pdf_extractor.py
│   ├── rag/                   
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   ├── reranker.py    
│   │   └── indexer.py
│   ├── utils/                   
│   │   ├── image_utils.py
│   │   └── pdf_utils.py
│   └── models.py
├── scripts/                    # CLI entry points for each pipeline stage
│   ├── run_marker.py           # PDF/image OCR (uses marker-pdf)
│   ├── run_clean_md.py         # raw → clean markdown
│   ├── run_etl.py              # Notion/files → raw markdown
│   └── run_rag.py              # clean markdown → Qdrant
├── evals/                      # evaluation suite
│   ├── cases.py                # golden + adversarial + distribution cases
│   ├── rubrics.py              # 4 anchored 1-5 rubrics
│   ├── judges.py               # LLM-as-judge (Ollama)
│   ├── run_evals.py            # hand-rolled runner (canonical)
│   └── run_pydantic_evals.py   # pydantic_evals runner
├── extras/                     # sandbox / non-operational scripts + reference
│   ├── run_deepeval.py         # DeepEval attempt — abandoned, do not run
│   ├── run_phoenix.py          # Phoenix OTel instrumentation / optional
│   ├── llm-eval-patterns.md    # eval methodology reference
│   ├── prompt-eval-designer.md # rubric design protocol reference
│   └── deepeval_info.md        # DeepEval docs dump (for reference)
├── memory/                     # persistent conversation memory 
├── data/                       # all data files 
├── images/                     # README screenshots
├── docker-compose.yml          # Qdrant + (optional) Phoenix 
├── requirements.txt            # all deps (single venv)
└── main.py                     # not used; entry points are scripts/* and assistant/*
```