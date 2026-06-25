# 🧠 notion-second-brain

> A fully local RAG agent over your Notion workspace — no cloud APIs, no subscriptions, no data leaving your machine.

A local-first "second brain" agent. Ingests Notion exports (and if needed live Notion pages), processes PDFs/images into markdown, indexes everything into Qdrant, and serves RAG queries via CLI (without memory) or Streamlit GUI (with memory within and between runs). All inference runs locally through Ollama. Designed to run well on native Windows.

Built as a complete local RAG stack: hybrid dense+sparse retrieval, cross-encoder reranking, sentence-aware chunking (with atomic code/table handling), file-based persistent memory, an anchored-rubric evaluation harness, and (optionally) Phoenix observability — all wired together with `pydantic-ai`. Optimised for a single 12 GB VRAM / 32 GB RAM system.

## The problem

I use [Notion](https://www.notion.com/) heavily for knowledge management, collecting and organizing my thoughts, notes, and research. As my notes have grown, it has become increasingly beneficial to summarize key sections or pages. While Notion includes a native AI package, I wanted to avoid it due to security, privacy and cost concerns. So I built this instead.

---

## ✨ Features

- 🔍 **Hybrid retrieval** — dense (nomic-embed-text) + sparse (BM25) with RRF fusion
- 🔁 **Cross-encoder reranking** — BAAI/bge-reranker-v2-m3 for precision on top of recall
- ✂️ **Sentence-aware chunking** — atomic handling of code blocks and tables (v4)
- 🧠 **Persistent memory** — file-based per-session and long-term memory injected at runtime
- 📊 **Evaluation harness** — hand-rolled + pydantic_evals with 4-criterion anchored rubric
- 🔭 **Phoenix observability** — optional OTel tracing via Arize Phoenix
- 🏠 **Fully local** — Ollama inference, Qdrant vector store, zero cloud dependencies

---

## 🚀 Quickstart (assuming existing md data)

```bash
docker compose up -d
python -m venv .venv && .venv\Scripts\activate   # Windows
# python -m venv .venv && source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
ollama pull nomic-embed-text && ollama pull gemma4:latest
python scripts/run_rag.py       # index existing data/clean/
python -m assistant.cli         # start chatting
```

---

## 📋 Requirements

- **Python** 3.12+
- **Ollama** running locally
- **Docker** for Qdrant + (optionally) Phoenix
- **Hardware baseline**: 12 GB VRAM, 32 GB RAM (tested on Windows; cross-platform)
- **Disk**: ~30 GB for [Ollama](https://ollama.com/) models + [Qdrant](https://qdrant.tech/) persistence

For lower-end hardware, adjust model size accordingly.

---

## 📁 Project layout

```
notion-second-brain/
├── assistant/                  # Agent runtime
│   ├── agent.py                # pydantic-ai Agent + per-call instantiation
│   ├── app.py                  # Streamlit GUI (with memory)
│   ├── cli.py                  # CLI REPL (no memory)
│   ├── memory.py               # file-based memory
│   └── tools.py                # retrieve_knowledge, fetch_notion_page
├── pipelines/
│   ├── etl/                    # Notion/files → raw markdown
│   ├── rag/                    # chunker, embeddings, reranker, indexer
│   ├── utils/
│   └── models.py
├── scripts/                    # Pipeline entry points
│   ├── run_marker.py
│   ├── run_clean_md.py
│   ├── run_etl.py
│   └── run_rag.py
├── evals/                      # Evaluation suite
│   ├── cases.py                # golden + adversarial + distribution cases
│   ├── rubrics.py              # 4 anchored 1–5 rubrics
│   ├── judges.py               # LLM-as-judge (Ollama)
│   ├── run_evals.py            # hand-rolled runner (canonical)
│   └── run_pydantic_evals.py   # pydantic_evals runner
├── extras/                     # Optional / reference
│   ├── run_deepeval.py         # DeepEval showcase (see deepeval_info.md)
│   ├── run_phoenix.py          # Phoenix OTel tracing
│   ├── deepeval_info.md        # DeepEval setup notes
│   ├── llm-eval-patterns.md    # Eval methodology reference
│   └── prompt-eval-designer.md # Rubric design protocol
├── memory/                     # Conversation memory (gitignored)
├── data/                       # All data files (gitignored)
├── images/                     # README screenshots
├── docker-compose.yml
└── requirements.txt
```

---

## ⚙️ Install

Single venv. One conflict: openai version swap to move between RAG/agent mode and marker mode (see [OpenAI conflict](#openai-conflict-marker-pdf-vs-pydantic-ai)).

### With pip

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

`requirements.txt` pulls `torch+cu130` — about 2.5 GB. Expect first install to download ~3–4 GB of wheels.

### With uv (alternative)

```bash
uv venv .venv && .venv\Scripts\activate
uv pip install -r requirements.txt
```

> The fully project-managed `uv` flow (with `pyproject.toml` + `uv.lock`) is theoretically cleaner, but encountered setup issues on this stack — `uv pip` or plain `pip` is the recommended path until those are resolved.

---

## 🗒️ Notion setup

Relies on the [notion-to-md-py](https://github.com/SwordAndTea/notion-to-md-py) library.
You need **two separate Notion integrations** because they serve different code paths:

- `NOTION_TO_MD_AUTH_TOKEN` — used by `scripts/run_etl.py` to bulk-fetch pages as markdown during ETL.
- `NOTION_ASSISTANT_AUTH_TOKEN` — used by the agent's `fetch_notion_page` tool to live-fetch a single page on demand.

If you don't need live-fetch, skip the second integration.

### Creating the integrations

1. Go to https://www.notion.so/profile/integrations → **+ New integration**
2. Name it (e.g. `Second Brain — Notion-to-MD`) → **Read content** capabilities only
3. Copy the **Internal Integration Secret** (`ntn_...`) into `.env`
4. Optionally repeat for a second integration (`NOTION_ASSISTANT_AUTH_TOKEN`)

### Granting page access

For each integration, open each page in Notion → **⋯** → **Connections** → add the integration. For workspace-wide access: **Settings** → **Connections** → add at workspace level.

> Without this step the integration returns empty results for every page.

---

## 🔧 Configure `.env`

```bash
# ── Notion ───────────────────────────────────────────────────────────────────
NOTION_ASSISTANT_AUTH_TOKEN=ntn_...
NOTION_TO_MD_AUTH_TOKEN=ntn_...

# ── ETL mode ─────────────────────────────────────────────────────────────────
LOAD_MODE="notion"            # "files" or "notion"
# ETL_PAGE_NAME="AI/ML/Data Science"  # limit to one page for first-run testing

# ── Marker (PDF / image OCR) ─────────────────────────────────────────────────
MARKER_USE_LLM=false
MARKER_FORCE_OCR=true
MARKER_WORKERS=2
MARKER_DISABLE_IMAGES=false

# ── Qdrant ───────────────────────────────────────────────────────────────────
QDRANT_URL=http://localhost:32768
FORCE_REINDEX=true            # set true on first run, or after schema changes
ENABLE_RERANK=true

# ── Ollama ───────────────────────────────────────────────────────────────────
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:latest
# Use gemma4 for the agent — qwen3 has a large KV-cache offload that
# spills to CPU and tanks throughput. Qwen3 is fine as eval judge.

# ── Memory ───────────────────────────────────────────────────────────────────
ENABLE_MEMORY=true
RECENT_LOG_DAYS=2

# ── Evaluations ──────────────────────────────────────────────────────────────
JUDGE_MODEL=qwen3:8b
```

### Verify GPU offload

```bash
ollama ps
```

Expected: `100% GPU`. If you see any CPU %, set `OLLAMA_NUM_GPU=99` at the shell level (not just `.env`), or recreate the model via a `Modelfile` with `num_gpu 99`.

---

## 📁 Data layout

```
data/
├── raw/
│   ├── documents/<page-name>/   # raw Notion exports
│   ├── images/<page-name>/      # downloaded page images
│   └── raw_md/                  # raw Notion text exports
├── crawled/                     # (not yet used) web crawler output
├── clean/
│   ├── pdfs_md/                 # PDF → md via marker
│   ├── images_md/               # image OCR via marker
│   └── clean_md/                # raw_md → cleaned md
└── pages.txt                    # page list for live Notion fetch
memory/
├── MEMORY.md                    # long-term distilled context
└── YYYY-MM-DD.md                # daily conversation logs
```

---

## 🔄 Pipeline stages

### 1 · ETL — Notion / files → raw markdown

```bash
python scripts/run_etl.py
```

Fetches from Notion (`LOAD_MODE=notion`) or reads local files (`LOAD_MODE=files`). Writes to `data/raw/raw_md/`.

> **First run:** set `ETL_PAGE_NAME="Some Page"` to test the Notion connection on a single page before pulling your whole workspace. Notion rate limits are aggressive on large workspaces.

### 2 · Cleaning — raw → clean markdown

```bash
python scripts/run_clean_md.py
```

LLM-based cleanup of `data/raw/raw_md/` → `data/clean/clean_md/`.

### 3 · Marker — PDF + image → markdown

```bash
python scripts/run_marker.py
```

Converts PDFs and images into markdown. Independent of ETL — only run when you have new source files.

Env vars: `MARKER_STEP` (`pdfs` / `images` / `all`), `MARKER_TEST_SUBDIR` (debug subset).

> **Alternatives that avoid the openai conflict:** PyMuPDF4LLM (lightest, no OCR), MinerU, Kreuzberg, Docling.

### 4 · RAG indexing — clean markdown → Qdrant

```bash
python scripts/run_rag.py
```

Applies sentence-aware chunking → hybrid embeddings → optional reranking → Qdrant upload with RRF fusion. Collection visible at `http://localhost:32768/dashboard#/collections`.

Key env vars: `FORCE_REINDEX=true` (schema changes), `RETRIEVAL_TOP_N=10`, `RERANK_TOP_K=3`.

> **Qdrant indexing note:** `indexed_vectors_count: 0` with `points_count > 0` is normal — HNSW indexing is deferred until data exceeds `indexing_threshold`. Lower the threshold in `pipelines/rag/indexer.py` for immediate indexing.

### 5 · Agent — query your second brain

#### CLI REPL (no memory)

```bash
python -m assistant.cli
python -m assistant.cli "What should I focus on this quarter?"
```

#### Streamlit GUI (with memory)

```bash
streamlit run assistant/app.py
```

Always run from the repo root.

![Streamlit UI](https://github.com/michailmitsakis/notion-second-brain/blob/main/images/Streamlit%20UI.png)

---

## 🧠 Memory

File-based memory at `memory/`:

- `MEMORY.md` — long-term distilled context (curated facts, preferences)
- `YYYY-MM-DD.md` — daily conversation logs

Each turn is appended as:

```
### HH:MM
User: ...
Assistant: ...
```

The agent loads `MEMORY.md` + the most recent `RECENT_LOG_DAYS` daily logs into its system prompt per call, implemented via per-call `Agent` instantiation (pydantic-ai 1.x freezes `system_prompt` after construction).

Toggle: `ENABLE_MEMORY=true|false`.

---

## 📊 Evaluations

Two runners, same anchored rubric and golden test set — run either or both for cross-validation:

| Script | Framework | Purpose |
|---|---|---|
| `python -m evals.run_evals` | Hand-rolled | Canonical. 4-criterion rubric (relevance, correctness, citation_quality, safety), LLM-as-judge via Ollama. |
| `python -m evals.run_pydantic_evals` | pydantic_evals | Same dataset + rubric via `Evaluator` + `EvaluationReason`. Runs all 3 tiers by default. |

Results written to `evals/results/*.json`. Set judge model via `.env`: `JUDGE_MODEL=qwen3:8b`.

### Eval methodology

Follows the frameworks in `extras/llm-eval-patterns.md` and `extras/prompt-eval-designer.md` — moving from vibes-based to statistically anchored evaluation (pointwise rubrics, 3-tier test suites, CI gates). Built on: [ai-engineering-from-scratch](https://github.com/rohitg00/ai-engineering-from-scratch/blob/main/phases/11-llm-engineering/10-evaluation/docs/en.md).

---

## 🔭 Observability — [Arize Phoenix](https://arize.com/docs/phoenix) (optional)

`extras/run_phoenix.py` wires Phoenix OTel tracing. Every `agent.run(...)`, LLM call, and tool call appears as a span in the Phoenix UI at `http://localhost:6006`.

```bash
pip install openinference-instrumentation-pydantic-ai opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-api
docker compose up -d phoenix
python -m extras.run_phoenix "your query here"
```

![Phoenix Trace UI](https://github.com/michailmitsakis/notion-second-brain/blob/main/images/Phoenix%20working.png)

> Don't run Phoenix tracing and DeepEval's `DeepEvalInstrumentationSettings` simultaneously — both wrap the same pydantic-ai OTel hooks.

---

## 🐳 Docker Compose

```bash
docker compose up -d          # starts both Qdrant and Phoenix
docker compose up -d qdrant   # Qdrant only
docker compose up -d phoenix  # Phoenix only
```

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "32768:6333"
      - "6334:6334"
    volumes:
      - ./data/qdrant:/qdrant/storage
    restart: unless-stopped

  phoenix:
    image: arizephoenix/phoenix:latest
    ports:
      - "6006:6006"
      - "4317:4317"
    volumes:
      - ./data/phoenix:/mnt/data
    restart: unless-stopped
```

---

## ⚠️ OpenAI conflict (marker-pdf vs pydantic-ai)

The single biggest setup friction. `pip check` surfaces it as:

```
marker-pdf 1.10.2 requires openai<2.0.0, but you have openai 2.41.0
```

- `marker-pdf` pins `openai<2.0.0`
- `pydantic-ai` pulls `openai>=2.0.0`
- They are incompatible in the same environment

**Workaround:** swap versions as needed (~10 seconds):

```bash
# Before running marker / ETL scripts
pip install "openai<2.0.0,>=1.65.2"

# Before running the agent / evals
pip install "openai>=2.0.0"
```

Only needed when you have new PDFs or images to OCR. Re-indexing already-cleaned markdown (`run_rag.py`) doesn't touch marker and doesn't need the swap.

### Decision tree

```
New PDFs / images to OCR?          → marker mode  (pip install openai<2, run_marker.py)
Re-index existing cleaned markdown? → agent mode   (pip install openai>=2, run_rag.py)
Chat / develop / run evals?        → agent mode
```

---

## 🛠️ Troubleshooting

| Symptom | Fix |
|---|---|
| Notion returns nothing | Pages not shared with integration — open page → ⋯ → Connections → add integration |
| Notion rate limits / 429s | Use `ETL_PAGE_NAME="Single Page"` for first-run validation |
| GPU offload (CPU %) | Set `OLLAMA_NUM_GPU=99` at shell level, or recreate model via Modelfile |
| pydantic-ai 404 against Ollama | Use `OllamaProvider(base_url="http://localhost:11434/v1")` — the `ollama:<model>` shorthand routes to the wrong path |
| Streamlit import errors | Always `cd` to repo root before running |
| Eval JSON missing | Default output: `evals/results/*.json`; override with `--output PATH` |
| Memory not used | Check `ENABLE_MEMORY=true`; `memory/MEMORY.md` is auto-created on first run |
| Stale chunks after schema change | Set `FORCE_REINDEX=true` to drop and rebuild the Qdrant collection |

---

## License

MIT — see [LICENSE](https://github.com/michailmitsakis/notion-second-brain/blob/main/LICENSE). This repo is meant to be used, adapted and improved upon based on individual user needs and system capabilities.