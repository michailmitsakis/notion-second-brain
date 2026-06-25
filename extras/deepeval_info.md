# DeepEval Integration Notes

`extras/run_deepeval.py` is a showcase script — not a canonical eval runner. It demonstrates DeepEval's pydantic-ai integration and `deepeval inspect` TUI locally, without requiring a Confident AI cloud account.

The two canonical runners (`evals/run_evals.py` and `evals/run_pydantic_evals.py`) cover all eval needs with a 4-criterion anchored rubric. DeepEval is kept as an optional extra for trace-level inspection.

## Setup

```bash
pip install deepeval openinference-instrumentation-pydantic-ai opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-api
deepeval set-ollama --model=qwen3:8b --save=dotenv
```

`set-ollama` writes `OLLAMA_MODEL_NAME`, `USE_OLLAMA_MODEL=1`, and `OLLAMA_BASE_URL` to `.env.local`. After running it, manually fix the base URL:

```
# .env.local
OLLAMA_BASE_URL=http://localhost:11434/v1   # /v1 suffix required — set-ollama omits it
```

## Running

```bash
set DEEPEVAL_DISABLE_TIMEOUTS=1
python -m extras.run_deepeval --tier adversarial
```

`DEEPEVAL_DISABLE_TIMEOUTS=1` is necessary because local Ollama inference is slow enough to exceed DeepEval's default 88s per-attempt timeout. In CI, use explicit overrides instead:

```bash
set DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=600
set DEEPEVAL_PER_TASK_TIMEOUT_SECONDS_OVERRIDE=1200
```

After a run, inspect traces in the terminal TUI:

```bash
pip install 'deepeval[inspect]'
deepeval inspect
```

`evals_iterator()` writes a rolling snapshot to `.deepeval/.latest_run_full.json` automatically. The zero-arg form picks it up without needing a path.

## Known friction points

**`Agent(instrument=...)` deprecation warning.** pydantic-ai 1.105+ warns that `instrument=DeepEvalInstrumentationSettings(...)` is deprecated in favour of `capabilities=[Instrumentation(...)]`. However, DeepEval's installed package (≤0.1.16) does not export `Instrumentation` — only `DeepEvalInstrumentationSettings`. The old form still works, just noisy. Suppress with:

```python
import warnings
from pydantic_ai.exceptions import PydanticAIDeprecationWarning
warnings.filterwarnings("ignore", category=PydanticAIDeprecationWarning)
```

**`FaithfulnessMetric` requires `retrieval_context`.** The metric needs the chunks your retriever returned, which aren't threaded through `evals_iterator` automatically. This would require instrumenting `retrieve_knowledge` to emit chunks into the OTel trace via `update_current_span(metadata=...)`. For the showcase, `FaithfulnessMetric` is dropped and only `AnswerRelevancyMetric` is used — it has no such requirement.

**pydantic-ai 404 against Ollama.** DeepEval's `set-ollama` writes `OLLAMA_BASE_URL=http://localhost:11434/` without the `/v1` suffix. pydantic-ai's OpenAI-compat layer needs `/v1`. Fix: use `OllamaProvider(base_url="http://localhost:11434/v1")` explicitly in `_build_agent()` rather than the `ollama:<model>` shorthand.

**`OpenAIModel` renamed.** pydantic-ai renamed `OpenAIModel` to `OpenAIChatModel` to distinguish it from `OpenAIResponsesModel`. Use `OpenAIChatModel` with `OllamaProvider` — Ollama exposes a Chat Completions-compatible API, not the Responses API.

**No metric scores in summary.** The run completes but scores show as 0 / empty if `dataset.evaluate(task)` is called before the task is awaited. The fix is to `await task` before calling `evaluate()`, which is what the current script does. If you still see empty scores, check that the judge model (`qwen3:8b`) is pulled and responding — a judge timeout silently produces null scores.

**`deepeval inspect` TUI prompt after each run.** Disable with:

```bash
set DEEPEVAL_NO_INSPECT_PROMPT=1
```

Then run `deepeval inspect` manually when you want to review traces.

## What DeepEval adds over the canonical runners

The canonical runners (`run_evals.py`, `run_pydantic_evals.py`) score outputs against a hand-written rubric. DeepEval adds:

- **Trace-level inspection** via `deepeval inspect` — see individual LLM spans, tool calls, latency, and per-span metric scores in a terminal TUI
- **`AnswerRelevancyMetric`** — model-graded relevance without needing expected outputs
- **`FaithfulnessMetric`** (when retrieval context is plumbed in) — scores whether the answer is grounded in retrieved chunks
- **Confident AI dashboard** (optional, cloud) — team-shared reports and continuous monitoring; not required for local use

For a 16-case eval suite running entirely locally, the canonical runners are sufficient. DeepEval becomes more valuable if the suite grows, multiple contributors need to share results, or you want automatic regression detection across releases.