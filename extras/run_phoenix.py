"""
Phoenix observability — instrument the Second Brain agent and view traces
in the Phoenix UI at http://localhost:6006.

Usage:
    python -m extras.run_phoenix
    python -m extras.run_phoenix "what is in my second brain?"

Traces appear at http://localhost:6006 under the "default" project.

NOTE: Don't run this while DeepEval's DeepEvalInstrumentationSettings is
also active — both wrap pydantic-ai's OTel hooks and will double-emit spans.
"""
from __future__ import annotations

import asyncio
import os
import sys

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.pydantic_ai import OpenInferenceSpanProcessor

from pydantic_ai import Agent
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.models.openai import OpenAIChatModel

from assistant.agent import SYSTEM_PROMPT
from assistant.tools import retrieve_knowledge, fetch_notion_page


# ── OTel setup ───────────────────────────────────────────────────────────────

_ENDPOINT = os.environ.get(
    "PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006"
) + "/v1/traces"

_tracer_provider = TracerProvider()
trace.set_tracer_provider(_tracer_provider)

_exporter = OTLPSpanExporter(endpoint=_ENDPOINT)  # no auth header for local instance
_tracer_provider.add_span_processor(OpenInferenceSpanProcessor())
_tracer_provider.add_span_processor(SimpleSpanProcessor(_exporter))


# ── Agent ─────────────────────────────────────────────────────────────────────

def build_agent() -> Agent:
    model_name = os.getenv("OLLAMA_MODEL", "gemma4:latest")
    if model_name.startswith("ollama:"):
        model_name = model_name.split(":", 1)[1]
    provider = OllamaProvider(base_url="http://localhost:11434/v1")
    model = OpenAIChatModel(model_name=model_name, provider=provider)
    return Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[retrieve_knowledge, fetch_notion_page],
        instrument=True,  # ← tells pydantic-ai to emit OTel spans
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is in my second brain?"

    agent = build_agent()
    print(f"Running agent with query: {query!r}")
    print(f"Traces will appear at http://localhost:6006\n")

    result = asyncio.run(agent.run(query))
    print(result.output)