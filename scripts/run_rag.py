"""Run RAG pipeline: load cleaned docs, chunk, and index.

This script expects cleaned documents under `data/clean` (output
from `run_clean_md.py`) and optional crawled JSON docs in
`data/crawled`.

It does NOT re-process PDF/image markdown; those are produced by
`run_marker.py` and are intentionally left out of the cleaning
pipeline per project conventions.
"""
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from pipelines.models import RawDocument
from pipelines.etl.notion_loader import load_from_files
from pipelines.rag.chunker import chunk_document
from pipelines.rag.indexer import index_chunks

try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm isn't installed — behave like a passthrough iterator
    def tqdm(iterable, **kwargs):
        return iterable

def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Truthy: 1/true/yes/on. Falsy: 0/false/no/off/''."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def main():
    # Quick verification: count markdown files per subfolder under data/clean
    clean_root = Path("data/clean")
    if clean_root.exists():
        total_md = 0
        for sub in sorted([p for p in clean_root.iterdir() if p.is_dir()], key=lambda p: p.name):
            count = len(list(sub.rglob("*.md")))
            total_md += count
            print(f"Found {count} .md files in {sub.name}")
        print(f"Total .md files under data/clean: {total_md}")

    # 1. Load cleaned docs (Notion + local cleaned markdown)
    notion_docs = load_from_files(clean_root)
    print(f"Loaded {len(notion_docs)} documents via load_from_files from {clean_root}")
    if clean_root.exists() and total_md > len(notion_docs):
        print("Warning: number of markdown files under data/clean is greater than loaded documents.\n"
              "load_from_files may not register all files or some files are non-markdown structured differently.")

    # 2. Load any crawled JSON documents
    crawled_docs = []
    for f in Path("data/crawled").glob("*.json"):
        try:
            crawled_docs.append(RawDocument.model_validate_json(f.read_text(encoding="utf-8")))
        except Exception:
            # skip malformed crawl files but continue processing
            print(f"Warning: failed to parse crawled doc {f}; skipping")

    all_docs = notion_docs + crawled_docs
    print(f"Total docs: {len(all_docs)}")

    # 2.5. FORCE_REINDEX: drop the Qdrant collection before re-indexing.
    # Set FORCE_REINDEX=1 in .env or the shell when the chunker/embedder
    # changes so old-version chunks don't sit alongside new ones.
    force_reindex = _env_bool("FORCE_REINDEX", default=False)
    if force_reindex:
        print("FORCE_REINDEX=1: Qdrant collection will be dropped and recreated.")

    # 3. Chunk + index (show progress)
    for doc in tqdm(all_docs, desc="Indexing documents"):
        # TODO: Add context back in --> Generating contextual prefixes with an
        # LLM per chunk multiplies runtime (e.g., one Ollama generate per chunk),
        # so it can be far more expensive than embedding alone.
        chunks = chunk_document(doc, add_context=False)
        # Index the chunks (embeddings + vector store)
        index_chunks(chunks, force_reindex=force_reindex)
        # Only force on the first batch; subsequent docs add to the fresh collection.
        force_reindex = False

    print("RAG pipeline complete.")


if __name__ == "__main__":
    main()