# assistant/tools.py
import os
from typing import List

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FusionQuery,
    Fusion,
    Prefetch,
    SparseVector,
)
from notion_client import Client
from notion_to_md import NotionToMarkdown
from dotenv import load_dotenv

from pipelines.etl.notion_loader import parse_page_id
from pipelines.rag.embeddings import get_dense_embedder, get_sparse_embedder
from pipelines.rag.reranker import get_reranker


COLLECTION_NAME = "second_brain"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:32768")

# Hybrid search fetches this many fused candidates; if rerank is enabled,
# the cross-encoder narrows them down to RERANK_TOP_K. If rerank is
# disabled, this is also the final number returned to the LLM.
RETRIEVAL_TOP_N = 10
RERANK_TOP_K = 3


def _env_bool(name: str, default: bool = False) -> bool:
    """Parse a boolean env var. Truthy: 1/true/yes/on. Falsy: 0/false/no/off/''."""
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def _qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


# ── Tool 1: Hybrid search + (optional) cross-encoder rerank ─────────────────

def retrieve_knowledge(query: str) -> str:
    """Search the Second Brain knowledge base for relevant chunks.

    Pipeline:
      1. Embed the query with the dense embedder (Ollama nomic-embed-text)
         and the sparse embedder (fastembed BM25).
      2. Qdrant does hybrid retrieval (dense + sparse) with Reciprocal
         Rank Fusion, returning the top RETRIEVAL_TOP_N fused candidates.
      3. (Optional, controlled by ENABLE_RERANK env var) Re-score with
         a cross-encoder (BAAI/bge-reranker-v2-m3) and return top
         RERANK_TOP_K. If disabled, return all RETRIEVAL_TOP_N fused
         candidates as-is.
      4. Format each chunk with its source title, url (if any), and
         relevance score for the LLM to consume.
    """
    if not query.strip():
        return "Empty query."

    dense = get_dense_embedder()
    sparse = get_sparse_embedder()

    dense_vec = dense.embed(query)
    sparse_vec = sparse.embed(query)
    qdrant_sparse = SparseVector(
        indices=list(sparse_vec.indices),
        values=list(sparse_vec.values),
    )

    client = _qdrant()
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(query=dense_vec, using=DENSE_VECTOR_NAME, limit=RETRIEVAL_TOP_N),
            Prefetch(query=qdrant_sparse, using=SPARSE_VECTOR_NAME, limit=RETRIEVAL_TOP_N),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=RETRIEVAL_TOP_N,
        with_payload=True,
    )
    points = getattr(response, "points", response) or []

    if not points:
        return "No relevant results found in the knowledge base."

    candidates = []
    for p in points:
        payload = p.payload or {}
        content = payload.get("content", "")
        chunk_id = payload.get("chunk_id", str(getattr(p, "id", "")))
        if content:
            candidates.append((chunk_id, content))

    if _env_bool("ENABLE_RERANK", default=True):
        reranker = get_reranker()
        reranked = reranker.rerank(query, candidates, top_k=RERANK_TOP_K)
        payload_by_id = {
            (p.payload or {}).get("chunk_id"): p.payload
            for p in points
            if p.payload and p.payload.get("chunk_id")
        }
        parts = []
        for chunk_id, content, score in reranked:
            payload = payload_by_id.get(chunk_id, {})
            title = payload.get("title", "unknown")
            url = payload.get("url", "")
            url_str = f" ({url})" if url else ""
            parts.append(
                f"[{title}]{url_str} (rerank score: {round(score, 3)})\n{content}"
            )
    else:
        # No rerank: take top RETRIEVAL_TOP_N fused candidates as-is, in
        # Qdrant's RRF order. RRF score is a relative rank signal, not
        # normalized, so we surface it as-is.
        payload_by_id = {
            (p.payload or {}).get("chunk_id"): p.payload
            for p in points
            if p.payload and p.payload.get("chunk_id")
        }
        parts = []
        for p in points[:RERANK_TOP_K]:
            payload = p.payload or {}
            title = payload.get("title", "unknown")
            url = payload.get("url", "")
            url_str = f" ({url})" if url else ""
            rrf_score = round(getattr(p, "score", 0.0), 3)
            content = payload.get("content", "")
            parts.append(
                f"[{title}]{url_str} (rrf score: {rrf_score})\n{content}"
            )

    return "\n\n---\n\n".join(parts)


# ── Tool 2: Live fetch a specific Notion page (expensive) ──────────────────

def fetch_notion_page(notion_url: str) -> str:
    """
    Fetch the latest content of a specific Notion page by URL or page ID.

    IMPORTANT: This operation is potentially slow and resource-intensive. It
    performs a full notion2md conversion of the provided page (the entire
    page, not per-subpage), so it should only be used when the exact Notion
    page URL or ID is available and the user explicitly requests the live
    contents. For routine lookups prefer `retrieve_knowledge` which queries
    the local Qdrant index (much faster).

    Use cases for this tool:
    - You need the absolutely latest, live page content and have the page URL/ID.
    - There is no local Qdrant collection available and you must fetch live.

    This tool requires the `NOTION_ASSISTANT_AUTH_TOKEN` environment variable
    to be set (loaded from `.env` at call time). The function returns the
    page markdown string on success or a clear error message on failure.
    """
    # Ensure .env values are loaded at call time (avoids import-order issues)
    load_dotenv()

    try:
        auth_key = os.environ.get("NOTION_ASSISTANT_AUTH_TOKEN")
        if not auth_key:
            return "Missing NOTION_ASSISTANT_AUTH_TOKEN; cannot fetch Notion page."

        notion = Client(auth=auth_key)
        n2m = NotionToMarkdown(notion)
        page_id = parse_page_id(notion_url)
        md_blocks = n2m.page_to_markdown(page_id)
        return n2m.to_markdown_string(md_blocks).get("parent", "")
    except Exception as e:
        return f"Failed to fetch Notion page: {e}"