# pipelines/rag/embeddings.py
"""
Dense and sparse embedding wrappers used by both the indexer and the
retrieval tool.

The dense embedder is `nomic-embed-text` served by Ollama (unchanged from
the previous pipeline) — 768-dim cosine vectors. The sparse embedder is
`Qdrant/bm25-multilingual` via `fastembed`, which returns vectors in a
format Qdrant accepts directly as named sparse vectors.

Both embedders are lazy-loaded on first use so that `import` does not
trigger network calls or model downloads.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable

import ollama
from fastembed import SparseTextEmbedding


# ── Dense (Ollama nomic-embed-text) ──────────────────────────────────────────

class DenseEmbedder:
    """Thin wrapper around `ollama.embeddings` for the dense half of hybrid."""

    def __init__(
        self,
        model: str = "nomic-embed-text:latest",
        num_ctx: int = 8192,
    ) -> None:
        self.model = model
        self.num_ctx = num_ctx

    def embed(self, text: str) -> list[float]:
        """Embed a single text. Retries on context-length errors with truncation."""
        for limit in (None, 3000, 2000, 1200):
            prompt = text if limit is None else text[:limit]
            try:
                return ollama.embeddings(
                    model=self.model,
                    prompt=prompt,
                    options={"num_ctx": self.num_ctx},
                )["embedding"]
            except Exception as e:
                msg = str(e).lower()
                if "exceeds the context length" not in msg:
                    raise
        raise RuntimeError(
            "Dense embedding input exceeded context length even after truncation."
        )

    def embed_batch(self, texts: Iterable[str]) -> list[list[float]]:
        """Sequential embed; Qdrant upsert is batched separately."""
        return [self.embed(t) for t in texts]


@lru_cache(maxsize=1)
def get_dense_embedder() -> DenseEmbedder:
    return DenseEmbedder(
        model=os.getenv("DENSE_EMBED_MODEL", "nomic-embed-text:latest"),
    )


# ── Sparse (fastembed BM25-multilingual) ────────────────────────────────────

class SparseEmbedder:
    """BM25-style sparse embedder using `fastembed` and the
    `Qdrant/bm25-multilingual` model. The returned objects are
    `fastembed.SparseEmbedding` namedtuples with `.indices` and `.values`
    that map directly to Qdrant's `SparseVector`.
    """

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        self.model = model
        # `SparseTextEmbedding` is lazy: it only downloads the model on
        # the first embed call, not on instantiation.
        self._encoder: SparseTextEmbedding | None = None

    def _encoder_instance(self) -> SparseTextEmbedding:
        if self._encoder is None:
            self._encoder = SparseTextEmbedding(model_name=self.model)
        return self._encoder

    def embed(self, text: str):
        """Embed a single text. Returns a `SparseEmbedding` w/ .indices/.values."""
        results = list(self._encoder_instance().embed([text]))
        return results[0]

    def embed_batch(self, texts: Iterable[str]):
        """Embed a batch of texts. Returns a list of `SparseEmbedding`."""
        return list(self._encoder_instance().embed(list(texts)))


@lru_cache(maxsize=1)
def get_sparse_embedder() -> SparseEmbedder:
    return SparseEmbedder(
        model=os.getenv("SPARSE_EMBED_MODEL", "Qdrant/bm25"),
    )