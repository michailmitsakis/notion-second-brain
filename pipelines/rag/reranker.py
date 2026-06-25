# pipelines/rag/reranker.py
"""
Cross-encoder reranker used to re-score the top fused candidates from
Qdrant before returning them to the LLM.

We use `BAAI/bge-reranker-v2-m3` (multilingual, 568MB). The model is
lazy-loaded on first `rerank()` call so importing this module does not
trigger downloads or load torch into memory.
"""
from __future__ import annotations

import os
from typing import Sequence

from sentence_transformers import CrossEncoder


class Reranker:
    """Cross-encoder reranker. Fail loudly if the model cannot load."""

    DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.getenv("RERANKER_MODEL", self.DEFAULT_MODEL)
        self._model: CrossEncoder | None = None

    def _load(self) -> CrossEncoder:
        if self._model is None:
            # Hard fail (Q-L): if torch/transformers missing or download
            # fails, the pipeline must error visibly.
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: Sequence[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, str, float]]:
        """Re-score (query, text) pairs and return top_k.

        Args:
            query: the user's search query.
            candidates: sequence of (candidate_id, candidate_text) tuples.
                The candidate_id is opaque to the reranker and is passed
                through so callers can map scores back to their chunks.
            top_k: if None, return all candidates re-sorted by score.
        """
        if not candidates:
            return []

        model = self._load()
        pairs = [(query, text) for _id, text in candidates]
        scores = model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )
        if top_k is not None:
            ranked = ranked[:top_k]

        return [(cid, text, float(score)) for (cid, text), score in ranked]


# Module-level singleton so we only load the model once per process.
_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker