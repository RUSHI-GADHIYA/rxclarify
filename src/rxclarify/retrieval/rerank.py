"""Cross-encoder reranking — the second stage of two-stage retrieval.

The pattern: retrieve wide and cheap, then rerank narrow and expensive.

A bi-encoder (what `dense.py` uses) embeds the query and the passage
*separately*, so it never sees them together and the comparison is a dot
product between two summaries. A cross-encoder reads query and passage in one
pass and scores the pair directly, which is far more accurate and far too slow
to run over 2,327 chunks. Running it over ~30 candidates is affordable, and
that is the whole design.

Runs locally on CPU via fastembed — no API, no key, no per-query cost.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from rxclarify.retrieval.base import RetrievedChunk, Retriever
from rxclarify.retrieval.hybrid import renumber

# 80 MB, and fast enough on CPU for ~30 candidates. `BAAI/bge-reranker-base` is
# stronger but 1 GB and noticeably slower — worth trying only if the small model
# proves to be the bottleneck on quality.
RERANKER_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 30


@lru_cache(maxsize=2)
def get_reranker(model_name: str = RERANKER_MODEL):
    # Imported lazily: loading pulls in onnxruntime and, on first use, downloads
    # the model. Nothing that merely imports this module should pay that.
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(model_name=model_name)


class RerankingRetriever:
    """Wraps any `Retriever`, reranking its candidates with a cross-encoder."""

    def __init__(
        self,
        base: Retriever,
        *,
        candidates: int = CANDIDATES,
        model_name: str = RERANKER_MODEL,
    ) -> None:
        self._base = base
        self.candidates = candidates
        self.model_name = model_name

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        depth = max(self.candidates, top_k)
        candidates = self._base.retrieve(query, top_k=depth)
        if not candidates:
            return []

        scores = list(get_reranker(self.model_name).rerank(query, [c.text for c in candidates]))

        # Cross-encoder scores are logits on their own scale, unrelated to the
        # base retriever's — they replace the score rather than adjusting it.
        rescored = [replace(c, score=float(s)) for c, s in zip(candidates, scores, strict=True)]
        rescored.sort(key=lambda c: -c.score)
        return renumber(rescored[:top_k])
