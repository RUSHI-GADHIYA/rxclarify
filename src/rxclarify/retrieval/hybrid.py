"""Hybrid retrieval: dense + sparse, fused with Reciprocal Rank Fusion.

**Why RRF and not weighted score blending.** Cosine similarity and `ts_rank_cd`
are on unrelated scales — cosine sits around 0.7-0.8 for everything here, while
ts_rank_cd is often below 0.01. Normalising them against each other requires a
weight you can only tune with labelled data, and that weight then silently
overfits to it. RRF throws the scores away and uses only *rank*, so there is
nothing to tune and nothing to overfit. It is what most production hybrid
search actually runs.

    score(d) = sum over retrievers of  1 / (k + rank(d))

`k` (60 by convention) damps the influence of the top rank so a single
retriever cannot dominate on its own.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import psycopg

from rxclarify.retrieval.base import RetrievedChunk, Retriever
from rxclarify.retrieval.dense import DenseRetriever
from rxclarify.retrieval.sparse import SparseRetriever

RRF_K = 60
CANDIDATES = 30


def rrf_fuse(
    rankings: Sequence[Sequence[RetrievedChunk]],
    *,
    k: int = RRF_K,
) -> list[RetrievedChunk]:
    """Fuse ranked lists into one, ordered by summed reciprocal rank.

    A chunk found by both retrievers outranks one found by either alone, which
    is the behaviour that makes hybrid search work.
    """
    scores: dict[int, float] = {}
    seen: dict[int, RetrievedChunk] = {}

    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            seen.setdefault(chunk.chunk_id, chunk)

    ordered = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [replace(seen[cid], score=score) for cid, score in ordered]


def renumber(chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
    """Reassign citation markers to 1..n after reordering.

    Markers are positional, so any stage that changes order must renumber —
    otherwise the prompt says [C1] for something that is no longer first.
    """
    return [replace(chunk, marker=i) for i, chunk in enumerate(chunks, start=1)]


class HybridRetriever:
    """Dense + sparse, fused by rank. Implements the `Retriever` protocol."""

    def __init__(
        self,
        conn: psycopg.Connection,
        *,
        candidates: int = CANDIDATES,
        rrf_k: int = RRF_K,
        # "strict" deliberately, even though it is the *worst* sparse retriever
        # standalone (recall@6 0.267 vs 0.644 for adaptive). Measured on the
        # gold set, hybrid recall@6 is 0.978 with strict and 0.933 with either
        # relaxed mode. Strict AND returns nothing on queries where lexical
        # matching has nothing useful to say, and contributing nothing to
        # fusion beats contributing six weak single-term matches that dilute
        # the dense ranking. Optimise the system, not the component.
        sparse_mode: str = "strict",
    ) -> None:
        self._dense: Retriever = DenseRetriever(conn)
        self._sparse: Retriever = SparseRetriever(conn, mode=sparse_mode)
        self.candidates = candidates
        self.rrf_k = rrf_k

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        # Pull deeper than top_k from each side: fusion can only reorder what it
        # is given, so a chunk ranked 20th by dense and 3rd by sparse is only
        # reachable if both lists run past top_k.
        depth = max(self.candidates, top_k)
        fused = rrf_fuse(
            [
                self._dense.retrieve(query, top_k=depth),
                self._sparse.retrieve(query, top_k=depth),
            ],
            k=self.rrf_k,
        )
        return renumber(fused[:top_k])
