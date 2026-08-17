"""Retrieval and behaviour metrics.

Deliberately dependency-free and pure: every function takes plain lists and
returns a float, so each one can be checked by hand against a worked example.
That matters more than usual here — a quietly wrong metric does not crash, it
just produces a confident number that sends you optimising the wrong thing.

Retrieval metrics are computed from `supporting_chunk_ids` in the gold set,
which are facts (each question was written from a known chunk), so these numbers
carry no LLM-judge caveat.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Fraction of the relevant chunks that appear in the top k.

    With one relevant chunk per question this is 0.0 or 1.0 — "did we find it".
    """
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(set(relevant))


def reciprocal_rank(retrieved: Sequence[int], relevant: Sequence[int]) -> float:
    """1/rank of the first relevant chunk; 0.0 if none was retrieved.

    Rewards putting the right chunk first rather than merely somewhere in the
    list — which matters because the generator reads the top of the list most
    carefully.
    """
    relevant_set = set(relevant)
    for index, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant_set:
            return 1.0 / index
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Binary-gain nDCG.

    With a single relevant chunk this is a monotone function of rank and adds
    little over MRR — report both, but do not read much into small gaps.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    gains = [1.0 if cid in relevant_set else 0.0 for cid in retrieved[:k]]
    ideal = [1.0] * min(len(relevant_set), k)
    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def precision_at_k(retrieved: Sequence[int], relevant: Sequence[int], k: int) -> float:
    """Fraction of the top k that is relevant.

    Bounded above by len(relevant)/k, so with one relevant chunk and k=6 the
    ceiling is 0.167. Compare configurations against each other, never against
    1.0.
    """
    if k <= 0:
        return 0.0
    window = retrieved[:k]
    if not window:
        return 0.0
    return len(set(window) & set(relevant)) / len(window)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` is 0-100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0
