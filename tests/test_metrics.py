"""Metric correctness, checked against worked examples.

A wrong metric does not crash — it produces a confident number that sends you
optimising the wrong thing. Every expected value below is derived by hand.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import (  # noqa: E402
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    rate,
    recall_at_k,
    reciprocal_rank,
)

RETRIEVED = [10, 20, 30, 40, 50, 60]


def test_recall_finds_the_chunk_inside_k():
    assert recall_at_k(RETRIEVED, [30], k=6) == 1.0


def test_recall_misses_the_chunk_outside_k():
    assert recall_at_k(RETRIEVED, [30], k=2) == 0.0


def test_recall_with_multiple_relevant_is_a_fraction():
    # 20 and 30 are inside the top 3; 99 is not -> 2 of 3.
    assert recall_at_k(RETRIEVED, [20, 30, 99], k=3) == pytest.approx(2 / 3)


def test_recall_with_no_relevant_chunks_is_zero():
    assert recall_at_k(RETRIEVED, [], k=6) == 0.0


@pytest.mark.parametrize(
    ("relevant", "expected"),
    [([10], 1.0), ([20], 0.5), ([30], 1 / 3), ([60], 1 / 6), ([999], 0.0)],
)
def test_reciprocal_rank_is_one_over_position(relevant, expected):
    assert reciprocal_rank(RETRIEVED, relevant) == pytest.approx(expected)


def test_reciprocal_rank_uses_the_first_hit_only():
    assert reciprocal_rank(RETRIEVED, [50, 20]) == pytest.approx(0.5)


def test_ndcg_is_one_when_the_relevant_chunk_ranks_first():
    assert ndcg_at_k(RETRIEVED, [10], k=6) == pytest.approx(1.0)


def test_ndcg_at_rank_two_matches_the_hand_calculation():
    # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(RETRIEVED, [20], k=6) == pytest.approx(1 / math.log2(3))


def test_ndcg_is_zero_when_nothing_relevant_is_retrieved():
    assert ndcg_at_k(RETRIEVED, [999], k=6) == 0.0


def test_precision_ceiling_is_one_over_k_for_a_single_relevant_chunk():
    assert precision_at_k(RETRIEVED, [30], k=6) == pytest.approx(1 / 6)


def test_precision_of_an_empty_result_is_zero():
    assert precision_at_k([], [30], k=6) == 0.0


def test_mean_of_empty_is_zero_not_a_crash():
    assert mean([]) == 0.0


@pytest.mark.parametrize(
    ("pct", "expected"),
    [(50, 3), (100, 5), (0, 1)],
)
def test_percentile_uses_nearest_rank(pct, expected):
    assert percentile([1, 2, 3, 4, 5], pct) == expected


def test_percentile_of_empty_is_zero():
    assert percentile([], 50) == 0.0


def test_rate_guards_division_by_zero():
    assert rate(0, 0) == 0.0
    assert rate(3, 4) == 0.75
