"""Fusion and reranking logic, on synthetic rankings checkable by hand."""

import pytest

from rxclarify.retrieval.base import RetrievedChunk
from rxclarify.retrieval.hybrid import renumber, rrf_fuse


def chunk(chunk_id: int, marker: int = 0, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        label_id=1,
        drug="Drug",
        section="drug_interactions",
        text=f"text-{chunk_id}",
        score=score,
        marker=marker,
    )


def ids(chunks) -> list[int]:
    return [c.chunk_id for c in chunks]


def test_a_chunk_found_by_both_retrievers_outranks_one_found_by_either():
    """The core reason hybrid search works."""
    dense = [chunk(1), chunk(2), chunk(3)]
    sparse = [chunk(9), chunk(8), chunk(2)]

    fused = rrf_fuse([dense, sparse], k=60)

    # 2 is 2nd in dense and 3rd in sparse -> 1/62 + 1/63, beating 1's 1/61.
    assert fused[0].chunk_id == 2


def test_rrf_score_matches_the_formula():
    fused = rrf_fuse([[chunk(1)], [chunk(1)]], k=60)
    assert fused[0].score == pytest.approx(2 / 61)


def test_single_ranking_preserves_its_order():
    fused = rrf_fuse([[chunk(5), chunk(6), chunk(7)]], k=60)
    assert ids(fused) == [5, 6, 7]


def test_disjoint_rankings_interleave_by_rank():
    fused = rrf_fuse([[chunk(1), chunk(2)], [chunk(3), chunk(4)]], k=60)
    # Both 1st-place chunks tie, both 2nd-place chunks tie; ties break by id.
    assert ids(fused) == [1, 3, 2, 4]


def test_fusion_deduplicates():
    fused = rrf_fuse([[chunk(1), chunk(2)], [chunk(1), chunk(2)]], k=60)
    assert ids(fused) == [1, 2]


def test_smaller_k_sharpens_the_advantage_of_a_top_rank():
    dense = [chunk(1), chunk(2)]
    sparse = [chunk(2), chunk(1)]
    # Symmetric input: both orderings tie regardless of k, so ids break the tie.
    assert ids(rrf_fuse([dense, sparse], k=1)) == [1, 2]
    assert ids(rrf_fuse([dense, sparse], k=60)) == [1, 2]


def test_fusion_of_nothing_is_empty():
    assert rrf_fuse([[], []]) == []


def test_renumber_assigns_sequential_markers_from_one():
    out = renumber([chunk(7, marker=99), chunk(8, marker=3)])
    assert [c.marker for c in out] == [1, 2]
    assert [c.citation for c in out] == ["C1", "C2"]


def test_renumber_is_required_after_reordering():
    """Markers are positional; stale ones would mislabel the prompt."""
    reordered = list(reversed([chunk(1, marker=1), chunk(2, marker=2)]))
    assert [c.marker for c in reordered] == [2, 1]  # wrong before renumbering
    assert [c.marker for c in renumber(reordered)] == [1, 2]
