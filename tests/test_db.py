import pytest

from rxclarify.db import to_pgvector
from rxclarify.ingest.fetch_labels import load_drug_list


def test_to_pgvector_renders_bracketed_csv():
    assert to_pgvector([1.0, -0.5]) == "[1.0,-0.5]"


def test_to_pgvector_accepts_ints_and_emits_floats():
    assert to_pgvector([1, 2]) == "[1.0,2.0]"


def test_drug_list_loads_flat_deduplicated_and_lowercased():
    generics = load_drug_list()

    assert len(generics) >= 40, "corpus should be ~50 drugs"
    assert len(generics) == len(set(generics)), "duplicate generic names"
    assert all(g == g.strip().lower() for g in generics)
    assert "warfarin sodium" in generics


@pytest.mark.parametrize("expected", ["clarithromycin", "fluconazole", "levothyroxine sodium"])
def test_drug_list_contains_key_interaction_drugs(expected):
    """These carry the interaction content the Phase 2 gold questions rely on."""
    assert expected in load_drug_list()
