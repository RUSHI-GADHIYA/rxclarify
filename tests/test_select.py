"""Candidate selection — the guard against ophthalmic drops and combo products."""

from rxclarify.ingest.select import is_combination, score_candidate, select_best


def _label(
    *,
    generics: list[str],
    routes: list[str],
    interactions: bool = True,
    body: int = 4000,
) -> dict:
    payload: dict = {
        "set_id": ["s"],
        "openfda": {"generic_name": generics, "route": routes},
        "dosage_and_administration": ["x" * body],
    }
    if interactions:
        payload["drug_interactions"] = ["Avoid strong CYP3A4 inhibitors."]
    return payload


def test_is_combination_detects_multiple_generic_names():
    assert is_combination(_label(generics=["SITAGLIPTIN", "METFORMIN"], routes=["ORAL"]))


def test_is_combination_detects_conjoined_single_name():
    assert is_combination(
        _label(generics=["SITAGLIPTIN AND METFORMIN HYDROCHLORIDE"], routes=["ORAL"])
    )


def test_is_combination_false_for_single_ingredient():
    assert not is_combination(_label(generics=["METFORMIN HYDROCHLORIDE"], routes=["ORAL"]))


def test_oral_single_ingredient_beats_ophthalmic():
    """The CILOXAN case: eye drops must not win a ciprofloxacin search."""
    drops = _label(generics=["CIPROFLOXACIN HYDROCHLORIDE"], routes=["OPHTHALMIC"], body=500)
    tablet = _label(generics=["CIPROFLOXACIN HYDROCHLORIDE"], routes=["ORAL"])

    assert select_best([drops, tablet]) is tablet


def test_single_ingredient_beats_combination_product():
    """The metformin case: plain metformin must win over SITAGLIPTIN AND METFORMIN."""
    combo = _label(generics=["SITAGLIPTIN AND METFORMIN HYDROCHLORIDE"], routes=["ORAL"])
    plain = _label(generics=["METFORMIN HYDROCHLORIDE"], routes=["ORAL"])

    assert select_best([combo, plain]) is plain


def test_single_ingredient_wins_even_when_combination_is_richer():
    """Ingredient purity outranks label size — the richness bonus is capped."""
    combo = _label(generics=["A AND B"], routes=["ORAL"], body=500_000)
    plain = _label(generics=["A"], routes=["ORAL"], body=1000)

    assert select_best([combo, plain]) is plain


def test_interaction_section_breaks_a_tie():
    without = _label(generics=["DRUG"], routes=["ORAL"], interactions=False)
    with_ = _label(generics=["DRUG"], routes=["ORAL"], interactions=True)

    assert score_candidate(with_) > score_candidate(without)
    assert select_best([without, with_]) is with_


def test_select_best_returns_none_for_no_candidates():
    assert select_best([]) is None


def test_select_best_falls_back_to_a_poor_candidate_when_it_is_the_only_one():
    drops = _label(generics=["X"], routes=["OPHTHALMIC"], interactions=False, body=100)
    assert select_best([drops]) is drops
