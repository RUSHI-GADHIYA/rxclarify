"""Pick the most useful label when openFDA returns several for one generic.

Taking the first hit is wrong often enough to matter: a search for
"ciprofloxacin hydrochloride" returns CILOXAN (ophthalmic drops) before any
systemic tablet, and "metformin hydrochloride" returns combination products
before plain metformin. Both produce a corpus that answers the wrong question,
and the Phase 2 gold set is built against this corpus — so the fix belongs
here, at ingest, not in retrieval.
"""

from __future__ import annotations

from rxclarify.ingest.parse import SECTION_SPEC

# Systemic routes answer the questions this corpus exists for; local routes
# (eye, skin, ear) carry almost no interaction content.
SYSTEMIC_ROUTES = {"ORAL", "INTRAVENOUS", "INTRAMUSCULAR", "SUBCUTANEOUS", "TRANSDERMAL"}
LOCAL_ROUTES = {"OPHTHALMIC", "TOPICAL", "OTIC", "NASAL", "RECTAL", "VAGINAL", "INHALATION"}

CANDIDATE_LIMIT = 5


def _strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str) and value.strip():
        return [value]
    return []


def _section_chars(payload: dict) -> int:
    total = 0
    for _, aliases in SECTION_SPEC:
        for alias in aliases:
            texts = _strings(payload.get(alias))
            if texts:
                total += sum(len(t) for t in texts)
                break
    return total


def is_combination(payload: dict) -> bool:
    openfda = payload.get("openfda") or {}
    generics = {g.strip().lower() for g in _strings(openfda.get("generic_name"))}
    if len(generics) > 1:
        return True
    # A single generic_name can still name a combination: "SITAGLIPTIN AND
    # METFORMIN HYDROCHLORIDE".
    return any(" and " in g for g in generics)


def score_candidate(payload: dict) -> int:
    """Higher is better. Single-ingredient, systemic, interaction-rich wins."""
    openfda = payload.get("openfda") or {}
    score = 0

    if not is_combination(payload):
        score += 100

    routes = {r.strip().upper() for r in _strings(openfda.get("route"))}
    if routes & SYSTEMIC_ROUTES:
        score += 50
    elif routes & LOCAL_ROUTES:
        score -= 60

    if _strings(payload.get("drug_interactions")):
        score += 40

    # Richer labels give retrieval more to work with, capped so a single
    # enormous label cannot outrank a clean single-ingredient match.
    score += min(_section_chars(payload) // 2000, 30)

    return score


def select_best(candidates: list[dict]) -> dict | None:
    """Highest-scoring candidate, ties broken by original openFDA ordering."""
    usable = [c for c in candidates if isinstance(c, dict)]
    if not usable:
        return None
    return max(usable, key=score_candidate)
