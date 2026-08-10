"""Turn a raw openFDA SPL payload into a normalized label + its useful sections."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sections a pharmacy technician or pharmacist actually cross-checks, in the
# order we want them stored. Each entry is (canonical_name, source_field_aliases).
# Everything else in the SPL is dropped: clinical_pharmacology alone would
# roughly double the corpus with text nobody queries.
SECTION_SPEC: list[tuple[str, tuple[str, ...]]] = [
    ("boxed_warning", ("boxed_warning",)),
    ("indications_and_usage", ("indications_and_usage",)),
    ("dosage_and_administration", ("dosage_and_administration",)),
    ("contraindications", ("contraindications",)),
    ("warnings_and_cautions", ("warnings_and_precautions", "warnings_and_cautions", "warnings")),
    ("drug_interactions", ("drug_interactions",)),
    ("use_in_specific_populations", ("use_in_specific_populations",)),
]

KEPT_SECTIONS = tuple(name for name, _ in SECTION_SPEC)

# SPL text is one hard-wrapped blob. Collapse runs of horizontal whitespace
# (including U+00A0, which SPL is littered with) but keep paragraph breaks,
# which the chunker splits on.
_INLINE_WS = re.compile("[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class Section:
    name: str
    text: str


@dataclass(frozen=True)
class LabelDoc:
    set_id: str
    spl_id: str | None
    query_generic: str
    brand_name: str | None
    generic_name: str | None
    manufacturer: str | None
    effective_time: str | None
    sections: list[Section] = field(default_factory=list)

    @property
    def display_name(self) -> str:
        """Human-facing label name, e.g. 'Coumadin (warfarin sodium)'."""
        brand = (self.brand_name or "").strip()
        generic = (self.generic_name or self.query_generic).strip()
        if brand and generic and brand.lower() != generic.lower():
            return f"{brand} ({generic})"
        return brand or generic


def clean_text(value: str) -> str:
    text = value.replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_WS.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    return text.strip()


def _first(values: object) -> str | None:
    """openFDA returns nearly every scalar as a single-element list."""
    if isinstance(values, list):
        for item in values:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(values, str) and values.strip():
        return values.strip()
    return None


def parse_label(payload: dict, query_generic: str) -> LabelDoc | None:
    """Normalize one openFDA result. Returns None if it has no usable sections."""
    openfda = payload.get("openfda") or {}

    set_id = _first(payload.get("set_id")) or _first(openfda.get("spl_set_id"))
    if not set_id:
        return None

    sections: list[Section] = []
    for canonical, aliases in SECTION_SPEC:
        for alias in aliases:
            raw = payload.get(alias)
            if not raw:
                continue
            joined = clean_text("\n\n".join(v for v in raw if isinstance(v, str)))
            if joined:
                sections.append(Section(canonical, joined))
            # First matching alias wins — warnings_and_precautions beats warnings.
            break

    if not sections:
        return None

    return LabelDoc(
        set_id=set_id,
        spl_id=_first(payload.get("id")),
        query_generic=query_generic,
        brand_name=_first(openfda.get("brand_name")),
        generic_name=_first(openfda.get("generic_name")) or query_generic,
        manufacturer=_first(openfda.get("manufacturer_name")),
        effective_time=_first(payload.get("effective_time")),
        sections=sections,
    )
