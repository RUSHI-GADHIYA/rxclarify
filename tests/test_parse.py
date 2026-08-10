from rxclarify.ingest.parse import clean_text, parse_label


def test_clean_text_collapses_whitespace_but_keeps_paragraphs():
    raw = "Line  one here.\n\n\n\nSecond   paragraph.\r\nSame paragraph."
    assert clean_text(raw) == "Line one here.\n\nSecond paragraph.\nSame paragraph."


def test_parse_label_extracts_kept_sections_and_metadata():
    payload = {
        "set_id": ["abc-123"],
        "id": ["spl-9"],
        "effective_time": ["20240101"],
        "drug_interactions": ["Avoid strong CYP3A4 inhibitors."],
        "boxed_warning": ["WARNING: bleeding risk."],
        # Not in SECTION_SPEC -> must be dropped.
        "clinical_pharmacology": ["Half-life is 40 hours."],
        "openfda": {
            "brand_name": ["Coumadin"],
            "generic_name": ["WARFARIN SODIUM"],
            "manufacturer_name": ["Acme"],
        },
    }

    label = parse_label(payload, "warfarin sodium")

    assert label is not None
    assert label.set_id == "abc-123"
    assert label.spl_id == "spl-9"
    assert label.brand_name == "Coumadin"
    assert label.display_name == "Coumadin (WARFARIN SODIUM)"

    names = [s.name for s in label.sections]
    assert names == ["boxed_warning", "drug_interactions"]  # SECTION_SPEC order, not payload order
    assert "clinical_pharmacology" not in names


def test_warnings_alias_precedence():
    """warnings_and_precautions wins over the legacy `warnings` field."""
    payload = {
        "set_id": ["s"],
        "warnings_and_precautions": ["Modern section."],
        "warnings": ["Legacy section."],
        "openfda": {"generic_name": ["drug"]},
    }
    label = parse_label(payload, "drug")
    assert label is not None
    assert [s.text for s in label.sections] == ["Modern section."]


def test_parse_label_returns_none_without_usable_sections():
    payload = {"set_id": ["s"], "clinical_pharmacology": ["irrelevant"], "openfda": {}}
    assert parse_label(payload, "drug") is None


def test_parse_label_returns_none_without_set_id():
    payload = {"drug_interactions": ["something"], "openfda": {}}
    assert parse_label(payload, "drug") is None


def test_display_name_falls_back_when_brand_matches_generic():
    payload = {
        "set_id": ["s"],
        "drug_interactions": ["x"],
        "openfda": {"brand_name": ["Metformin"], "generic_name": ["metformin"]},
    }
    label = parse_label(payload, "metformin")
    assert label is not None
    assert label.display_name == "Metformin"
