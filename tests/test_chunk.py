from rxclarify.ingest.chunk import MIN_CHARS, TARGET_CHARS, chunk_label, split_section
from rxclarify.ingest.parse import LabelDoc, Section


def _label(sections: list[Section]) -> LabelDoc:
    return LabelDoc(
        set_id="s",
        spl_id=None,
        query_generic="warfarin sodium",
        brand_name="Coumadin",
        generic_name="warfarin sodium",
        manufacturer=None,
        effective_time=None,
        sections=sections,
    )


def test_short_section_is_one_chunk():
    assert split_section("A short interactions note.") == ["A short interactions note."]


def test_paragraphs_pack_up_to_target():
    para = "x" * 500
    pieces = split_section("\n\n".join([para] * 5))
    assert all(len(p) <= TARGET_CHARS for p in pieces)
    # 5x500 chars cannot fit in one 1200-char chunk.
    assert len(pieces) > 1


def test_oversized_paragraph_is_split_on_sentences():
    sentence = "Concomitant use raises exposure. "
    pieces = split_section(sentence * 100)
    assert len(pieces) > 1
    # Sentence-boundary splitting means no piece starts mid-word.
    assert all(p.startswith("Concomitant") for p in pieces)


def test_runt_fragment_is_folded_into_previous_chunk():
    body = "y" * (TARGET_CHARS - 100)
    runt = "z" * (MIN_CHARS - 10)
    pieces = split_section(f"{body}\n\n{runt}")
    assert len(pieces) == 1
    assert pieces[0].endswith(runt)


def test_no_chunk_exceeds_the_target():
    """Chunks must stay inside bge-small's 512-token window with room to spare."""
    pieces = split_section("Sentence number one is here. " * 200)
    assert all(len(p) <= TARGET_CHARS for p in pieces)


def test_unpunctuated_run_on_text_is_hard_wrapped():
    """SPL tables flatten to huge strings with no sentence breaks."""
    pieces = split_section("nopunctuation" * 500)
    assert len(pieces) > 1
    assert all(len(p) <= TARGET_CHARS for p in pieces)


def test_chunk_label_prefixes_embed_text_with_drug_and_section():
    label = _label([Section("drug_interactions", "Avoid azoles.")])
    chunks = chunk_label(label)

    assert len(chunks) == 1
    chunk = chunks[0]
    # The model reads `text`; the vector is built from `embed_text`, which
    # carries the drug name even though the passage never says "warfarin".
    assert chunk.text == "Avoid azoles."
    assert chunk.embed_text == "Coumadin (warfarin sodium) — Drug Interactions: Avoid azoles."
    assert chunk.section == "drug_interactions"
    assert chunk.ordinal == 0
    assert chunk.char_count == len("Avoid azoles.")


def test_ordinals_restart_per_section():
    long_text = "w" * (TARGET_CHARS * 2)
    label = _label(
        [
            Section("boxed_warning", long_text),
            Section("drug_interactions", long_text),
        ]
    )
    chunks = chunk_label(label)

    boxed = [c.ordinal for c in chunks if c.section == "boxed_warning"]
    interactions = [c.ordinal for c in chunks if c.section == "drug_interactions"]
    assert boxed == list(range(len(boxed)))
    assert interactions == list(range(len(interactions)))
