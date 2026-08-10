"""Phase 1 baseline chunker: fixed-size splitting within section boundaries.

Deliberately naive. Phase 2 measures semantic chunking and parent-document
retrieval against this baseline, so the numbers only mean something if the
starting point is the obvious thing everyone builds first.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rxclarify.ingest.parse import LabelDoc

TARGET_CHARS = 1200
OVERLAP_CHARS = 150
# Below this, a fragment is noise (a stray heading, a table caption) and is
# folded into its neighbour rather than embedded on its own.
MIN_CHARS = 120

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

SECTION_LABELS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
}


@dataclass(frozen=True)
class Chunk:
    section: str
    ordinal: int
    text: str
    embed_text: str

    @property
    def char_count(self) -> int:
        return len(self.text)


def _hard_wrap(sentence: str) -> list[str]:
    """Last-resort split for a single 'sentence' longer than TARGET_CHARS.

    SPL tables flatten into enormous run-on strings with no sentence
    punctuation. Without this cap such a passage would exceed bge-small's
    512-token window and be silently truncated at embed time — losing the tail
    of the passage from the index while it still displays in full.
    """
    return [sentence[i : i + TARGET_CHARS] for i in range(0, len(sentence), TARGET_CHARS)]


def _tail_sentences(sentences: list[str], budget: int) -> list[str]:
    """The trailing whole sentences that fit in `budget` characters."""
    carry: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        extra = len(sentence) + (1 if carry else 0)
        if total + extra > budget:
            break
        carry.insert(0, sentence)
        total += extra
    return carry


def _split_oversized(block: str) -> list[str]:
    """Split a single over-long paragraph on sentence boundaries.

    Overlap is carried as whole trailing sentences rather than a character
    slice, so no chunk begins mid-word.
    """
    sentences: list[str] = []
    for raw in _SENTENCE_END.split(block):
        if not raw:
            continue
        sentences.extend(_hard_wrap(raw) if len(raw) > TARGET_CHARS else [raw])

    pieces: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        addition = len(sentence) + (1 if current else 0)
        if current and current_len + addition > TARGET_CHARS:
            pieces.append(" ".join(current))
            current = _tail_sentences(current, OVERLAP_CHARS)
            current_len = sum(len(s) for s in current) + max(len(current) - 1, 0)
            addition = len(sentence) + (1 if current else 0)
        current.append(sentence)
        current_len += addition

    if current:
        pieces.append(" ".join(current))
    return pieces


def split_section(text: str) -> list[str]:
    """Pack paragraphs up to TARGET_CHARS, splitting any paragraph that overflows."""
    blocks = [b.strip() for b in _PARAGRAPH.split(text) if b.strip()]

    packed: list[str] = []
    current = ""
    for block in blocks:
        if len(block) > TARGET_CHARS:
            if current:
                packed.append(current)
                current = ""
            packed.extend(_split_oversized(block))
            continue

        candidate = f"{current}\n\n{block}" if current else block
        if len(candidate) <= TARGET_CHARS:
            current = candidate
        else:
            packed.append(current)
            current = block
    if current:
        packed.append(current)

    # Fold runt fragments into the previous chunk so we never embed a bare heading.
    merged: list[str] = []
    for piece in packed:
        if merged and len(piece) < MIN_CHARS:
            merged[-1] = f"{merged[-1]}\n\n{piece}"
        else:
            merged.append(piece)
    return merged


def chunk_label(label: LabelDoc) -> list[Chunk]:
    """Chunk every section of a label.

    `text` is what the model reads. `embed_text` prepends the drug name and
    section so the vector carries the subject even when the paragraph says
    only "this drug" — a real failure mode when a query names the drug.
    """
    chunks: list[Chunk] = []
    for section in label.sections:
        label_name = SECTION_LABELS.get(section.name, section.name)
        prefix = f"{label.display_name} — {label_name}: "
        for ordinal, piece in enumerate(split_section(section.text)):
            chunks.append(
                Chunk(
                    section=section.name,
                    ordinal=ordinal,
                    text=piece,
                    embed_text=prefix + piece,
                )
            )
    return chunks
