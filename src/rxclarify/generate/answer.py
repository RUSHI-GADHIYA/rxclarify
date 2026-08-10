"""retrieve -> prompt -> generate -> validate citations."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rxclarify.config import get_settings
from rxclarify.generate.prompt import (
    PROMPT_VERSION,
    REFUSAL_TOKEN,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from rxclarify.llm.base import LLMProvider
from rxclarify.retrieval.base import RetrievedChunk, Retriever

_CITATION = re.compile(r"\[\s*C(\d+)\s*\]", re.IGNORECASE)


@dataclass
class Answer:
    question: str
    text: str
    chunks: list[RetrievedChunk]
    cited_markers: list[int] = field(default_factory=list)
    # Markers the model produced that were never in the context. Any value here
    # is a hard hallucination signal — countable without a judge model, which
    # makes it the cheapest guardrail check in the pipeline.
    invalid_markers: list[int] = field(default_factory=list)
    refused: bool = False
    model: str = ""
    prompt_version: str = PROMPT_VERSION
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None

    @property
    def uncited(self) -> bool:
        """True when a non-refusal answer cites nothing at all."""
        return not self.refused and not self.cited_markers

    def cited_chunks(self) -> list[RetrievedChunk]:
        by_marker = {c.marker: c for c in self.chunks}
        return [by_marker[m] for m in self.cited_markers if m in by_marker]


def parse_citations(text: str, valid_markers: set[int]) -> tuple[list[int], list[int]]:
    """Return (cited markers present in context, cited markers that were not).

    Both lists are deduplicated and returned in first-appearance order.
    """
    seen: list[int] = []
    for match in _CITATION.finditer(text):
        marker = int(match.group(1))
        if marker not in seen:
            seen.append(marker)
    valid = [m for m in seen if m in valid_markers]
    invalid = [m for m in seen if m not in valid_markers]
    return valid, invalid


def is_refusal(text: str) -> bool:
    return text.strip().upper().startswith(REFUSAL_TOKEN)


def answer_question(
    question: str,
    *,
    retriever: Retriever,
    provider: LLMProvider,
    top_k: int | None = None,
    max_tokens: int | None = None,
) -> Answer:
    settings = get_settings()
    top_k = top_k if top_k is not None else settings.top_k
    max_tokens = max_tokens if max_tokens is not None else settings.max_tokens

    chunks = retriever.retrieve(question, top_k=top_k)
    completion = provider.complete(
        system=SYSTEM_PROMPT,
        user=build_user_prompt(question, chunks),
        max_tokens=max_tokens,
    )

    valid_markers = {c.marker for c in chunks}
    cited, invalid = parse_citations(completion.text, valid_markers)

    return Answer(
        question=question,
        text=completion.text,
        chunks=chunks,
        cited_markers=cited,
        invalid_markers=invalid,
        refused=is_refusal(completion.text),
        model=completion.model,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        latency_ms=completion.latency_ms,
    )
