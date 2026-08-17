"""Run the LCEL chain and validate what came back."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.retrievers import BaseRetriever

from rxclarify.config import get_settings
from rxclarify.generate.chain import build_chain
from rxclarify.generate.prompt import PROMPT_VERSION, REFUSAL_TOKEN
from rxclarify.retrieval.base import RetrievedChunk
from rxclarify.retrieval.langchain_retriever import to_chunk

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


def _model_name(chat_model: BaseChatModel) -> str:
    for attr in ("model_name", "model_id", "model"):
        value = getattr(chat_model, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(chat_model).__name__


def answer_question(
    question: str,
    *,
    retriever: BaseRetriever,
    chat_model: BaseChatModel,
    top_k: int | None = None,
) -> Answer:
    settings = get_settings()
    if top_k is not None:
        # BaseRetriever is a pydantic model; k is a declared field on ours.
        retriever = retriever.model_copy(update={"k": top_k})
    elif getattr(retriever, "k", None) is None:
        retriever = retriever.model_copy(update={"k": settings.top_k})

    chain = build_chain(retriever, chat_model)

    started = time.perf_counter()
    result = chain.invoke(question)
    elapsed_ms = (time.perf_counter() - started) * 1000

    chunks = [to_chunk(d) for d in result["docs"]]
    message: AIMessage = result["message"]
    text = (message.text or "").strip()

    usage = message.usage_metadata or {}
    valid_markers = {c.marker for c in chunks}
    cited, invalid = parse_citations(text, valid_markers)

    return Answer(
        question=question,
        text=text,
        chunks=chunks,
        cited_markers=cited,
        invalid_markers=invalid,
        refused=is_refusal(text),
        model=_model_name(chat_model),
        input_tokens=usage.get("input_tokens"),
        output_tokens=usage.get("output_tokens"),
        latency_ms=elapsed_ms,
    )
