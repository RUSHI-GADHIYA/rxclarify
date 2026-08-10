"""Retriever interface.

Everything downstream (prompt building, citation validation, the Phase 2 eval
harness) depends only on this shape, so hybrid retrieval and reranking can be
dropped in later without touching the generation path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    label_id: int
    drug: str
    section: str
    text: str
    score: float
    # Position in the prompt, 1-based. This is the number the model cites as
    # [C1], [C2], ... — deliberately not the database id, so the model never
    # sees or invents primary keys.
    marker: int = 0

    @property
    def citation(self) -> str:
        return f"C{self.marker}"


@runtime_checkable
class Retriever(Protocol):
    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]: ...
