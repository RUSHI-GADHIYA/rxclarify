"""Provider-agnostic completion interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, *, system: str, user: str, max_tokens: int) -> Completion: ...
