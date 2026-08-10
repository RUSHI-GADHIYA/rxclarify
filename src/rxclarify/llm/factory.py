"""Select a provider from configuration (or an explicit override)."""

from __future__ import annotations

from rxclarify.config import get_settings
from rxclarify.llm.base import LLMProvider


def get_provider(name: str | None = None) -> LLMProvider:
    provider = (name or get_settings().llm_provider).lower()

    if provider == "bedrock":
        from rxclarify.llm.bedrock import BedrockProvider

        return BedrockProvider()
    if provider == "ollama":
        from rxclarify.llm.ollama import OllamaProvider

        return OllamaProvider()

    raise ValueError(f"unknown LLM provider {provider!r}; expected 'bedrock' or 'ollama'")
