"""Resolve the configured provider to a LangChain chat model.

Both providers return a `BaseChatModel`, which is what lets the LCEL chain in
generate/chain.py stay provider-agnostic: swapping OpenAI for Bedrock changes
one env var and nothing else in the pipeline.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from rxclarify.config import get_settings

PROVIDERS = ("openai", "bedrock")


def get_chat_model(name: str | None = None, *, max_tokens: int | None = None) -> BaseChatModel:
    provider = (name or get_settings().llm_provider).lower()

    if provider == "openai":
        from rxclarify.llm.openai import build_openai_chat

        return build_openai_chat(max_tokens=max_tokens)
    if provider == "bedrock":
        from rxclarify.llm.bedrock import build_bedrock_chat

        return build_bedrock_chat(max_tokens=max_tokens)

    raise ValueError(f"unknown LLM provider {provider!r}; expected one of {PROVIDERS}")


def provider_name(name: str | None = None) -> str:
    return (name or get_settings().llm_provider).lower()
