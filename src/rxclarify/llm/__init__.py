"""LLM providers behind one protocol, so generation and eval can swap freely."""

from rxclarify.llm.base import Completion, LLMProvider
from rxclarify.llm.factory import get_provider

__all__ = ["Completion", "LLMProvider", "get_provider"]
