"""Retrieval strategies. Phase 1 ships dense only; Phase 2 adds sparse + hybrid + rerank."""

from rxclarify.retrieval.base import RetrievedChunk, Retriever
from rxclarify.retrieval.dense import DenseRetriever

__all__ = ["RetrievedChunk", "Retriever", "DenseRetriever"]
