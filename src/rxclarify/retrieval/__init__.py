"""Retrieval strategies. Phase 1 ships dense only; Phase 2 adds sparse + hybrid + rerank."""

from rxclarify.retrieval.base import RetrievedChunk, Retriever
from rxclarify.retrieval.dense import DenseRetriever
from rxclarify.retrieval.langchain_retriever import PgVectorRetriever, to_chunk, to_document

__all__ = [
    "RetrievedChunk",
    "Retriever",
    "DenseRetriever",
    "PgVectorRetriever",
    "to_chunk",
    "to_document",
]
