"""LangChain `BaseRetriever` over the pgvector dense index.

This is the seam between RxClarify's own retrieval code and LCEL. Retrieval
logic stays in `dense.py`; this class only adapts `RetrievedChunk` to
`Document` so the retriever can be piped into a chain.

Citation markers (C1..Cn) are assigned here, by rank, because ordering
authority has to live in exactly one place — the prompt, the answer, and the
citation validator must all agree on what "C3" refers to.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from rxclarify.retrieval.base import RetrievedChunk
from rxclarify.retrieval.dense import DenseRetriever


class PgVectorRetriever(BaseRetriever):
    """Adapts DenseRetriever to the LangChain retriever interface."""

    # BaseRetriever is a pydantic model; a psycopg connection is not a pydantic
    # type, hence arbitrary_types_allowed.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    conn: Any
    k: int = 6

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        chunks = DenseRetriever(self.conn).retrieve(query, top_k=self.k)
        return [to_document(chunk) for chunk in chunks]


class ProtocolRetriever(BaseRetriever):
    """Adapts *any* rxclarify `Retriever` to LangChain's interface.

    `PgVectorRetriever` hardcodes dense retrieval. This wraps whatever you give
    it — sparse, hybrid, reranked — so the LCEL chain, the eval harness, and the
    UI can all run alternative retrievers without a variant adapter for each.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    retriever: Any
    k: int = 6

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return [to_document(c) for c in self.retriever.retrieve(query, top_k=self.k)]


def to_document(chunk: RetrievedChunk) -> Document:
    return Document(
        id=str(chunk.chunk_id),
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "label_id": chunk.label_id,
            "drug": chunk.drug,
            "section": chunk.section,
            "score": chunk.score,
            "marker": chunk.marker,
        },
    )


def to_chunk(document: Document) -> RetrievedChunk:
    meta = document.metadata
    return RetrievedChunk(
        chunk_id=int(meta["chunk_id"]),
        label_id=int(meta["label_id"]),
        drug=meta["drug"],
        section=meta["section"],
        text=document.page_content,
        score=float(meta["score"]),
        marker=int(meta["marker"]),
    )
