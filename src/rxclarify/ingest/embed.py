"""Embeddings via fastembed (ONNX, CPU) — no PyTorch install required.

bge-small-en-v1.5 is an asymmetric model: passages are embedded as-is, queries
are embedded with an instruction prefix. fastembed's `query_embed` applies that
prefix, so document and query paths must use the two different methods below.
Using `embed()` for queries silently costs several points of recall.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import TYPE_CHECKING

from rxclarify.config import EMBEDDING_DIM, EMBEDDING_MODEL

if TYPE_CHECKING:  # pragma: no cover
    from fastembed import TextEmbedding


@lru_cache(maxsize=1)
def get_model() -> TextEmbedding:
    # Imported lazily: loading fastembed pulls in onnxruntime and, on first run,
    # downloads the model. Nothing that merely imports this module should pay that.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=EMBEDDING_MODEL)


def embed_documents(texts: Iterable[str], *, batch_size: int = 64) -> list[list[float]]:
    model = get_model()
    vectors = [v.tolist() for v in model.embed(list(texts), batch_size=batch_size)]
    _assert_dim(vectors)
    return vectors


def embed_query(text: str) -> list[float]:
    model = get_model()
    vectors = [v.tolist() for v in model.query_embed([text])]
    _assert_dim(vectors)
    return vectors[0]


def _assert_dim(vectors: list[list[float]]) -> None:
    """Fail loudly on a dimension mismatch rather than at the pgvector INSERT."""
    if vectors and len(vectors[0]) != EMBEDDING_DIM:
        raise RuntimeError(
            f"{EMBEDDING_MODEL} returned {len(vectors[0])}-dim vectors, "
            f"but the schema declares vector({EMBEDDING_DIM}). "
            "Update EMBEDDING_DIM in config.py and db/schema.sql together."
        )
