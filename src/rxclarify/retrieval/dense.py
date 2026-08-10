"""Dense retrieval: cosine ANN over the pgvector HNSW index.

This is the Phase 1 baseline. Phase 2 measures hybrid (dense + BM25-style
full-text) and cross-encoder reranking against these numbers.
"""

from __future__ import annotations

import psycopg

from rxclarify.db import to_pgvector
from rxclarify.ingest.embed import embed_query
from rxclarify.retrieval.base import RetrievedChunk

# `<=>` is pgvector's cosine distance (0 = identical). Reported as similarity so
# higher is better, matching how every other scorer in this project reads.
_SQL = """
SELECT
    c.id            AS chunk_id,
    c.label_id      AS label_id,
    c.section       AS section,
    c.text          AS text,
    1 - (c.embedding <=> %(query)s::vector) AS score,
    COALESCE(NULLIF(l.brand_name, ''), l.generic_name, l.query_generic) AS drug
FROM chunks c
JOIN labels l ON l.id = c.label_id
WHERE c.embedding IS NOT NULL
ORDER BY c.embedding <=> %(query)s::vector
LIMIT %(limit)s
"""


class DenseRetriever:
    def __init__(self, conn: psycopg.Connection) -> None:
        self._conn = conn

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        vector = to_pgvector(embed_query(query))
        with self._conn.cursor() as cur:
            cur.execute(_SQL, {"query": vector, "limit": top_k})
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                label_id=row["label_id"],
                drug=row["drug"],
                section=row["section"],
                text=row["text"],
                score=float(row["score"]),
                marker=index,
            )
            for index, row in enumerate(rows, start=1)
        ]
