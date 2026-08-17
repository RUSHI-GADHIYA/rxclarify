"""Sparse (lexical) retrieval over the Postgres full-text index.

The other half of hybrid search. Dense embeddings match meaning but blur exact
tokens — drug names, dose strings like "6 mg/kg", allele names like HLA-B*1502.
Lexical search matches those exactly and is hopeless at paraphrase.

No new infrastructure: the `tsv` generated column and its GIN index have been in
`db/schema.sql` since Phase 1, precisely so this could be added without a
migration.

**Query strictness is the whole design problem here.** `websearch_to_tsquery`
joins terms with AND, so a question like "How is allopurinol titrated in a
patient with normal kidney function?" becomes
`allopurinol & titrat & patient & normal & kidney & function` — six terms that
must all appear in one ~1,200-character chunk. Measured, that returned 0.6
results per query and recall@6 of 0.267.

Rewriting AND to OR fixed sparse in isolation (recall@6 0.267 -> 0.622) and
*hurt the hybrid* (0.978 -> 0.933), because every query then returned six weak
single-term matches that diluted the dense signal during fusion. Hence three
modes, and the measured default below.
"""

from __future__ import annotations

from typing import Literal

import psycopg

from rxclarify.retrieval.base import RetrievedChunk

Mode = Literal["strict", "loose", "adaptive"]

# Below this many strict hits, `adaptive` retries with the loose query.
MIN_STRICT_HITS = 3

# Only ' & ' with surrounding spaces is rewritten, so phrase operators (`<->`)
# produced by quoted input survive untouched.
_STRICT = "websearch_to_tsquery('english', %(query)s)"
_LOOSE = f"replace({_STRICT}::text, ' & ', ' | ')::tsquery"


def _sql(tsquery_expr: str) -> str:
    return f"""
    WITH q AS (SELECT {tsquery_expr} AS tsq)
    SELECT
        c.id            AS chunk_id,
        c.label_id      AS label_id,
        c.section       AS section,
        c.text          AS text,
        ts_rank_cd(c.tsv, q.tsq) AS score,
        COALESCE(NULLIF(l.brand_name, ''), l.generic_name, l.query_generic) AS drug
    FROM chunks c
    JOIN labels l ON l.id = c.label_id
    CROSS JOIN q
    WHERE c.tsv @@ q.tsq
    ORDER BY score DESC, c.id
    LIMIT %(limit)s
    """


class SparseRetriever:
    """Postgres full-text retrieval.

    `adaptive` runs the strict AND query first and falls back to OR only when
    that returns too little — high-precision matches when they exist, some
    matches rather than none when they do not.
    """

    def __init__(self, conn: psycopg.Connection, *, mode: Mode = "adaptive") -> None:
        self._conn = conn
        self.mode = mode

    def _query(self, expr: str, query: str, limit: int) -> list[dict]:
        with self._conn.cursor() as cur:
            cur.execute(_sql(expr), {"query": query, "limit": limit})
            return cur.fetchall()

    def retrieve(self, query: str, *, top_k: int) -> list[RetrievedChunk]:
        if self.mode == "loose":
            rows = self._query(_LOOSE, query, top_k)
        elif self.mode == "strict":
            rows = self._query(_STRICT, query, top_k)
        else:
            rows = self._query(_STRICT, query, top_k)
            if len(rows) < min(MIN_STRICT_HITS, top_k):
                rows = self._query(_LOOSE, query, top_k)

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
