"""Postgres connection helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from rxclarify.config import get_settings


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Open a connection with dict rows. Commits on clean exit, rolls back on error."""
    settings = get_settings()
    with psycopg.connect(settings.database_url, row_factory=dict_row) as conn:
        yield conn


def to_pgvector(values: Sequence[float]) -> str:
    """Render a float sequence in pgvector's text input format.

    Sent as text and cast with `::vector` in SQL, which avoids depending on the
    separate `pgvector` adapter package.
    """
    return "[" + ",".join(repr(float(v)) for v in values) + "]"


def table_counts(conn: psycopg.Connection) -> dict[str, int]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                (SELECT count(*) FROM labels)                          AS labels,
                (SELECT count(*) FROM chunks)                          AS chunks,
                (SELECT count(*) FROM chunks WHERE embedding IS NULL)  AS chunks_missing_embedding
            """
        )
        row = cur.fetchone()
    return dict(row) if row else {}
