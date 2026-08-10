"""Idempotent upsert of parsed labels + embedded chunks into Postgres."""

from __future__ import annotations

import json

import psycopg

from rxclarify.db import to_pgvector
from rxclarify.ingest.chunk import Chunk
from rxclarify.ingest.parse import LabelDoc


def upsert_label(conn: psycopg.Connection, label: LabelDoc, raw: dict) -> int:
    """Insert or refresh a label row, returning its id.

    Keyed on set_id (stable across SPL revisions), so re-ingesting the same drug
    updates in place instead of duplicating.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO labels (set_id, spl_id, query_generic, brand_name,
                                generic_name, manufacturer, effective_time, raw)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (set_id) DO UPDATE SET
                spl_id         = EXCLUDED.spl_id,
                query_generic  = EXCLUDED.query_generic,
                brand_name     = EXCLUDED.brand_name,
                generic_name   = EXCLUDED.generic_name,
                manufacturer   = EXCLUDED.manufacturer,
                effective_time = EXCLUDED.effective_time,
                raw            = EXCLUDED.raw,
                fetched_at     = now()
            RETURNING id
            """,
            (
                label.set_id,
                label.spl_id,
                label.query_generic,
                label.brand_name,
                label.generic_name,
                label.manufacturer,
                label.effective_time,
                json.dumps(raw),
            ),
        )
        row = cur.fetchone()
    if row is None:  # pragma: no cover - RETURNING always yields a row here
        raise RuntimeError(f"failed to upsert label {label.set_id}")
    return int(row["id"])


def prune_superseded(conn: psycopg.Connection, query_generic: str, keep_label_id: int) -> int:
    """Drop other labels ingested for the same generic, returning how many.

    Labels are keyed on set_id, but the *choice* of which SPL represents a drug
    can change between runs — tuning candidate selection (ingest/select.py) is
    exactly such a change. Without this, an older pick survives the re-ingest
    and its chunks stay retrievable: re-running ingest after fixing the
    ciprofloxacin selection would still leave the ophthalmic label in the index.
    One label per generic is the invariant; enforce it on every write.
    """
    with conn.cursor() as cur:
        cur.execute(
            "DELETE FROM labels WHERE query_generic = %s AND id <> %s",
            (query_generic, keep_label_id),
        )
        return cur.rowcount


def replace_chunks(
    conn: psycopg.Connection,
    label_id: int,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> int:
    """Swap in a fresh chunk set for one label.

    Delete-then-insert rather than upsert: re-chunking with different parameters
    changes how many chunks a section produces, and stale trailing ordinals from
    a previous run would otherwise survive and pollute retrieval.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE label_id = %s", (label_id,))
        if not chunks:
            return 0
        cur.executemany(
            """
            INSERT INTO chunks (label_id, section, ordinal, text, char_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s::vector)
            """,
            [
                (
                    label_id,
                    chunk.section,
                    chunk.ordinal,
                    chunk.text,
                    chunk.char_count,
                    to_pgvector(vector),
                )
                for chunk, vector in zip(chunks, embeddings, strict=True)
            ],
        )
    return len(chunks)
