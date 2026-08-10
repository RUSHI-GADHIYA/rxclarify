"""Orchestration for the ingest pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from rxclarify.db import connect
from rxclarify.ingest.chunk import chunk_label
from rxclarify.ingest.embed import embed_documents
from rxclarify.ingest.fetch_labels import fetch_all, load_drug_list
from rxclarify.ingest.load import prune_superseded, replace_chunks, upsert_label
from rxclarify.ingest.parse import parse_label
from rxclarify.ingest.select import select_best


@dataclass
class IngestReport:
    labels_ingested: int = 0
    chunks_written: int = 0
    labels_superseded: int = 0
    from_cache: int = 0
    fetched_live: int = 0
    no_label_found: list[str] = field(default_factory=list)
    no_usable_sections: list[str] = field(default_factory=list)


def run_ingest(
    *,
    refresh: bool = False,
    limit: int | None = None,
    progress=None,
) -> IngestReport:
    """Fetch -> parse -> chunk -> embed -> load for every drug in drug_list.yml.

    `progress` is an optional callable(message: str) for CLI output.
    """

    def say(message: str) -> None:
        if progress:
            progress(message)

    generics = load_drug_list()
    if limit is not None:
        generics = generics[:limit]

    say(f"Fetching {len(generics)} labels from openFDA (cached responses reused)...")
    results, missing = fetch_all(generics, refresh=refresh)

    report = IngestReport(
        no_label_found=missing,
        from_cache=sum(1 for r in results if r.from_cache),
        fetched_live=sum(1 for r in results if not r.from_cache),
    )

    with connect() as conn:
        for result in results:
            payload = select_best(result.candidates)
            label = parse_label(payload, result.generic) if payload else None
            if label is None:
                report.no_usable_sections.append(result.generic)
                say(f"  skip  {result.generic}: no usable sections")
                continue

            chunks = chunk_label(label)
            if not chunks:
                report.no_usable_sections.append(result.generic)
                say(f"  skip  {result.generic}: produced no chunks")
                continue

            embeddings = embed_documents(c.embed_text for c in chunks)

            label_id = upsert_label(conn, label, payload)
            superseded = prune_superseded(conn, result.generic, label_id)
            written = replace_chunks(conn, label_id, chunks, embeddings)
            # Commit per label so a failure midway leaves earlier work durable.
            conn.commit()

            report.labels_ingested += 1
            report.chunks_written += written
            report.labels_superseded += superseded
            note = f" (replaced {superseded} stale label(s))" if superseded else ""
            say(f"  ok    {label.display_name}: {written} chunks{note}")

    return report
