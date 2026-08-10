"""`python -m rxclarify.ingest` — equivalent to `rxclarify ingest`."""

from __future__ import annotations

from rxclarify.ingest.pipeline import run_ingest


def main() -> None:
    report = run_ingest(progress=print)
    print(
        f"\nIngested {report.labels_ingested} labels / {report.chunks_written} chunks "
        f"({report.from_cache} from cache, {report.fetched_live} fetched live)."
    )
    if report.no_label_found:
        print(f"No openFDA label for: {', '.join(report.no_label_found)}")
    if report.no_usable_sections:
        print(f"No usable sections for: {', '.join(report.no_usable_sections)}")


if __name__ == "__main__":
    main()
