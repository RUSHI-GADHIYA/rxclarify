"""RxClarify command line."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rxclarify.config import get_settings
from rxclarify.db import connect, table_counts
from rxclarify.generate.answer import answer_question
from rxclarify.llm.factory import PROVIDERS, get_chat_model, provider_name
from rxclarify.retrieval.langchain_retriever import PgVectorRetriever

app = typer.Typer(
    add_completion=False,
    help="Grounded RAG over public FDA drug labels.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def ingest(
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-fetch from openFDA, ignoring the disk cache.")
    ] = False,
    limit: Annotated[
        int | None, typer.Option("--limit", help="Only ingest the first N drugs (smoke test).")
    ] = None,
) -> None:
    """Fetch, chunk, embed, and load drug labels into Postgres."""
    from rxclarify.ingest.pipeline import run_ingest

    report = run_ingest(refresh=refresh, limit=limit, progress=console.print)

    console.print(
        f"\n[bold green]Ingested[/] {report.labels_ingested} labels / "
        f"{report.chunks_written} chunks "
        f"({report.from_cache} cached, {report.fetched_live} fetched live)."
    )
    if report.labels_superseded:
        console.print(f"[dim]Removed {report.labels_superseded} superseded label(s).[/]")
    if report.no_label_found:
        console.print(f"[yellow]No openFDA label:[/] {', '.join(report.no_label_found)}")
    if report.no_usable_sections:
        console.print(f"[yellow]No usable sections:[/] {', '.join(report.no_usable_sections)}")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The clinical question to answer.")],
    top_k: Annotated[int | None, typer.Option("--top-k", "-k")] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help=f"Override RX_LLM_PROVIDER: {'|'.join(PROVIDERS)}."),
    ] = None,
    show_context: Annotated[
        bool, typer.Option("--show-context", help="Print the retrieved excerpts.")
    ] = False,
) -> None:
    """Answer a question from the ingested labels, with citations."""
    settings = get_settings()
    try:
        chat_model = get_chat_model(provider)
        with connect() as conn:
            result = answer_question(
                question,
                retriever=PgVectorRetriever(conn=conn, k=top_k or settings.top_k),
                chat_model=chat_model,
                top_k=top_k,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, then re-raised as exit
        _explain_failure(exc, provider_name(provider))
        raise typer.Exit(code=1) from exc

    if show_context:
        table = Table(title="Retrieved context", show_lines=False)
        table.add_column("#", justify="right", style="cyan", no_wrap=True)
        table.add_column("Score", justify="right")
        table.add_column("Drug")
        table.add_column("Section")
        for chunk in result.chunks:
            table.add_row(chunk.citation, f"{chunk.score:.3f}", chunk.drug, chunk.section)
        console.print(table)

    style = "yellow" if result.refused else "green"
    title = "Insufficient evidence" if result.refused else "Answer"
    console.print(Panel(result.text or "(empty response)", title=title, border_style=style))

    cited = ", ".join(f"C{m}" for m in result.cited_markers) or "none"
    console.print(f"[dim]model={result.model}  cited={cited}[/]")

    if result.invalid_markers:
        bad = ", ".join(f"C{m}" for m in result.invalid_markers)
        console.print(f"[bold red]Hallucinated citations:[/] {bad} (not in retrieved context)")
    elif result.uncited:
        console.print("[yellow]Warning:[/] answer cited no excerpts.")


def _explain_failure(exc: Exception, provider_name: str) -> None:
    """Turn the predictable setup failures into actionable messages.

    These three account for essentially every first-run failure; a stack trace
    for "you have not added AWS keys yet" is not useful to anyone.
    """
    text = str(exc)

    if "OPENAI_API_KEY" in text or "openai_api_key" in text:
        console.print(
            "[bold red]No OpenAI API key found.[/]\n"
            "Put [cyan]OPENAI_API_KEY=sk-...[/] in .env (it is gitignored), or export it "
            "in your shell. To use AWS instead, run with [cyan]--provider bedrock[/]."
        )
        return

    if "AuthenticationError" in type(exc).__name__ or "Incorrect API key" in text:
        console.print(
            "[bold red]OpenAI rejected the API key.[/]\n"
            "Check for a stale or truncated value in .env.\n"
            f"[dim]{text}[/]"
        )
        return

    if "RateLimit" in type(exc).__name__ or "insufficient_quota" in text:
        console.print(
            "[bold red]OpenAI rate limit or quota exceeded.[/]\n"
            "If this is a brand-new key, confirm the account has billing credit.\n"
            f"[dim]{text}[/]"
        )
        return

    # Parenthesised deliberately: without it, `and` binds tighter than `or` and
    # this branch swallows Postgres's "relation ... does not exist" below.
    if provider_name == "openai" and ("model_not_found" in text or "invalid_model" in text):
        console.print(
            "[bold red]OpenAI does not recognise that model.[/]\n"
            f"Check [cyan]RX_OPENAI_MODEL[/] in .env.\n[dim]{text}[/]"
        )
        return

    if "Could not resolve AWS credentials" in text or "NoCredentialsError" in type(exc).__name__:
        console.print(
            "[bold red]No AWS credentials found.[/]\n"
            "Set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION in .env, and "
            "request model access for Claude Haiku 4.5 in the Bedrock console for that "
            "region. To use OpenAI instead, run with [cyan]--provider openai[/]."
        )
        return

    if "AccessDenied" in text or "don't have access to the model" in text:
        console.print(
            "[bold red]Bedrock denied the request.[/]\n"
            "Usually one of: model access not granted for this model in this region, "
            "or the IAM identity lacks bedrock:InvokeModel.\n"
            f"[dim]{text}[/]"
        )
        return

    if "does not exist" in text or "Connection refused" in text or "could not connect" in text:
        console.print(
            "[bold red]Cannot reach Postgres.[/]\n"
            "Start it with [cyan]docker compose up -d[/], then run [cyan]rxclarify ingest[/]."
        )
        return

    console.print(f"[bold red]{type(exc).__name__}:[/] {text}")


@app.command("db-stats")
def db_stats() -> None:
    """Show corpus size and configured providers."""
    settings = get_settings()
    with connect() as conn:
        counts = table_counts(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT section, count(*) AS n
                FROM chunks GROUP BY section ORDER BY n DESC
                """
            )
            sections = cur.fetchall()

    table = Table(title="RxClarify corpus")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("labels", str(counts.get("labels", 0)))
    table.add_row("chunks", str(counts.get("chunks", 0)))
    missing = counts.get("chunks_missing_embedding", 0)
    table.add_row(
        "chunks missing embedding",
        f"[red]{missing}[/]" if missing else "0",
    )
    console.print(table)

    if sections:
        by_section = Table(title="Chunks by section")
        by_section.add_column("Section")
        by_section.add_column("Chunks", justify="right")
        for row in sections:
            by_section.add_row(row["section"], str(row["n"]))
        console.print(by_section)

    console.print(
        f"[dim]provider={settings.llm_provider}  model={settings.active_model}  "
        f"top_k={settings.top_k}[/]"
    )


if __name__ == "__main__":
    app()
