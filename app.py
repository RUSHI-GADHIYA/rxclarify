#!/usr/bin/env python
"""RxClarify — single entry point.

    python app.py

Brings up everything the app needs (database container, corpus) and opens a
browser UI for asking grounded questions against the drug-label corpus.

This file is a shell: preflight checks plus UI wiring. Every piece of domain
logic lives in the `rxclarify` package and is imported, not duplicated. The
`rxclarify` CLI remains the scriptable interface; this is the easy front door.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Importing config also loads .env into the process environment, which is what
# makes OPENAI_API_KEY visible to the OpenAI SDK. Do it before anything reads
# the environment.
from rxclarify.config import get_settings  # noqa: E402

SERVER_PORT = 8003

EXAMPLE_QUESTIONS = [
    "Can a patient on warfarin take fluconazole?",
    "Does clarithromycin interact with simvastatin?",
    "What are the boxed warning risks of combining an opioid with a benzodiazepine?",
    "What is the maximum daily dose of metformin?",
    "Does ibuprofen interact with naproxen?",
]


# --------------------------------------------------------------------------
# Preflight
# --------------------------------------------------------------------------
# Every check prints one plain line and, on failure, says exactly what to do.
# A stack trace here would be useless: these are setup problems, not bugs.


def say(message: str) -> None:
    print(message, flush=True)


def fail(message: str) -> None:
    print(f"\n{message}\n", flush=True)
    sys.exit(1)


def _run(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=REPO_ROOT)


def check_api_key() -> None:
    settings = get_settings()
    provider = settings.llm_provider

    if provider == "openai":
        import os

        if not os.environ.get("OPENAI_API_KEY", "").strip():
            fail(
                "No OpenAI API key.\n\n"
                f"  Add this line to {REPO_ROOT / '.env'} and run again:\n"
                "      OPENAI_API_KEY=sk-...\n\n"
                "  (.env is gitignored, so the key is never committed.)"
            )
    say(f"  provider     {provider} / {settings.active_model}")


def check_docker() -> None:
    if shutil.which("docker") is None:
        fail(
            "Docker is not installed, or not on PATH.\n\n"
            "  RxClarify stores its corpus in Postgres, which runs in Docker.\n"
            "  Install Docker Desktop, then run this again."
        )

    if _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=30).returncode == 0:
        say("  docker       running")
        return

    # The daemon is down. On Windows we can start Docker Desktop ourselves;
    # it takes 30-60s to accept connections, so poll rather than assume.
    if sys.platform == "win32":
        exe = Path("C:/Program Files/Docker/Docker/Docker Desktop.exe")
        if exe.exists():
            say("  docker       not running - starting Docker Desktop...")
            subprocess.Popen([str(exe)])
            deadline = time.time() + 120
            while time.time() < deadline:
                time.sleep(5)
                if _run(["docker", "info", "--format", "{{.ID}}"], timeout=30).returncode == 0:
                    say("  docker       running")
                    return

    fail(
        "Docker is installed but the daemon is not responding.\n\n"
        "  Start Docker Desktop, wait for it to say 'Engine running', then run this again."
    )


def check_database() -> None:
    result = _run(["docker", "compose", "up", "-d"], timeout=180)
    if result.returncode != 0:
        fail("Could not start the database container.\n\n  " + (result.stderr or "").strip())

    deadline = time.time() + 120
    status = ""
    while time.time() < deadline:
        probe = _run(
            ["docker", "inspect", "-f", "{{.State.Health.Status}}", "rxclarify-db"], timeout=30
        )
        status = (probe.stdout or "").strip()
        if status == "healthy":
            say("  database     healthy")
            return
        time.sleep(3)

    fail(
        f"The database container did not become healthy (last status: {status or 'unknown'}).\n\n"
        "  Try:  docker compose logs db"
    )


def check_corpus() -> None:
    from rxclarify.db import connect, table_counts

    try:
        with connect() as conn:
            counts = table_counts(conn)
    except Exception as exc:  # noqa: BLE001 - reported as a setup problem, not a crash
        fail(f"Could not query the database.\n\n  {exc}")

    labels = counts.get("labels", 0)
    chunks = counts.get("chunks", 0)
    missing = counts.get("chunks_missing_embedding", 0)

    if labels == 0:
        say("\n  The corpus is empty.")
        say("  Ingesting builds it from cached openFDA data - no network, no API cost,")
        say("  but embedding ~2,300 chunks on CPU takes a few minutes.\n")
        # Must happen before Gradio starts; a prompt after launch would hang
        # behind the server with nothing on screen to explain why.
        if input("  Ingest now? [Y/n] ").strip().lower() in ("", "y", "yes"):
            from rxclarify.ingest.pipeline import run_ingest

            report = run_ingest(progress=lambda m: say(f"    {m}"))
            say(
                f"\n  ingested     {report.labels_ingested} labels / {report.chunks_written} chunks"
            )
        else:
            fail("Cannot answer questions without a corpus.")
        return

    say(f"  corpus       {labels} labels / {chunks} chunks")

    if missing:
        say(
            f"\n  WARNING: {missing} chunks have no embedding and are invisible to retrieval.\n"
            "  Answers will be drawn from an incomplete corpus. Fix with:  rxclarify ingest\n"
        )


def preflight() -> None:
    say("\nRxClarify - starting up\n")
    check_api_key()
    check_docker()
    check_database()
    check_corpus()
    say("")


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

# Styles are inlined rather than passed to Blocks(css=...): Gradio 6 dropped the
# `css` and `theme` parameters, and inline style attributes work on every
# version. The three accents are semantic — failure, caution, success — and are
# legible on both the light and dark Gradio themes.
_BANNER = (
    "border-left:3px solid {c};padding:.55rem .85rem;"
    "background:{c}14;border-radius:2px;line-height:1.5"
)
BAD, WARN, OK = "#B03A32", "#A9691B", "#2F7D53"


def banner_html(colour: str, title: str, body: str = "") -> str:
    detail = f"<br><span style='opacity:.85'>{body}</span>" if body else ""
    return f"<div style='{_BANNER.format(c=colour)}'><b>{title}</b>{detail}</div>"


def estimate_cost(input_tokens: int | None, output_tokens: int | None) -> float | None:
    price = get_settings().price_per_mtok
    if price is None or input_tokens is None:
        return None
    per_in, per_out = price
    return input_tokens / 1e6 * per_in + (output_tokens or 0) / 1e6 * per_out


def ask(question: str, top_k: int):
    """Run one question through the chain and render the result."""
    import gradio as gr

    question = (question or "").strip()
    if not question:
        return (
            gr.update(value="", visible=False),
            "Type a question, or pick one of the examples below.",
            gr.update(value=[], visible=False),
            "",
        )

    from rxclarify.db import connect
    from rxclarify.generate.answer import answer_question
    from rxclarify.llm.factory import get_chat_model
    from rxclarify.retrieval.langchain_retriever import PgVectorRetriever

    try:
        chat_model = get_chat_model()
        with connect() as conn:
            result = answer_question(
                question,
                retriever=PgVectorRetriever(conn=conn, k=top_k),
                chat_model=chat_model,
                top_k=top_k,
            )
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI instead of the console
        return (
            gr.update(value=banner_html(BAD, "Request failed", str(exc)), visible=True),
            "",
            gr.update(value=[], visible=False),
            "",
        )

    # The hallucinated-citation case is the one this project exists to catch,
    # so it gets the loudest treatment available.
    if result.invalid_markers:
        bad = ", ".join(f"C{m}" for m in result.invalid_markers)
        banner = banner_html(
            BAD,
            f"Hallucinated citation: {bad}",
            "The model cited an excerpt it was never shown. Treat this answer as unreliable.",
        )
    elif result.refused:
        banner = banner_html(
            WARN,
            "Insufficient evidence",
            "The retrieved excerpts do not support an answer, so the assistant declined "
            "rather than guessing. This is correct behaviour.",
        )
    elif result.uncited:
        banner = banner_html(
            WARN,
            "Answer cited no excerpts",
            "Nothing was grounded to a source. Treat with caution.",
        )
    else:
        cited = ", ".join(f"C{m}" for m in result.cited_markers)
        banner = banner_html(OK, "Grounded answer", f"Cited {cited}")

    rows = [
        [c.citation, f"{c.score:.3f}", c.drug, c.section.replace("_", " "), c.text]
        for c in result.chunks
    ]

    bits = [f"model `{result.model}`"]
    if result.input_tokens is not None:
        bits.append(f"{result.input_tokens:,} in / {result.output_tokens:,} out")
    if result.latency_ms is not None:
        bits.append(f"{result.latency_ms / 1000:.1f}s")
    cost = estimate_cost(result.input_tokens, result.output_tokens)
    if cost is not None:
        bits.append(f"~${cost:.5f}")
    meta = (
        "<div style='font-size:.82rem;opacity:.72;font-variant-numeric:tabular-nums'>"
        f"{' &nbsp;·&nbsp; '.join(bits)}</div>"
    )

    return (
        gr.update(value=banner, visible=True),
        result.text,
        gr.update(value=rows, visible=True),
        meta,
    )


def build_ui():
    import gradio as gr

    settings = get_settings()

    with gr.Blocks(title="RxClarify", analytics_enabled=False) as demo:
        gr.Markdown(
            "## RxClarify\n"
            "Grounded question answering over public FDA drug labels. Every factual "
            "sentence is cited to a retrieved excerpt, and questions the corpus "
            "cannot support are refused rather than guessed.\n\n"
            "*Public label data only — no patient data, and not clinical advice.*"
        )

        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="Can a patient on warfarin take fluconazole?",
                lines=2,
                scale=5,
                autofocus=True,
            )
            with gr.Column(scale=1, min_width=140):
                ask_btn = gr.Button("Ask", variant="primary")
                top_k = gr.Slider(
                    minimum=1,
                    maximum=12,
                    value=settings.top_k,
                    step=1,
                    label="Excerpts",
                    info="How many to retrieve",
                )

        banner = gr.HTML(visible=False)
        answer = gr.Markdown(label="Answer")
        meta = gr.HTML()

        with gr.Accordion("Retrieved excerpts", open=False):
            gr.Markdown(
                "What retrieval actually found, in the order the model saw it. "
                "Every `[Cn]` in the answer should appear here."
            )
            context = gr.Dataframe(
                headers=["#", "Score", "Drug", "Section", "Excerpt"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True,
                visible=False,
                max_height=420,
            )

        gr.Examples(examples=[[q] for q in EXAMPLE_QUESTIONS], inputs=[question])

        outputs = [banner, answer, context, meta]
        ask_btn.click(ask, inputs=[question, top_k], outputs=outputs)
        question.submit(ask, inputs=[question, top_k], outputs=outputs)

    return demo


def main() -> None:
    preflight()
    say(f"Opening http://127.0.0.1:{SERVER_PORT}  (Ctrl+C to stop)\n")
    build_ui().launch(
        server_name="127.0.0.1",  # localhost only; never expose a public share link
        server_port=SERVER_PORT,
        inbrowser=True,
        show_error=True,
        quiet=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("\nStopped.")
