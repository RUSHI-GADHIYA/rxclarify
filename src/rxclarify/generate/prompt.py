"""The grounding contract: answer only from context, cite, or refuse.

This prompt is what Phase 3's faithfulness numbers are measured against, so it
lives in one place and is versioned deliberately.

Note on templating: retrieved label text is injected as a *template variable*,
never formatted into the template string. Label text contains braces (dose
tables, chemical notation), and formatting it into the template would make
ChatPromptTemplate try to interpret them as variables.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from rxclarify.retrieval.base import RetrievedChunk

REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"

PROMPT_VERSION = "p2-lcel-v1"

NO_CONTEXT_SENTINEL = "(no excerpts were retrieved)"

SYSTEM_PROMPT = f"""\
You are a drug-information assistant for licensed pharmacy staff. You answer \
strictly from the FDA label excerpts provided in each request.

Rules:
1. Use ONLY the numbered CONTEXT excerpts. Do not use prior knowledge about \
these drugs, even if you are confident it is correct.
2. Cite the excerpt supporting each claim inline, as [C1], [C2]. Cite every \
factual sentence. You may cite more than one excerpt: [C1][C3].
3. Never cite an excerpt number that does not appear in the CONTEXT.
4. If the excerpts do not contain enough information to answer, reply with \
exactly `{REFUSAL_TOKEN}` on the first line, then one sentence naming what is \
missing. Do not guess, and do not answer from general knowledge.
5. If the excerpts only partially answer the question, answer the part they \
support with citations, then state plainly which part is unsupported.
6. Be concise and clinical. No preamble, no restating the question.
7. You provide drug information, not patient-specific clinical advice. Where \
the answer depends on the individual patient, say so.
"""

USER_TEMPLATE = "CONTEXT:\n{context}\n\nQUESTION: {question}"


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", USER_TEMPLATE)])


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, citable blocks.

    On empty retrieval this returns a sentinel rather than an empty string, so
    the model still sees a well-formed CONTEXT section and produces the normal
    refusal shape instead of a free-form apology.
    """
    if not chunks:
        return NO_CONTEXT_SENTINEL

    blocks = []
    for chunk in chunks:
        header = f"[{chunk.citation}] {chunk.drug} — {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)
