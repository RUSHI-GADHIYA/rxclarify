"""The grounding contract: answer only from context, cite, or refuse.

This prompt is the thing Phase 3's faithfulness numbers are measured against,
so it is kept in one place and versioned deliberately. Phase 3 hardens it with
a Bedrock contextual-grounding guardrail; the refusal path exists here from the
start so we can measure the unguarded baseline first.
"""

from __future__ import annotations

from rxclarify.retrieval.base import RetrievedChunk

REFUSAL_TOKEN = "INSUFFICIENT_EVIDENCE"

PROMPT_VERSION = "p1-grounded-v1"

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


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        header = f"[{chunk.citation}] {chunk.drug} — {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        # Keep the contract identical on the empty-retrieval path so the model
        # produces the same refusal shape rather than a free-form apology.
        return (
            "CONTEXT:\n(no excerpts were retrieved)\n\n"
            f"QUESTION: {question}\n\n"
            f"Reply with exactly `{REFUSAL_TOKEN}` and one sentence explaining why."
        )

    return f"CONTEXT:\n{format_context(chunks)}\n\nQUESTION: {question}"
