"""Faithfulness: is every claim in the answer supported by the retrieved context?

**Why this is hand-written rather than RAGAS.** RAGAS 0.4.3 imports
`langchain_community.chat_models.vertexai`, which does not exist in
langchain-community 1.x — the stack this project standardises on. Installing it
also forced `openai` back from 3.1 to 2.54. Pinning the whole dependency tree
backwards to satisfy one eval library is a bad trade against a working
pipeline, so the metric is implemented directly. It is ~80 lines and uses the
provider abstraction already in `rxclarify.llm`.

The algorithm is the standard one, and worth understanding rather than treating
as a black box:

  1. Decompose the answer into atomic factual claims.
  2. Show the judge the retrieved context and ask, per claim, whether the
     context supports it.
  3. faithfulness = supported claims / total claims.

**Reference-free.** The gold set's `ideal_answer` is never shown to the judge —
only the answer and the context it was given. That matters here: this gold set
was author-written rather than expert-reviewed, and a reference-free metric
keeps that limitation out of the calculation entirely.

The judge must be stronger than the system under test, and must stay fixed
across runs — changing it invalidates comparison with every earlier number.
"""

from __future__ import annotations

import json
import re

CLAIMS_SYSTEM = """\
You break an answer into atomic factual claims.

Rules:
- One verifiable assertion per claim; split compound sentences.
- Drop hedges, restatements of the question, and advice to consult a clinician.
- Drop citation markers like [C1].
- Preserve specifics exactly: doses, percentages, drug names, conditions.

Return ONLY a JSON array of strings. No prose, no code fence."""

VERDICT_SYSTEM = """\
You check whether a CONTEXT supports each CLAIM.

Supported means the context states the claim or directly entails it. Outside
knowledge and general plausibility do not count.

Two rules that decide most borderline cases:
- Evidence may be spread across several sentences or several context blocks.
  Combine them. "X has a moderate association with Y reactions" in one sentence
  plus "Y reactions include Z" in the next DOES support "X is associated with Z".
- Judge the substance, not the wording. A paraphrase, a reordering, or a
  narrower restatement of what the context says is supported.

Mark unsupported when the claim adds a specific the context does not contain -
a number, a threshold, a drug, or a population that simply is not there.

Return ONLY a JSON array of objects, one per claim, in the same order:
  [{"claim_index": 0, "supported": true, "why": "<8 words max>"}]
No prose, no code fence."""


def _parse_json_array(text: str) -> list:
    """Tolerate a code fence or stray prose around the JSON."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*]", cleaned, flags=re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    return parsed if isinstance(parsed, list) else []


def extract_claims(judge, answer: str) -> list[str]:
    response = judge.invoke([("system", CLAIMS_SYSTEM), ("human", f"ANSWER:\n{answer}")])
    return [c for c in _parse_json_array(response.text) if isinstance(c, str) and c.strip()]


def verdicts(judge, claims: list[str], contexts: list[str]) -> list[dict]:
    if not claims:
        return []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims))
    context_block = "\n\n---\n\n".join(contexts)
    response = judge.invoke(
        [
            ("system", VERDICT_SYSTEM),
            ("human", f"CONTEXT:\n{context_block}\n\nCLAIMS:\n{numbered}"),
        ]
    )
    parsed = _parse_json_array(response.text)
    return [v for v in parsed if isinstance(v, dict)]


def score_answer(judge, answer: str, contexts: list[str]) -> dict:
    """Faithfulness for one answer, plus the working so it can be audited."""
    claims = extract_claims(judge, answer)
    if not claims:
        # No factual claims (e.g. a refusal) is vacuously faithful, but flag it
        # so these do not quietly inflate the mean.
        return {"faithfulness": None, "n_claims": 0, "claims": [], "verdicts": []}

    results = verdicts(judge, claims, contexts)
    supported = sum(1 for v in results if v.get("supported") is True)
    return {
        "faithfulness": supported / len(claims),
        "n_claims": len(claims),
        "n_supported": supported,
        "claims": claims,
        "verdicts": results,
    }
