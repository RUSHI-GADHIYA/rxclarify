# Evaluation

The measurement layer. Headline numbers live in the root `README.md`; this
explains how they are produced and what they can honestly support.

## Running it

```powershell
# Free. No LLM, no API key. Re-run as often as you like.
python eval/validate_gold_set.py
python eval/harness.py --config all
python eval/harness.py --config all --top-k 3

# Costs money. ~$0.025 per config for 61 questions.
python eval/harness.py --config hybrid --mode generate

# Costs money. ~$0.0156 per answer judged.
python eval/judge.py --config hybrid --limit 40
```

Results land in `eval/results/<config>_<mode>.json`, with per-question detail so
any aggregate can be traced back to the answer that produced it.

## Files

| File | Purpose |
|---|---|
| `gold_set.jsonl` | 61 questions: 46 answerable, 15 adversarial |
| `validate_gold_set.py` | Checks the gold set against the live corpus before anything trusts it |
| `metrics.py` | Recall@k, MRR, nDCG, precision — pure functions, unit-tested against worked examples |
| `harness.py` | Runs a retriever (and optionally the model) over the gold set |
| `faithfulness.py` | Claim-level faithfulness judge |
| `judge.py` | Scores faithfulness over a completed generation run |

## The gold set

Each answerable question was written from one specific chunk, so its
`supporting_chunk_id` is **a fact, not an opinion**. That is what makes the
retrieval numbers trustworthy regardless of who wrote the questions.

The 15 adversarial questions come in two kinds, both provably unanswerable:

- **absent drug** (7) — ibuprofen, naproxen, amoxicillin, aspirin, atenolol, sildenafil, celecoxib. None is in the corpus; the validator asserts this.
- **absent section** (8) — the drug *is* present, but the question asks about a section deliberately dropped at ingest (pharmacokinetics, adverse-reaction incidence, how-supplied, overdosage, clinical studies). Harder and more useful: the drug name retrieves strongly, so only real grounding produces a refusal.

### What it can and cannot support

| Metric | Ground truth | Trustworthy? |
|---|---|---|
| Recall@k, MRR, nDCG, precision | Known `supporting_chunk_id` | **Yes — factual** |
| Correct-refusal / over-refusal | Drugs and sections provably absent | **Yes — factual** |
| Hallucinated-citation, uncited rate | String check against retrieved markers | **Yes — factual** |
| Latency, tokens, cost | Provider usage fields | **Yes — factual** |
| Faithfulness | LLM judge | **Estimate** — see below |

The set is **author-written, not reviewed by a pharmacist**. That caveats one
row, not the table. Faithfulness is also *reference-free*: the judge sees only
the answer and the context it was given, never the written `ideal_answer`, so
the weakest part of the gold set stays out of the calculation.

## Why faithfulness is hand-written instead of RAGAS

RAGAS 0.4.3 imports `langchain_community.chat_models.vertexai`, which does not
exist in langchain-community 1.x — the stack this project uses. Installing it
also forced `openai` back from 3.1 to 2.54. Pinning the whole dependency tree
backwards to satisfy one eval library was a bad trade against a working
pipeline, so `faithfulness.py` implements the standard algorithm directly:

1. Decompose the answer into atomic factual claims.
2. Ask the judge, per claim, whether the retrieved context supports it.
3. `faithfulness = supported / total`.

About 80 lines, no new dependencies, and the prompt is inspectable — which
mattered when the first version produced a false positive.

## Judge model

`gpt-5.6-terra`: stronger than the `gpt-5.6-luna` under test, ~2.5× cheaper than
`sol`. **Keep it fixed** — changing the judge invalidates comparison with every
earlier run.

Judges have false positives. On the first five answers, one of two flagged
claims was wrong: the judge required single-sentence support, while the context
established the claim across two sentences. The prompt now says evidence may
span sentences and blocks, which fixed it. Assume some residual error and treat
faithfulness as a close estimate.

Refusals are skipped — they contain no factual claims, so judging them spends
money to learn nothing. Refusal correctness is measured for free by the harness.

## Reproducibility

Generation and judging both run at `reasoning.effort=none` with
`temperature=0`. That setting is load-bearing: at any higher effort
`langchain-openai` silently drops temperature and runs stop being comparable.

There is **no seed**. Setting `reasoning` routes through OpenAI's Responses API,
which has no `seed` parameter, and the Chat Completions path that accepts one
strips `temperature` for this model family instead — the worse trade. Runs are
near-deterministic, not bit-reproducible. Record the model ID, effort level and
date beside every result, and re-run a baseline before comparing against numbers
from an earlier session.

## Still outstanding

**Chunking ablation** — fixed-size (current) vs semantic vs parent-document.
Needs a `chunks.strategy` column so the three can coexist, plus three full
re-embeddings. Costs nothing in API spend; it is the most time-expensive and
least architecturally interesting item in Phase 2, which is why it is last.
