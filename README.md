# RxClarify — grounded drug-label Q&A with citations and a refusal path

A retrieval-augmented assistant over public FDA drug labels. It answers
interaction, dosing, and contraindication questions **only** from retrieved
label text, cites the supporting excerpt inline, and says
`INSUFFICIENT_EVIDENCE` when the retrieved context does not support an answer.

> **Status: Phase 1 of 4 — working baseline.** Dense retrieval and grounded
> generation are running end to end. The measured-quality work (hybrid
> retrieval, reranking, a hand-labelled gold set, RAGAS) is Phase 2–3; the
> results table below is deliberately empty until those numbers are real.

> **Data:** public [openFDA](https://open.fda.gov/apis/drug/label/) prescription
> labels only. No patient data of any kind. Not clinical advice.

## Why this exists

Pharmacy staff cross-check interactions and dosing across scattered PDFs and
monographs. In that setting a confident wrong answer is worse than no answer —
so the design goal is not fluency, it is **groundedness you can measure**: every
factual sentence carries a citation, every citation is validated against what
was actually retrieved, and unanswerable questions get refused rather than
guessed.

## Architecture

```
  data/drug_list.yml (58 generics)
            │
            ▼
  openFDA /drug/label.json ──► candidate selection ──► section extraction
  (5 candidates per drug,       (single-ingredient,     (7 clinically
   cached to data/raw/)          systemic, rich)         relevant sections)
            │
            ▼
  chunking (~1200 chars, within-section)
            │
            ├─ text        → shown to the model
            └─ embed_text  → "Drug — Section: ..." → bge-small-en-v1.5 (384d)
                                        │
                                        ▼
                        Postgres + pgvector  ┌ embedding  → HNSW (cosine)
                                             └ tsv        → GIN (Phase 2 sparse)
                                        │
  question ─► query_embed ─► dense ANN top-k ─┘
                                        │
                          LCEL chain ────┤
                                        ▼
    RunnableParallel{question, docs} ▸ assign(message = format ▸ prompt ▸ model)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │  ChatOpenAI      → gpt-5.6-luna       │
                    │  ChatBedrock…    → Claude Haiku 4.5   │
                    └───────────────────┬───────────────────┘
                                        ▼
                     citation validation → Answer
                     (markers not in context = hallucination)
```

The chain stops at the chat model rather than piping through `StrOutputParser`,
because the raw `AIMessage` carries `usage_metadata` — and per-query token cost
is one of the numbers this project exists to report.

## Results

Measured on a 61-question gold set (46 answerable + 15 adversarial). Retrieval
and generation are reported separately because they fail for different reasons
and are fixed by different changes.

### Retrieval

Scored against known `supporting_chunk_id`s — no LLM judge involved, so these
numbers carry no model-grading caveat. `k` is how many excerpts reach the model.

| Config | Recall@6 | MRR | nDCG@6 | Recall@3 | p50 latency |
|---|---|---|---|---|---|
| dense only | 0.935 | 0.647 | 0.718 | 0.783 | 9 ms |
| sparse only | 0.652 | 0.412 | 0.471 | 0.522 | 14 ms |
| **hybrid (dense + sparse, RRF)** | **0.978** | 0.686 | 0.758 | 0.826 | 10 ms |
| hybrid + cross-encoder rerank | 0.935 | **0.718** | **0.773** | **0.870** | 1311 ms |

**Hybrid retrieval is the default**: +4.3 points of recall over dense for one
millisecond. Reranking is not, and the reason is the interesting part — see
below.

### Generation and grounding

Hybrid retrieval, `gpt-5.6-luna`, all 61 questions.

| Metric | Result |
|---|---|
| Faithfulness (40 answers, 206 claims, `gpt-5.6-terra` judge) | **0.994** |
| Hallucinated-citation rate | **0.000** |
| Uncited-answer rate | **0.000** |
| Correct refusal on adversarial questions | **1.000** (15/15) |
| Over-refusal on answerable questions | **0.000** |
| Cost per question | $0.00041 |
| p50 / p95 latency | 1.2 s / 2.4 s |

Total spend for every number on this page, including all ablations: **≈ $0.85**.

## What the measurements changed

Three findings that only a measured system produces, each of which changed the
code.

**Reranking trades recall for precision — so it depends on `k`.** At k=6 the
cross-encoder promoted 7 gold chunks to first place but pushed 2 out of the top
6 entirely: MRR up, recall down. At k=3 it wins outright (0.870 vs 0.826). The
rule that falls out: rerank when you can only afford few excerpts, skip it when
you can afford more. It also costs 130× the latency, which decides most
borderline cases.

**Optimising a component made the system worse.** Postgres `websearch_to_tsquery`
joins terms with AND, so a natural-language question demanded six terms in one
chunk and matched nothing — sparse recall@6 was 0.267. Relaxing it to OR nearly
tripled sparse in isolation (0.267 → 0.622) and *dropped hybrid* from 0.978 to
0.933, because six weak single-term matches per query diluted the dense ranking
during fusion. The best hybrid uses the worst standalone sparse retriever.
Strict AND contributes nothing when lexical matching has nothing useful to say,
and contributing nothing beats contributing noise.

**The eval caught a bug in the eval.** One adversarial question asked for the
incidence of dizziness with gabapentin, on the assumption that trial figures
live only in `adverse_reactions`, which is not ingested. The system answered it
correctly with valid citations and was scored as a failure — because the
gabapentin *warnings* section states the figure. The gold label was wrong, not
the system. It is relabelled and kept, and `eval/validate_gold_set.py` now
probes each adversarial question with the vocabulary its **answer** would use
rather than the question's own words, which is what would have caught it.

## Quickstart

Requires Python 3.11+, Docker, and an OpenAI API key.

```bash
python -m venv .venv                  # then activate it
pip install -e ".[dev]"
cp .env.example .env                  # paste your OPENAI_API_KEY into it
python app.py                         # everything else is automatic
```

`app.py` starts Docker if it is stopped, brings up the database, offers to build
the corpus if it is empty, and opens the UI in your browser. On Windows,
`run.bat` does the same by double-click.

<!-- TODO: add a screenshot or short GIF of the UI here — this is the demo
     artifact a reviewer looks for first. -->

### Or drive it from the command line

```bash
rxclarify ingest                      # ~58 labels, ~2,300 chunks, a few minutes
rxclarify ask "Can a patient on warfarin take fluconazole?" --show-context
rxclarify db-stats
```

Re-running `ingest` is safe: openFDA responses are cached to `data/raw/`, chunks
are replaced per label, and superseded labels are pruned.

### Switching to Claude on Bedrock

1. IAM identity with `bedrock:InvokeModel` (add `bedrock:ApplyGuardrail` now — Phase 3 needs it).
2. Request model access for **Claude Haiku 4.5** in the Bedrock console, **per region**. This is the step that is easy to miss.
3. Put `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_REGION` in `.env`, then:

```bash
rxclarify ask "Does clarithromycin interact with simvastatin?" --provider bedrock
```

Bedrock goes through `ChatBedrockConverse`, so it drops into the same LCEL chain
as OpenAI — switching providers changes one env var and nothing in the pipeline.

## Commands

| Command | Purpose |
|---|---|
| `python app.py` | Start everything and open the UI. `run.bat` on Windows. |
| `rxclarify ingest [--limit N] [--refresh]` | Fetch, chunk, embed, load. `--refresh` bypasses the disk cache. |
| `rxclarify ask "<question>" [--show-context] [-k N] [--provider ...]` | Answer with citations. |
| `rxclarify db-stats` | Corpus size, chunks per section, embedding-coverage check. |

## Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangChain LCEL (`langchain-core` 1.5) | Retriever, prompt, and model compose as one pipeable chain |
| Vector + sparse store | Postgres 17 + pgvector, HNSW + GIN | One store for both halves of Phase 2 hybrid search |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU) | 384-dim, no PyTorch dependency |
| Generation | `ChatOpenAI` → gpt-5.6-luna / `ChatBedrockConverse` → Claude Haiku 4.5 | Both are `BaseChatModel`, so the chain is provider-agnostic |
| Interfaces | Gradio UI (`app.py`) + Typer CLI (`rxclarify`) | One-command demo for people, scriptable commands for pipelines |

## Design notes

**Query and document embeddings use different code paths.** bge is asymmetric:
passages go through `embed()`, questions through `query_embed()`, which applies
the model's instruction prefix. Using one for both silently costs recall.

**Chunks carry their subject into the vector.** Label prose says "this drug"
constantly. Embedding `"Coumadin (warfarin sodium) — Drug Interactions: ..."`
instead of the bare paragraph keeps the drug name retrievable.

**Candidate selection happens at ingest, not retrieval.** openFDA's first hit
for "ciprofloxacin hydrochloride" is CILOXAN, an eye drop; for "metformin" it is
a combination product. `ingest/select.py` scores five candidates on
single-ingredient, systemic route, and interaction content. The gold set is
built against this corpus, so the fix belongs upstream of it.

**Hallucinated citations are counted without a judge model.** The model only
ever sees `[C1]..[Cn]` markers, never database IDs. Any marker outside the
retrieved set is an unambiguous fabrication — the cheapest grounding check in
the pipeline, and it runs on every query.

**Reasoning effort is pinned to `none`, deliberately.** GPT-5-family models are
reasoning models, and `langchain-openai` *silently* strips `temperature` for
them at any effort above `none` — no error, no warning. Answering from six
supplied excerpts is extraction, not reasoning, so `none` is at once the
cheapest setting and the only one where `temperature=0` is honoured.
`tests/test_llm.py` asserts this, because a regression would quietly invalidate
every Phase 2 number rather than failing loudly.

Relatedly, there is no `seed`: setting `reasoning` routes through OpenAI's
Responses API, which has no such parameter — though it still appears in
LangChain's payload dict, so only a real call reveals the `TypeError`. Runs are
near-deterministic, not bit-reproducible.

**Label text is passed as a template variable, never formatted into the
prompt.** SPL dose tables are full of braces; interpolating them into a
`ChatPromptTemplate` would make LangChain try to resolve them as variables.

## Roadmap

- **Phase 1 — baseline (done).** Ingestion, dense retrieval, grounded generation, citation validation, CLI.
- **Phase 2 — measurement (done, except the chunking ablation).** 61-question gold set, eval harness, hybrid retrieval, cross-encoder reranking, faithfulness judge. The chunking ablation (fixed vs semantic vs parent-document) is still outstanding: it needs a schema change and three full re-embeddings, and teaches less than the rest.
- **Phase 3 — guardrails.** Bedrock contextual grounding + PII redaction, adversarial refusal set, tracing.
- **Phase 4 — production.** FastAPI + React with citation highlighting, containerized deploy, GitHub Actions gating merges on a faithfulness threshold.

## Known limitations

- One label per generic. Real products have many SPLs; this corpus takes the best-scoring one.
- Retrieval is single-drug biased. A "drug A + drug B" question tends to return only the drug whose label names the interaction — a known motivation for Phase 2 hybrid retrieval.
- `empagliflozin` resolves to a combination product (no single-ingredient prescription SPL scored high enough in the top 5).
- Generation now requires an API key. There is no free local fallback, so `rxclarify ask` costs a fraction of a cent per query rather than nothing.
- **The gold set is author-written, not reviewed by a pharmacist.** Questions were written from real chunk text, so `supporting_chunk_id` is a fact and every retrieval and refusal number above is solid. Faithfulness is the one metric graded by a model, and it is reference-free — the written "ideal answer" never enters the calculation. Spot-checking the judge found one false positive out of two flags before the prompt was corrected, so treat 0.994 as a close estimate rather than an exact figure.
- 46 answerable questions is enough to separate 0.65 from 0.98, not enough to resolve differences of one or two points. Read the table for direction, not precision.