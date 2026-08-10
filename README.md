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
                                        ▼
              grounded prompt (cite [C1]..[Cn] or refuse)
                                        │
                    ┌───────────────────┴───────────────────┐
                    │  Claude Haiku 4.5 on Bedrock (prod)   │
                    │  qwen2.5:3b via Ollama (free dev)     │
                    └───────────────────┬───────────────────┘
                                        ▼
                     citation validation → Answer
                     (markers not in context = hallucination)
```

## Results

Populated in Phase 2/3 against a 100-question hand-labelled gold set. Retrieval
metrics (context precision/recall) are reported separately from generation
metrics (faithfulness/answer relevancy), because they fail for different reasons
and get fixed by different changes.

| Metric | Dense only | + Hybrid | + Rerank |
|---|---|---|---|
| RAGAS faithfulness | — | — | — |
| RAGAS answer relevancy | — | — | — |
| Context precision | — | — | — |
| Context recall | — | — | — |
| Recall@6 / MRR | — | — | — |
| Unsupported-answer rate (adversarial set) | — | — | — |
| p50 latency / cost per query | — | — | — |

## Quickstart

Requires Python 3.11+, Docker, and (for the free path) [Ollama](https://ollama.com).

```bash
cp .env.example .env                  # defaults to the local Ollama provider
docker compose up -d                  # Postgres 17 + pgvector, schema auto-applied
pip install -e ".[dev]"
ollama pull qwen2.5:3b
rxclarify ingest                      # ~58 labels, ~2,300 chunks, a few minutes
rxclarify ask "Can a patient on warfarin take fluconazole?" --show-context
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

## Commands

| Command | Purpose |
|---|---|
| `rxclarify ingest [--limit N] [--refresh]` | Fetch, chunk, embed, load. `--refresh` bypasses the disk cache. |
| `rxclarify ask "<question>" [--show-context] [-k N] [--provider ...]` | Answer with citations. |
| `rxclarify db-stats` | Corpus size, chunks per section, embedding-coverage check. |

## Stack

| Layer | Choice | Why |
|---|---|---|
| Vector + sparse store | Postgres 17 + pgvector, HNSW + GIN | One store for both halves of Phase 2 hybrid search |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed (ONNX, CPU) | 384-dim, no PyTorch dependency |
| Generation | Claude Haiku 4.5 on Bedrock / qwen2.5:3b on Ollama | One `LLMProvider` protocol, swap by config |
| CLI | Typer + Rich | FastAPI + React arrive in Phase 4 |

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

## Roadmap

- **Phase 1 — baseline (done).** Ingestion, dense retrieval, grounded generation, citation validation, CLI.
- **Phase 2 — measurement.** Hybrid (dense + full-text) retrieval, reranking, 100-question gold set, RAGAS, chunking ablation.
- **Phase 3 — guardrails.** Bedrock contextual grounding + PII redaction, adversarial refusal set, tracing.
- **Phase 4 — production.** FastAPI + React with citation highlighting, containerized deploy, GitHub Actions gating merges on a faithfulness threshold.

## Known limitations

- One label per generic. Real products have many SPLs; this corpus takes the best-scoring one.
- Retrieval is single-drug biased. A "drug A + drug B" question tends to return only the drug whose label names the interaction — a known motivation for Phase 2 hybrid retrieval.
- `qwen2.5:3b` is a development convenience. It over-refuses and pads answers; it is not the basis for any published number.
- `empagliflozin` resolves to a combination product (no single-ingredient prescription SPL scored high enough in the top 5).
