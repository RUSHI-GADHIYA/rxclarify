# RxClarify Runbook

How to start, run, test, and troubleshoot the project. Written for Windows +
PowerShell, which is what this repo is developed on.

Every command below assumes you are in the repo root:

```powershell
cd C:\Users\rushi\Downloads\claude_RAG
```

---

## The short version

After one-time setup (§0), this is all you need:

```powershell
.\.venv\Scripts\python.exe app.py
```

Or double-click **`run.bat`** — batch files ignore PowerShell's execution
policy, so it works even if §0.1 is unresolved.

`app.py` runs the checks in §1 and §2 for you: it starts Docker if it is
stopped, brings up the database and waits for it to be healthy, offers to build
the corpus if it is empty, warns if any chunk is missing an embedding, then
opens the UI in your browser.

**Expected startup output:**

```
RxClarify - starting up

  provider     openai / gpt-5.6-luna
  docker       running
  database     healthy
  corpus       58 labels / 2327 chunks

Opening http://127.0.0.1:7860  (Ctrl+C to stop)
```

Any line that fails prints one plain sentence telling you what to fix — no
stack trace. The rest of this runbook is the manual equivalent, which is what
you want when something breaks or when you are scripting.

---

## 0. One-time setup

You only do this once per machine. Skip to §1 if the project already runs.

### 0.1 Allow PowerShell to activate the virtualenv

Windows blocks script execution by default, so `Activate.ps1` fails with
*"running scripts is disabled on this system"*. Fix it once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

`RemoteSigned` still blocks unsigned scripts downloaded from the internet; it
only permits local ones. This is the standard developer setting.

> **Don't want to change the policy?** You never have to activate. Calling the
> executables by full path works under any policy:
> `.\.venv\Scripts\rxclarify.exe db-stats`. That is the style used in §6 so the
> commands work either way.

### 0.2 Create the virtualenv and install the project

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

`-e` installs in *editable* mode: your edits to `src/` take effect immediately
with no reinstall. `[dev]` adds pytest and ruff.

### 0.3 Create your `.env`

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in **`OPENAI_API_KEY=sk-...`**. Everything else has a working default.
`.env` is gitignored — your key never gets committed.

### 0.4 Verify the install

```powershell
.\.venv\Scripts\rxclarify.exe --help
```

You should see the three commands: `ingest`, `ask`, `db-stats`.

---

## 1. Start the runtime

Two things must be running: **Docker Desktop** (which hosts Postgres) and the
**database container**.

```powershell
# 1. Start Docker Desktop if it isn't running (takes ~30-60s to become ready)
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# 2. Wait until the daemon answers, then start the database
docker info --format '{{.ServerVersion}}'      # errors until the daemon is up
docker compose up -d
```

Check it came up healthy:

```powershell
docker ps --filter name=rxclarify-db --format "{{.Names}} | {{.Status}}"
```

**Expected:** `rxclarify-db | Up 2 minutes (healthy)`

If it says `(health: starting)`, wait ten seconds and check again. Postgres
takes a moment to accept connections.

> Your data lives in a Docker **named volume** (`claude_rag_pgdata`), not in the
> container. Restarting or recreating the container does not lose the corpus.

---

## 2. Verify the corpus

```powershell
.\.venv\Scripts\rxclarify.exe db-stats
```

**Expected:**

```
labels                     58
chunks                   2327
chunks missing embedding    0
```

plus a per-section breakdown and a config line showing your provider and model.

**How to read this:**

| Row | What it means if wrong |
|---|---|
| `labels` = 0 | Database is empty — run the ingest (§5) |
| `chunks` = 0 but labels > 0 | Ingest failed partway; re-run it |
| `chunks missing embedding` > 0 | **Broken.** Some chunks are unsearchable. Re-run the ingest. |

That last row is the one that matters. A chunk without an embedding is invisible
to retrieval, so the system would quietly answer from an incomplete corpus.

---

## 3. Run the test suite

```powershell
.\.venv\Scripts\python.exe -m pytest
```

**Expected:** `53 passed`

These tests are **free and offline** — no API key, no database, no network. The
chat model is faked, so you can run them as often as you like. Run them after
every change.

Useful variations:

```powershell
.\.venv\Scripts\python.exe -m pytest -v                    # one line per test
.\.venv\Scripts\python.exe -m pytest tests\test_llm.py     # one file
.\.venv\Scripts\python.exe -m pytest -k citation           # tests matching a name
.\.venv\Scripts\python.exe -m pytest -x                    # stop at first failure
```

### Check style too

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests    # lint
.\.venv\Scripts\python.exe -m ruff format src tests   # auto-format
```

---

## 4. Ask questions (this costs money)

Each query costs roughly **$0.0004** — about 4 cents per hundred.

```powershell
.\.venv\Scripts\rxclarify.exe ask "Can a patient on warfarin take fluconazole?"
```

Add `--show-context` to see what retrieval actually found, which is the single
most useful debugging flag in the project:

```powershell
.\.venv\Scripts\rxclarify.exe ask "Does clarithromycin interact with simvastatin?" --show-context
```

### The five queries worth running

These exercise different behaviours. Run all five to convince yourself the
system works.

| # | Query | What it proves |
|---|---|---|
| 1 | `"Can a patient on warfarin take fluconazole?"` | Retrieval + citation on a real interaction |
| 2 | `"Does clarithromycin interact with simvastatin?"` | A contraindication, correctly identified |
| 3 | `"What are the boxed warning risks of combining an opioid with a benzodiazepine?"` | Retrieval across *two different labels* |
| 4 | `"What is the capital of France?"` | Refusal on an obviously off-topic question |
| 5 | `"Does ibuprofen interact with naproxen?"` | **The important one.** Neither drug is in the corpus. Retrieval still returns six confident-looking chunks about *other* drugs. A naive RAG answers from them; this one must refuse. |

### Reading the output

```
model=gpt-5.6-luna  cited=C1, C2
```

- `cited=` — which excerpts the answer used. Every `[Cn]` in the text should map to a row in the `--show-context` table.
- **Red "Hallucinated citations"** — the model cited an excerpt it was never shown. This is a genuine failure and should never appear.
- **Yellow "answer cited no excerpts"** — it answered without grounding. Suspicious unless it was a refusal.

### Other options

```powershell
# Retrieve more or fewer chunks (default 6)
.\.venv\Scripts\rxclarify.exe ask "..." -k 10

# Use AWS Bedrock instead of OpenAI for one query
.\.venv\Scripts\rxclarify.exe ask "..." --provider bedrock
```

---

## 5. Rebuild the corpus

You normally never need this — the corpus is already loaded and survives
restarts. Run it after changing chunking, embeddings, or the drug list.

```powershell
.\.venv\Scripts\rxclarify.exe ingest
```

**Expected:** `Ingested 58 labels / 2327 chunks (58 cached, 0 fetched live)`

`58 cached` means it reused the openFDA responses in `data/raw/` — **no network
calls, no API cost**. Embeddings are recomputed locally on CPU, which takes a
few minutes.

```powershell
# Just the first 3 drugs, for a quick smoke test
.\.venv\Scripts\rxclarify.exe ingest --limit 3

# Re-download from openFDA, ignoring the cache (~58 requests)
.\.venv\Scripts\rxclarify.exe ingest --refresh
```

Re-running is safe: chunks are replaced per label and superseded labels are
pruned, so you cannot end up with duplicates.

### Starting completely fresh

Only if the database is genuinely broken. **This destroys the corpus:**

```powershell
docker compose down -v     # -v deletes the data volume
docker compose up -d       # schema is re-applied automatically
.\.venv\Scripts\rxclarify.exe ingest
```

---

## 6. Look inside the database

Worth doing at least once — it makes the retrieval layer concrete.

```powershell
# Interactive SQL session (type \q to quit)
docker compose exec db psql -U rx -d rxclarify
```

Or run a single query:

```powershell
# Which drugs are loaded?
docker compose exec -T db psql -U rx -d rxclarify -c "SELECT query_generic, brand_name FROM labels ORDER BY query_generic LIMIT 10;"

# Confirm both indexes exist (HNSW = dense search, GIN = Phase 2 hybrid)
docker compose exec -T db psql -U rx -d rxclarify -c "SELECT indexname FROM pg_indexes WHERE tablename='chunks';"

# Read an actual chunk
docker compose exec -T db psql -U rx -d rxclarify -c "SELECT drug.brand_name, c.section, left(c.text, 200) FROM chunks c JOIN labels drug ON drug.id=c.label_id WHERE c.section='drug_interactions' LIMIT 3;"
```

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell execution policy | §0.1, or call `.exe` files directly |
| `No OpenAI API key found` | `OPENAI_API_KEY` blank in `.env`, or you're in a shell that started before you saved it | Fill it in; open a new terminal |
| `OpenAI rejected the API key` | Truncated or stale key | Re-copy it; check for stray quotes or spaces |
| `rate limit or quota exceeded` | New account with no billing credit | Add credit at platform.openai.com |
| `Cannot reach Postgres` | Docker Desktop or container not running | §1 |
| `error during connect ... dockerDesktopLinuxEngine` | Docker daemon down | Start Docker Desktop, wait for it |
| `db-stats` shows 0 labels | Empty database | §5 |
| `Input should be 'openai' or 'bedrock'` | `RX_LLM_PROVIDER` in `.env` has an old value | Set it to `openai` |
| `ModuleNotFoundError: rxclarify` | Project not installed into the venv | §0.2 |
| Answer shows red "Hallucinated citations" | Real grounding failure | Note the question — this is a genuine bug worth investigating |

**Reading errors:** the CLI turns the predictable setup failures into plain
guidance rather than a stack trace. A raw traceback means something genuinely
unexpected — read the last line first.

---

## 8. Shutting down

```powershell
docker compose stop     # stop the database, keep the data
```

Or just leave it running; it costs nothing meaningful. Use `docker compose down`
to remove the container (data still survives in the volume). Only
`docker compose down -v` destroys data.

---

## 9. Acceptance checklist

Run through this to confirm the whole project is healthy:

- [ ] `docker ps` shows `rxclarify-db` as `(healthy)`
- [ ] `rxclarify db-stats` → 58 labels, 2327 chunks, **0** missing embeddings
- [ ] `pytest` → 53 passed
- [ ] `ruff check src tests` → All checks passed
- [ ] Query 1 answers with citations, and every `[Cn]` appears in `--show-context`
- [ ] Query 5 (ibuprofen/naproxen) **refuses** despite retrieval returning chunks
- [ ] No red "Hallucinated citations" on any query
- [ ] `rxclarify ingest` → `58 cached, 0 fetched live` (proves the cache works)

If all eight pass, Phase 1 is working correctly.

---

## Command reference

| Command | Cost | Needs DB | Needs API key |
|---|---|---|---|
| `pytest` | free | no | no |
| `ruff check src tests` | free | no | no |
| `rxclarify db-stats` | free | yes | no |
| `rxclarify ingest` | free* | yes | no |
| `rxclarify ask "..."` | ~$0.0004 | yes | yes |

\* Free from cache. `--refresh` re-downloads from openFDA, which is also free but
subject to a 1,000 requests/day limit.
