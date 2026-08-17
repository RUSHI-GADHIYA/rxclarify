"""Check the gold set against the live corpus before trusting any metric.

Three failure modes this catches, all of which corrupt results silently:

1. A `supporting_chunk_id` that does not exist -> recall for that question is
   permanently 0 and the retriever gets blamed.
2. An "absent drug" adversarial question naming a drug that *is* in the corpus
   -> a correct answer is scored as a failed refusal.
3. An "absent section" adversarial question whose answer happens to appear in
   an ingested section anyway -> same problem, harder to spot.

Run: python eval/validate_gold_set.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rxclarify.db import connect  # noqa: E402

GOLD_PATH = Path(__file__).parent / "gold_set.jsonl"


def load_gold() -> list[dict]:
    with GOLD_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    gold = load_gold()
    problems: list[str] = []

    answerable = [g for g in gold if g["answerable"]]
    adversarial = [g for g in gold if not g["answerable"]]

    with connect() as conn, conn.cursor() as cur:
        # 1. Every supporting chunk id resolves.
        wanted = sorted({cid for g in answerable for cid in g["supporting_chunk_ids"]})
        cur.execute("SELECT id FROM chunks WHERE id = ANY(%s)", (wanted,))
        found = {r["id"] for r in cur.fetchall()}
        for g in answerable:
            for cid in g["supporting_chunk_ids"]:
                if cid not in found:
                    problems.append(f"{g['id']}: chunk {cid} does not exist")
            if not g["supporting_chunk_ids"]:
                problems.append(f"{g['id']}: answerable but has no supporting chunk")

        # 2. Absent-drug questions must name drugs the corpus really lacks.
        cur.execute("SELECT lower(query_generic) AS g, lower(generic_name) AS n FROM labels")
        corpus_terms = " | ".join(f"{r['g']} {r['n'] or ''}" for r in cur.fetchall())
        for g in adversarial:
            if g["category"] != "adversarial_absent_drug":
                continue
            for drug in g["drugs"]:
                if drug.lower() in corpus_terms:
                    problems.append(f"{g['id']}: '{drug}' IS in the corpus - not adversarial")

        # 3. Absent-section questions: the drug SHOULD be present (that is the
        #    point - its name retrieves strongly), but the asked-about content
        #    should not be.
        #
        #    Probing with the question text is not enough. q060 asked for the
        #    incidence of dizziness with gabapentin; the question shares no
        #    wording with "dizziness occurred in 17% of patients", so full-text
        #    search on the question found nothing and the question looked safe.
        #    It was not - the corpus states the figure, and the eval run caught
        #    it only because the system answered correctly and was marked wrong.
        #
        #    So probe on `probe_terms` - the words the *answer* would contain -
        #    scoped to that drug's own label.
        print("\nabsent-section questions - probing the answer's vocabulary:")
        for g in adversarial:
            if g["category"] != "adversarial_absent_section":
                continue
            terms = g.get("probe_terms")
            if not terms:
                problems.append(f"{g['id']}: absent-section question has no probe_terms")
                continue

            hits = []
            for term in terms:
                cur.execute(
                    """
                    SELECT c.id, c.section, left(c.text, 100) AS preview
                    FROM chunks c
                    JOIN labels l ON l.id = c.label_id
                    WHERE l.query_generic = ANY(%(drugs)s) AND c.text ILIKE %(pattern)s
                    LIMIT 2
                    """,
                    {"drugs": g["drugs"], "pattern": f"%{term}%"},
                )
                hits.extend(cur.fetchall())

            if not hits:
                print(f"  {g['id']}  clean - no chunk in this drug's label mentions {terms}")
            elif g.get("probe_reviewed"):
                print(
                    f"  {g['id']}  {len(hits)} incidental match(es), reviewed and kept adversarial"
                )
            else:
                problems.append(
                    f"{g['id']}: probe terms {terms} match {len(hits)} chunk(s) in this drug's "
                    f"label (e.g. chunk {hits[0]['id']} in {hits[0]['section']}). Either the "
                    f"question is answerable, or set probe_reviewed:true after checking."
                )

    print(f"\n{len(gold)} questions: {len(answerable)} answerable, {len(adversarial)} adversarial")

    by_category: dict[str, int] = {}
    for g in gold:
        by_category[g["category"]] = by_category.get(g["category"], 0) + 1
    for cat, n in sorted(by_category.items()):
        print(f"  {cat:32} {n}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("\nOK: all supporting chunks exist; no adversarial drug is in the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
