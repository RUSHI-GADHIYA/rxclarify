"""Score faithfulness over a completed generation run.

Reads `eval/results/<config>_generate.json` (produced by harness.py in
--mode generate) so the answers being judged are exactly the ones measured, and
no answer is generated twice.

    python eval/judge.py --config hybrid --limit 40

The judge is `gpt-5.6-terra`: stronger than the `gpt-5.6-luna` under test, and
~2.5x cheaper than `sol`. Keep it fixed — changing the judge invalidates
comparison with every earlier run.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from faithfulness import score_answer  # noqa: E402
from metrics import mean  # noqa: E402

from rxclarify.config import PRICING  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"
JUDGE_MODEL = "gpt-5.6-terra"


def build_judge(model: str):
    from langchain_openai import ChatOpenAI

    # Same reasoning-effort constraint as the answering model: GPT-5 family
    # models silently drop `temperature` at any effort above "none", and a judge
    # that is not deterministic makes runs incomparable.
    return ChatOpenAI(
        model=model,
        max_tokens=1500,
        reasoning={"effort": "none"},
        temperature=0,
        timeout=120,
        max_retries=3,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="hybrid")
    parser.add_argument("--limit", type=int, default=40, help="questions to judge (cost control)")
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    args = parser.parse_args()

    source = RESULTS_DIR / f"{args.config}_generate.json"
    if not source.exists():
        raise SystemExit(
            f"{source} not found.\n"
            f"Run first:  python eval/harness.py --config {args.config} --mode generate"
        )

    run = json.loads(source.read_text(encoding="utf-8"))
    # Refusals contain no factual claims to verify, so judging them spends money
    # to learn nothing. Refusal correctness is already measured, for free, by
    # the harness.
    candidates = [r for r in run["per_question"] if r.get("answer") and not r.get("refused")][
        : args.limit
    ]

    judge = build_judge(args.judge_model)
    print(f"judging {len(candidates)} answers from {source.name} with {args.judge_model}\n")

    scored, unfaithful = [], []
    for i, record in enumerate(candidates, start=1):
        result = score_answer(judge, record["answer"], record["contexts"])
        result["id"] = record["id"]
        scored.append(result)

        f = result["faithfulness"]
        flag = ""
        if f is not None and f < 1.0:
            unfaithful.append(result)
            flag = "  <-- unsupported claim"
        shown = "n/a" if f is None else f"{f:.2f}"
        print(
            f"  [{i:>2}/{len(candidates)}] {result['id']}  {shown}  "
            f"({result['n_claims']} claims){flag}"
        )

    values = [s["faithfulness"] for s in scored if s["faithfulness"] is not None]
    total_claims = sum(s["n_claims"] for s in scored)

    price = PRICING.get(args.judge_model)
    # Two judge calls per answer (decompose, then verify); measured token use
    # sits near these figures for this corpus.
    approx_cost = (
        (len(candidates) * 4200 / 1e6 * price[0] + len(candidates) * 600 / 1e6 * price[1])
        if price
        else 0.0
    )

    summary = {
        "config": args.config,
        "judge_model": args.judge_model,
        "judged": len(candidates),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "faithfulness_mean": mean(values),
        "answers_fully_faithful": sum(1 for v in values if v == 1.0),
        "total_claims": total_claims,
        "approx_judge_cost_usd": round(approx_cost, 4),
        "per_answer": scored,
    }

    out = RESULTS_DIR / f"{args.config}_faithfulness.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")

    print(f"\n  faithfulness      {summary['faithfulness_mean']:.3f}")
    print(f"  fully faithful    {summary['answers_fully_faithful']}/{len(values)} answers")
    print(f"  claims checked    {total_claims}")
    print(f"  judge cost        ~${approx_cost:.4f}")
    print(f"  -> {out.relative_to(REPO_ROOT)}")

    if unfaithful:
        print(f"\n  {len(unfaithful)} answer(s) with an unsupported claim:")
        for u in unfaithful[:5]:
            bad = [
                u["claims"][v["claim_index"]]
                for v in u["verdicts"]
                if v.get("supported") is False and v.get("claim_index", -1) < len(u["claims"])
            ]
            for claim in bad[:2]:
                print(f"    {u['id']}: {claim[:100]}")


if __name__ == "__main__":
    main()
