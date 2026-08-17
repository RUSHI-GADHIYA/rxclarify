"""Evaluation harness.

Two modes, deliberately separate:

  --mode retrieval   (default)  no LLM calls at all. Free, instant, re-runnable.
  --mode generate               also calls the model. Costs money.

That split is the reason Phase 2 fits in a $3 budget. Retrieval ablations are
decided by Recall@k and MRR, which need no judge — so comparing four retriever
configurations costs nothing, and the model is only invoked when the question is
about generation quality.

Usage:
    python eval/harness.py --config dense
    python eval/harness.py --config all
    python eval/harness.py --config hybrid_rerank --mode generate --limit 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from metrics import (  # noqa: E402
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    rate,
    recall_at_k,
    reciprocal_rank,
)

from rxclarify.config import PRICING, get_settings  # noqa: E402
from rxclarify.db import connect  # noqa: E402

GOLD_PATH = Path(__file__).parent / "gold_set.jsonl"
RESULTS_DIR = Path(__file__).parent / "results"

CONFIGS = ("dense", "sparse", "hybrid", "hybrid_rerank")


def build_retriever(name: str, conn):
    """Every config implements the same `Retriever` protocol."""
    from rxclarify.retrieval.dense import DenseRetriever
    from rxclarify.retrieval.hybrid import HybridRetriever
    from rxclarify.retrieval.rerank import RerankingRetriever
    from rxclarify.retrieval.sparse import SparseRetriever

    if name == "dense":
        return DenseRetriever(conn)
    if name == "sparse":
        return SparseRetriever(conn)
    if name == "hybrid":
        return HybridRetriever(conn)
    if name == "hybrid_rerank":
        return RerankingRetriever(HybridRetriever(conn))
    raise ValueError(f"unknown config {name!r}; expected one of {CONFIGS}")


def load_gold(limit: int | None = None) -> list[dict]:
    with GOLD_PATH.open(encoding="utf-8") as fh:
        gold = [json.loads(line) for line in fh if line.strip()]
    return gold[:limit] if limit else gold


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    price = PRICING.get(get_settings().active_model)
    if not price:
        return 0.0
    per_in, per_out = price
    return input_tokens / 1e6 * per_in + output_tokens / 1e6 * per_out


def run(config: str, *, mode: str, limit: int | None, top_k: int) -> dict:
    gold = load_gold(limit)
    answerable = [g for g in gold if g["answerable"]]
    adversarial = [g for g in gold if not g["answerable"]]

    per_question: list[dict] = []

    chat_model = None
    if mode == "generate":
        from rxclarify.llm.factory import get_chat_model

        chat_model = get_chat_model()

    with connect() as conn:
        retriever = build_retriever(config, conn)

        for item in gold:
            started = time.perf_counter()
            record: dict = {"id": item["id"], "category": item["category"]}

            if mode == "generate":
                from rxclarify.generate.answer import answer_question
                from rxclarify.retrieval.langchain_retriever import ProtocolRetriever

                # ProtocolRetriever adapts any config to LangChain uniformly,
                # so every retriever goes through the identical answer path.
                result = answer_question(
                    item["question"],
                    retriever=ProtocolRetriever(retriever=retriever, k=top_k),
                    chat_model=chat_model,
                    top_k=top_k,
                )
                retrieved = result.chunks

                record.update(
                    refused=result.refused,
                    uncited=result.uncited,
                    invalid_markers=result.invalid_markers,
                    input_tokens=result.input_tokens or 0,
                    output_tokens=result.output_tokens or 0,
                    answer=result.text,
                    contexts=[c.text for c in result.chunks],
                    question=item["question"],
                )
            else:
                retrieved = retriever.retrieve(item["question"], top_k=top_k)

            record["latency_ms"] = (time.perf_counter() - started) * 1000
            record["retrieved_ids"] = [c.chunk_id for c in retrieved]
            per_question.append(record)

    by_id = {r["id"]: r for r in per_question}

    # --- retrieval metrics: answerable questions only -----------------------
    # Adversarial questions have no relevant chunk by construction, so scoring
    # them here would drag every average toward zero and mean nothing.
    retrieval: dict[str, float] = {}
    if answerable:
        recalls, rrs, ndcgs, precisions = [], [], [], []
        for item in answerable:
            got = by_id[item["id"]]["retrieved_ids"]
            want = item["supporting_chunk_ids"]
            recalls.append(recall_at_k(got, want, top_k))
            rrs.append(reciprocal_rank(got, want))
            ndcgs.append(ndcg_at_k(got, want, top_k))
            precisions.append(precision_at_k(got, want, top_k))
        retrieval = {
            f"recall@{top_k}": mean(recalls),
            "mrr": mean(rrs),
            f"ndcg@{top_k}": mean(ndcgs),
            f"precision@{top_k}": mean(precisions),
            # Sparse full-text returns only chunks that actually match the
            # query, which is often fewer than top_k. Without this, precision
            # looks anomalously high for a retriever that is in fact failing to
            # return anything at all on many questions.
            "mean_retrieved": mean([len(by_id[i["id"]]["retrieved_ids"]) for i in answerable]),
            "n": len(answerable),
        }

    summary: dict = {
        "config": config,
        "mode": mode,
        "top_k": top_k,
        "model": get_settings().active_model if mode == "generate" else None,
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "questions": len(gold),
        "retrieval": retrieval,
        "latency_ms": {
            "p50": percentile([r["latency_ms"] for r in per_question], 50),
            "p95": percentile([r["latency_ms"] for r in per_question], 95),
        },
    }

    # --- behaviour metrics: only meaningful once the model has answered -----
    if mode == "generate":
        refused_adversarial = sum(1 for g in adversarial if by_id[g["id"]]["refused"])
        refused_answerable = sum(1 for g in answerable if by_id[g["id"]]["refused"])
        hallucinated = sum(1 for r in per_question if r.get("invalid_markers"))
        uncited = sum(1 for r in per_question if r.get("uncited"))
        in_tok = sum(r.get("input_tokens", 0) for r in per_question)
        out_tok = sum(r.get("output_tokens", 0) for r in per_question)

        summary["behaviour"] = {
            "correct_refusal_rate": rate(refused_adversarial, len(adversarial)),
            "over_refusal_rate": rate(refused_answerable, len(answerable)),
            "hallucinated_citation_rate": rate(hallucinated, len(per_question)),
            "uncited_answer_rate": rate(uncited, len(per_question)),
        }
        summary["cost"] = {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "usd_total": round(estimate_cost(in_tok, out_tok), 5),
            "usd_per_question": round(estimate_cost(in_tok, out_tok) / max(len(gold), 1), 6),
        }

    summary["per_question"] = per_question
    return summary


def report(summary: dict) -> None:
    r = summary["retrieval"]
    print(f"\n=== {summary['config']} ({summary['mode']}, k={summary['top_k']}) ===")
    if r:
        k = summary["top_k"]
        print(f"  recall@{k}      {r[f'recall@{k}']:.3f}   ({r['n']} answerable questions)")
        print(f"  MRR            {r['mrr']:.3f}")
        print(f"  nDCG@{k}        {r[f'ndcg@{k}']:.3f}")
        # The 1/k ceiling only holds when k results come back; a retriever that
        # returns fewer scores higher on precision while doing worse overall.
        print(
            f"  precision@{k}   {r[f'precision@{k}']:.3f}   "
            f"(max {1 / k:.3f} when {k} returned; mean returned {r['mean_retrieved']:.1f})"
        )
    lat = summary["latency_ms"]
    print(f"  latency        p50 {lat['p50']:.0f}ms  p95 {lat['p95']:.0f}ms")

    if b := summary.get("behaviour"):
        print(f"  correct refusal   {b['correct_refusal_rate']:.3f}")
        print(f"  over-refusal      {b['over_refusal_rate']:.3f}")
        print(f"  hallucinated cite {b['hallucinated_citation_rate']:.3f}")
        print(f"  uncited answer    {b['uncited_answer_rate']:.3f}")
    if c := summary.get("cost"):
        print(
            f"  cost           ${c['usd_total']:.5f} total, ${c['usd_per_question']:.6f}/question"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="dense", help=f"{'|'.join(CONFIGS)}|all")
    parser.add_argument("--mode", default="retrieval", choices=("retrieval", "generate"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=6)
    args = parser.parse_args()

    configs = CONFIGS if args.config == "all" else (args.config,)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for config in configs:
        summary = run(config, mode=args.mode, limit=args.limit, top_k=args.top_k)
        report(summary)
        out = RESULTS_DIR / f"{config}_{args.mode}.json"
        out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"  -> {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
