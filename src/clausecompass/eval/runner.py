"""Day 3/4 - retrieval eval runner. One timestamped JSON record per run.

Takes any object implementing the Retriever protocol, so the ablation swaps
retrievers without touching the scoring code.

Two scopes, because Day 3 measured different bottlenecks in each:
  corpus_wide  - search all 18,280 chunks. The realistic hard case.
  doc_scoped   - filter to the question's doc_id first. Per-contract Q&A, and an
                 upper bound on what re-ranking alone could reach.

Questions with no gold chunks (unanswerable, aggregation) are excluded from the
aggregates - every metric returns 1.0 for empty gold, so including them would
hand out free points.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from clausecompass.eval.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

GOLDEN = "evals/golden/golden.jsonl"
METRICS = ["recall@5", "recall@10", "recall@50", "hit@5", "hit@10", "mrr", "ndcg@5"]


def load_golden(path: str = GOLDEN) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(retriever, config_name: str, scope: str = "corpus_wide", top_k: int = 50,
        golden_path: str = GOLDEN, quiet: bool = False) -> dict:
    golden = load_golden(golden_path)
    results, latencies = [], []

    for q in golden:
        doc_id = q.get("doc_id") if scope == "doc_scoped" else None
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], top_k, doc_id)
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = [h.chunk_id for h in hits]
        gold = set(q["gold_chunk_ids"])
        row = {
            "id": q["id"], "type": q["type"], "category": q.get("category"),
            "answerable": q["answerable"], "n_gold": len(gold),
            "recall@5": recall_at_k(gold, retrieved, 5),
            "recall@10": recall_at_k(gold, retrieved, 10),
            "recall@50": recall_at_k(gold, retrieved, 50),
            "hit@5": hit_at_k(gold, retrieved, 5),
            "hit@10": hit_at_k(gold, retrieved, 10),
            "mrr": mrr(gold, retrieved),
            "ndcg@5": ndcg_at_k(gold, retrieved, 5),
            "retrieved_top5": retrieved[:5],
        }
        if q["type"] == "near_miss" and q.get("trap_chunk_ids"):
            trap = set(q["trap_chunk_ids"])
            top5 = set(retrieved[:5])
            row["trap_in_top5"] = bool(trap & top5)
            row["trap_beat_gold"] = bool(trap & top5) and not (gold & top5)
        results.append(row)

    scored = [r for r in results if r["n_gold"] > 0]
    agg = {m: round(statistics.mean(r[m] for r in scored), 4) for m in METRICS}
    agg["n_scored"] = len(scored)
    agg["n_excluded_no_gold"] = len(results) - len(scored)
    agg["latency_p50_ms"] = round(statistics.median(latencies), 1)
    agg["latency_p95_ms"] = round(sorted(latencies)[int(0.95 * len(latencies)) - 1], 1)

    by_type = {}
    for t in sorted({r["type"] for r in scored}):
        sub = [r for r in scored if r["type"] == t]
        by_type[t] = {"n": len(sub),
                      "recall@5": round(statistics.mean(r["recall@5"] for r in sub), 3),
                      "recall@50": round(statistics.mean(r["recall@50"] for r in sub), 3),
                      "mrr": round(statistics.mean(r["mrr"] for r in sub), 3)}

    label = f"{config_name}-{scope}"
    record = {"config": config_name, "scope": scope, "label": label,
              "retriever": getattr(retriever, "name", "?"), "git_sha": git_sha(),
              "top_k": top_k, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "aggregate": agg, "by_type": by_type, "per_question": results}

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{label}.json"
    out.write_text(json.dumps(record, indent=2))

    if not quiet:
        print(f"\n=== {label} @ {git_sha()} (top_k={top_k}) ===")
        for k, v in agg.items():
            print(f"  {k:20} {v}")
        print("  by type:")
        for t, s in by_type.items():
            print(f"    {t:12} n={s['n']:<4} recall@5={s['recall@5']:<6} "
                  f"recall@50={s['recall@50']:<6} mrr={s['mrr']}")
        traps = [r for r in results if "trap_in_top5" in r]
        if traps:
            print(f"  near-miss traps: {sum(r['trap_in_top5'] for r in traps)}/{len(traps)} "
                  f"in top-5, {sum(r['trap_beat_gold'] for r in traps)}/{len(traps)} beat the answer")
        print(f"saved -> {out}")
    return record


def build(name: str, collection: str = "clauses", **kw):
    """Assemble a retriever by name. Imports are local so a dense-only run does
    not pay for loading the cross-encoder."""
    from clausecompass.retrieval.dense import DenseRetriever
    if name == "dense":
        return DenseRetriever(collection=collection)
    from clausecompass.retrieval.sparse import BM25Retriever
    if name == "bm25":
        return BM25Retriever()
    from clausecompass.retrieval.fusion import HybridRetriever
    hyb = HybridRetriever(DenseRetriever(collection=collection), BM25Retriever(),
                          rrf_k=kw.get("rrf_k", 60))
    if name == "hybrid":
        return hyb
    if name == "hybrid_rerank":
        from clausecompass.retrieval.rerank import RerankingRetriever
        return RerankingRetriever(hyb, candidates=kw.get("candidates", 50),
                                  split_long=kw.get("split_long", True),
                                  strip_entity=kw.get("strip_entity", True))
    raise SystemExit(f"unknown retriever {name!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retriever", default="dense",
                    choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    ap.add_argument("--scope", default="corpus_wide",
                    choices=["corpus_wide", "doc_scoped", "both"])
    ap.add_argument("--collection", default="clauses")
    ap.add_argument("--config", help="run label (defaults to the retriever name)")
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rrf-k", type=int, default=60)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--no-split-long", action="store_true")
    ap.add_argument("--no-strip-entity", action="store_true",
                    help="keep the 'Under the <party> ...' prefix in the rerank query")
    a = ap.parse_args()

    r = build(a.retriever, a.collection, rrf_k=a.rrf_k, candidates=a.candidates,
              split_long=not a.no_split_long,
              strip_entity=not a.no_strip_entity)
    for sc in (["corpus_wide", "doc_scoped"] if a.scope == "both" else [a.scope]):
        run(r, a.config or a.retriever, sc, a.top_k)
