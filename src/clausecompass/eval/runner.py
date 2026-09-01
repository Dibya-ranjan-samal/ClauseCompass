"""Day 3 - retrieval eval runner. Writes a timestamped JSON record per run.

Two configs, because the answer differs by product shape:

  corpus_wide  - search all 530 contracts. The realistic hard case; the question
                 names the contract but retrieval must find it among 18k chunks.
  doc_scoped   - filter to the question's doc_id first. The per-contract Q&A
                 product, and an upper bound on what re-ranking could reach.

Questions with no gold chunks (unanswerable, aggregation) are excluded from the
retrieval aggregates - every metric returns 1.0 for empty gold, so including
them would hand out free points. They are counted and reported separately.
"""

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from clausecompass.eval.metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
COLLECTION = "clauses"
METRICS = ["recall@5", "recall@10", "recall@50", "hit@5", "hit@10", "mrr", "ndcg@5"]


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run(config: str, top_k: int = 50, golden_path: str = "evals/golden/golden.jsonl"):
    golden = [json.loads(l) for l in open(golden_path)]
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    client = QdrantClient(url="http://localhost:6333")

    results, latencies = [], []
    for q in golden:
        qfilter = None
        if config == "doc_scoped" and q.get("doc_id"):
            qfilter = models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=q["doc_id"]))])

        t0 = time.perf_counter()
        qv = model.encode(QUERY_PREFIX + q["question"], normalize_embeddings=True)
        hits = client.query_points(COLLECTION, query=qv.tolist(), limit=top_k,
                                   with_payload=True, query_filter=qfilter).points
        latencies.append((time.perf_counter() - t0) * 1000)

        retrieved = [h.payload["chunk_id"] for h in hits]
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
        # near-miss questions record whether the trap clause outranked the answer
        if q["type"] == "near_miss" and q.get("trap_chunk_ids"):
            trap = set(q["trap_chunk_ids"])
            row["trap_in_top5"] = bool(trap & set(retrieved[:5]))
            row["trap_beat_gold"] = bool(
                trap & set(retrieved[:5])) and not (gold & set(retrieved[:5]))
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

    record = {"config": config, "git_sha": git_sha(), "top_k": top_k,
              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "aggregate": agg, "by_type": by_type, "per_question": results}

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-{config}.json"
    out.write_text(json.dumps(record, indent=2))

    print(f"\n=== {config} @ {git_sha()} (top_k={top_k}) ===")
    for k, v in agg.items():
        print(f"  {k:20} {v}")
    print("  by type:")
    for t, s in by_type.items():
        print(f"    {t:12} n={s['n']:<4} recall@5={s['recall@5']:<6} "
              f"recall@50={s['recall@50']:<6} mrr={s['mrr']}")
    traps = [r for r in results if "trap_in_top5" in r]
    if traps:
        print(f"  near-miss traps: {sum(r['trap_in_top5'] for r in traps)}/{len(traps)} "
              f"pulled the trap into top-5, "
              f"{sum(r['trap_beat_gold'] for r in traps)}/{len(traps)} beat the answer")
    print(f"saved -> {out}")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="corpus_wide",
                    choices=["corpus_wide", "doc_scoped", "both"])
    ap.add_argument("--top-k", type=int, default=50)
    a = ap.parse_args()
    for cfg in (["corpus_wide", "doc_scoped"] if a.config == "both" else [a.config]):
        run(cfg, a.top_k)
