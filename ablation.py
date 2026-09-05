"""Day 4 - parameter sweep and the ablation table.

Sweep and table live in one file because they are the same concern: produce the
comparison that goes in the README. Run phases separately so a slow re-ranking
sweep does not block the cheap ones.

  uv run python ablation.py sweep-rrf     # cheap, no cross-encoder
  uv run python ablation.py sweep-cands   # slow, one cross-encoder run each
  uv run python ablation.py table
"""

import glob
import json
import sys

HEADLINE = ["dense_baseline", "bm25_only", "hybrid_rrf", "hybrid_rerank",
            "hybrid_rerank_fixed"]


def sweep_rrf():
    from clausecompass.eval.runner import build, run
    from clausecompass.retrieval.fusion import HybridRetriever
    dense = build("dense")
    from clausecompass.retrieval.sparse import BM25Retriever
    sparse = BM25Retriever()
    # rrf_k controls how steeply rank matters. Hybrid lost to dense on recall@5,
    # so the question is whether a smaller k lets dense's better-ordered top
    # hits outweigh BM25's noisier ones.
    for rrf_k in (10, 20, 60, 120):
        run(HybridRetriever(dense, sparse, rrf_k=rrf_k), f"rrf_k{rrf_k}",
            "corpus_wide", quiet=True)
        print(f"  rrf_k={rrf_k} done")
    # fewer BM25 candidates = less noise entering the fusion
    for per_k in (10, 25):
        run(HybridRetriever(dense, sparse, rrf_k=60, per_retriever_k=per_k),
            f"bm25cand{per_k}", "corpus_wide", quiet=True)
        print(f"  per_retriever_k={per_k} done")


def sweep_cands():
    from clausecompass.eval.runner import build, run
    from clausecompass.retrieval.fusion import HybridRetriever
    from clausecompass.retrieval.rerank import RerankingRetriever
    from clausecompass.retrieval.sparse import BM25Retriever
    hyb = HybridRetriever(build("dense"), BM25Retriever(), rrf_k=60)
    # More candidates raises the re-ranker's ceiling but costs latency linearly.
    # Looking for the plateau, not a round number.
    for cands in (20, 50, 100):
        run(RerankingRetriever(hyb, candidates=cands), f"rerank_cand{cands}",
            "corpus_wide", quiet=True)
        print(f"  candidates={cands} done")


def table():
    runs = {}
    for f in sorted(glob.glob("runs/*.json")):
        r = json.load(open(f))
        runs[r.get("label", r["config"])] = r          # keep the latest per label

    def emit(labels, title):
        print(f"\n### {title}\n")
        print(f"| {'Configuration':26} | R@5   | R@10  | R@50  | MRR   | nDCG@5 | p95 ms |")
        print(f"|{'-'*28}|-------|-------|-------|-------|--------|--------|")
        for lb in labels:
            a = runs[lb]["aggregate"]
            print(f"| {lb:26} | {a['recall@5']:.3f} | {a['recall@10']:.3f} | "
                  f"{a['recall@50']:.3f} | {a['mrr']:.3f} | {a['ndcg@5']:.3f}  | "
                  f"{a['latency_p95_ms']:6.0f} |")

    for scope in ("corpus_wide", "doc_scoped"):
        labels = [f"{c}-{scope}" for c in HEADLINE if f"{c}-{scope}" in runs]
        if labels:
            emit(labels, f"Ablation - {scope}")

    sweeps = sorted(lb for lb in runs
                    if lb.startswith(("rrf_k", "bm25cand", "rerank_cand",
                                      "dense_noprefix", "rerank_windowonly")))
    if sweeps:
        emit(sweeps, "Parameter sweep (corpus_wide)")

    print("\n### Per-type recall@5\n")
    types = ["single_hop", "near_miss", "multi_hop"]
    print(f"| {'Configuration':26} | " + " | ".join(f"{t:10}" for t in types) + " |")
    print(f"|{'-'*28}|" + "|".join(["-"*12]*len(types)) + "|")
    for scope in ("corpus_wide", "doc_scoped"):
        for c in HEADLINE:
            lb = f"{c}-{scope}"
            if lb not in runs:
                continue
            bt = runs[lb]["by_type"]
            cells = " | ".join(f"{bt[t]['recall@5']:<10.3f}" if t in bt else f"{'-':<10}"
                               for t in types)
            print(f"| {lb:26} | {cells} |")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else "table"
    {"sweep-rrf": sweep_rrf, "sweep-cands": sweep_cands, "table": table}[phase]()
