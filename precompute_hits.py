"""Precompute golden-set retrieval once, so generation evals are pure LLM work.

The Day 4 winner (doc-scoped hybrid + cross-encoder) costs ~11.6 s p95 per
query. Running it inside every generation eval would mean ~29 minutes of
re-ranking per pass. The golden set is fixed, so retrieve once and write the
hits to disk. Cheaper than a cache layer and there is nothing to invalidate.
"""

import json
import sys
from pathlib import Path

from clausecompass.config import RetrievalConfig
from clausecompass.eval.runner import load_golden

OUT = Path("data/interim/golden_hits.json")
TOP_K = 6          # what the generator sees


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "rerank"
    cfg = RetrievalConfig()

    from clausecompass.retrieval.dense import DenseRetriever
    dense = DenseRetriever(collection=cfg.collection)
    if which == "dense":
        retriever = dense
    else:
        from clausecompass.retrieval.fusion import HybridRetriever
        from clausecompass.retrieval.rerank import RerankingRetriever
        from clausecompass.retrieval.sparse import BM25Retriever
        retriever = RerankingRetriever(
            HybridRetriever(dense, BM25Retriever(), rrf_k=cfg.rrf_k,
                            per_retriever_k=cfg.bm25_candidates),
            candidates=cfg.rerank_candidates, split_long=cfg.split_long_chunks,
            strip_entity=cfg.strip_entity_for_rerank)

    golden = load_golden()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    for i, q in enumerate(golden, 1):
        key = f"{which}:{q['id']}"
        if key in out:
            continue
        hits = retriever.search(q["question"], TOP_K, q.get("doc_id"))
        out[key] = [{"chunk_id": h.chunk_id, "text": h.text, "score": h.score}
                    for h in hits]
        if i % 10 == 0:
            OUT.write_text(json.dumps(out))      # checkpoint; the run is ~20 min
            print(f"  {i}/{len(golden)}", flush=True)
    OUT.write_text(json.dumps(out))
    n = sum(1 for k in out if k.startswith(f"{which}:"))
    empty = sum(1 for k, v in out.items() if k.startswith(f"{which}:") and not v)
    print(f"{which}: {n} questions cached, {empty} with zero hits -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
