"""Reciprocal Rank Fusion of several ranked lists."""

from clausecompass.obs.trace import tracer

from .base import Hit


def rrf_fuse(rankings: list[list[Hit]], k: int = 60, top_n: int = 50) -> list[Hit]:
    """Merge ranked lists by RANK, not score.

    Scores from different retrievers are not comparable - cosine 0.81 and BM25
    14.2 live on different scales, so averaging them is meaningless. Ranks are
    comparable, so each list contributes 1/(k+rank). Lower k weights the top of
    each list more heavily.
    """
    scores: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for lst in rankings:
        for rank, hit in enumerate(lst, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit.chunk_id, hit)
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [Hit(cid, s, by_id[cid].text, by_id[cid].metadata) for cid, s in ordered]


class HybridRetriever:
    name = "hybrid_rrf"

    def __init__(self, dense, sparse, rrf_k=60, per_retriever_k=50):
        self.dense, self.sparse = dense, sparse
        self.rrf_k, self.per_k = rrf_k, per_retriever_k

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        # Asymmetric on purpose. Dense must fetch at least k or the fused list
        # cannot fill k slots, so it gets max(k, per_k). BM25 gets per_k exactly,
        # because per_retriever_k exists to limit how much sparse NOISE enters
        # the fusion - clamping it to k, as an earlier version did, silently
        # disabled the knob (every per_k value scored identically to 4 decimals).
        d = self.dense.search(query, max(self.per_k, k), doc_id)
        s = self.sparse.search(query, self.per_k, doc_id)
        with tracer.span("fuse.rrf", rrf_k=self.rrf_k) as sp:
            out = rrf_fuse([d, s], k=self.rrf_k, top_n=k)
            sp["n_dense"], sp["n_sparse"], sp["n_fused"] = len(d), len(s), len(out)
            return out
