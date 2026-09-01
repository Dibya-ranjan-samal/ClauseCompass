"""Retrieval configuration, chosen by the Day 4 ablation rather than by taste.

The winning configuration DIFFERS BY SCOPE, which is the day's main finding:

  corpus_wide  - plain dense wins (recall@5 0.363). BM25 raises the candidate
                 ceiling (0.585 -> 0.608) but ranks noisily, and the
                 cross-encoder cannot recover ordering from a pool that is only
                 60% correct. Nothing beat the Day 3 baseline on recall@5.
  doc_scoped   - hybrid + cross-encoder wins decisively (0.575 -> 0.762,
                 MRR 0.442 -> 0.657), but ONLY with the entity qualifier
                 stripped from the re-ranking query.

`strip_entity` is scope-dependent and must not be set globally: it is worth
+0.172 recall@5 doc-scoped and -0.173 corpus-wide. See notes/day4-explained.md.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalConfig:
    collection: str = "clauses"          # prefixed; the ablation kept the prefix
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_device: str = "mps"

    top_k_retrieve: int = 50
    top_k_answer: int = 5

    # fusion - rrf_k=10 beat the standard 60 on recall@5 (0.356 vs 0.338)
    rrf_k: int = 10
    bm25_candidates: int = 50            # fewer BM25 candidates measured WORSE

    # re-ranking - CPU because CrossEncoder on MPS is flaky, and it costs ~12s
    # p95 against 37 ms for dense. Only worth it doc-scoped.
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_device: str = "cpu"
    rerank_candidates: int = 50
    split_long_chunks: bool = True       # ~neutral on recall@5, +0.04 MRR
    strip_entity_for_rerank: bool = True  # doc-scoped only - see module docstring


CORPUS_WIDE = RetrievalConfig()
DOC_SCOPED = RetrievalConfig()
