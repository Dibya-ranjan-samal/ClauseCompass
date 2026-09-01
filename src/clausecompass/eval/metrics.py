"""Retrieval metrics. Deterministic and pure - unit-tested in tests/test_metrics.py.

All four take the gold chunk ids and the ranked list of retrieved chunk ids.

CAUTION: every metric returns 1.0 when `gold` is empty, meaning "nothing was
expected, so nothing was missed". Questions with no gold chunks (unanswerable,
aggregation) must therefore be EXCLUDED from retrieval aggregates - otherwise
they contribute free 1.0s and inflate the score. runner.py filters on
`gold_chunk_ids` for exactly this reason.
"""

import math


def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of gold chunks appearing in the top k."""
    if not gold:
        return 1.0
    return len(gold & set(retrieved[:k])) / len(gold)


def hit_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """1.0 if ANY gold chunk is in the top k. Usually what a user feels."""
    if not gold:
        return 1.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def mrr(gold: set[str], retrieved: list[str]) -> float:
    """Reciprocal rank of the first gold hit; 0.0 if none was retrieved."""
    if not gold:
        return 1.0
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Binary-relevance nDCG: rewards ranking gold chunks nearer the top."""
    if not gold:
        return 1.0
    dcg = sum(1.0 / math.log2(i + 1)
              for i, cid in enumerate(retrieved[:k], 1) if cid in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
