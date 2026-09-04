"""Cross-encoder re-ranking.

A bi-encoder embeds query and chunk separately, so it never compares them
directly. A cross-encoder feeds the pair through one model and scores the
interaction - far more accurate, but not precomputable. So it reorders a
shortlist it did not have to find, which means it can NEVER exceed the base
retriever's recall@candidates. That ceiling is the thing to watch.
"""

import re

from clausecompass.obs.trace import tracer
from sentence_transformers import CrossEncoder

from .base import Hit

# The golden questions are entity-qualified ("Under the Adams Golf Endorsement
# agreement, ...") so that RETRIEVAL can identify the document. For RE-RANKING
# that prefix is noise: only 21% of gold clauses contain the party name, so the
# cross-encoder sees a company in the query that is absent from the passage and
# penalises the correct clause. Document identification and clause relevance are
# different jobs and want different queries.
ENTITY_PREFIX = re.compile(r"^Under the .*? agreement,\s*", re.IGNORECASE)

# bge-reranker-base is stronger but ~1.1 GB; the MiniLM cross-encoder is ~90 MB
# and downloads reliably. Day 2 lost hours to Hugging Face resetting large-file
# transfers, so the fallback is deliberate.
PRIMARY = "BAAI/bge-reranker-base"
FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankingRetriever:
    name = "hybrid_rerank"

    def __init__(self, base, model_name=PRIMARY, device="cpu", candidates=50,
                 max_length=512, split_long=True, strip_entity=True):
        # device="cpu": CrossEncoder on MPS is flaky across sentence-transformers
        # versions, and 50 pairs on CPU is still well under a second.
        self.base = base
        self.candidates = candidates
        self.split_long = split_long
        self.strip_entity = strip_entity
        self.max_chars = max_length * 3          # rough chars-per-token estimate
        try:
            self.model = CrossEncoder(model_name, max_length=max_length, device=device)
            self.model_name = model_name
        except Exception as e:
            print(f"  {model_name} unavailable ({type(e).__name__}); "
                  f"falling back to {FALLBACK}")
            self.model = CrossEncoder(FALLBACK, max_length=max_length, device=device)
            self.model_name = FALLBACK

    def _score(self, query: str, hits: list[Hit]) -> list[float]:
        if not self.split_long:
            return [float(s) for s in
                    self.model.predict([(query, h.text) for h in hits], batch_size=16)]

        # Chunks longer than max_length are truncated, making their tails
        # invisible to the re-ranker. Our structural chunks reach 4,192 chars
        # against a ~1,536-char window - 2.7x - so splitting in HALF is not
        # enough: 20% of chunks still overflow after one split. Window the whole
        # chunk and keep the best-scoring window.
        pairs, owner = [], []
        for i, h in enumerate(hits):
            if len(h.text) > self.max_chars:
                for start in range(0, len(h.text), self.max_chars):
                    pairs.append((query, h.text[start:start + self.max_chars]))
                    owner.append(i)
            else:
                pairs.append((query, h.text))
                owner.append(i)
        raw = self.model.predict(pairs, batch_size=16)
        best = [float("-inf")] * len(hits)
        for score, i in zip(raw, owner):
            best[i] = max(best[i], float(score))
        return best

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        # Fetch at least k candidates, else re-ranking cannot return k.
        cands = self.base.search(query, max(self.candidates, k), doc_id)
        if not cands:
            return []
        with tracer.span("rerank", n_candidates=len(cands),
                         strip_entity=self.strip_entity) as sp:
            # retrieval keeps the entity qualifier; re-ranking drops it
            rq = ENTITY_PREFIX.sub("", query) if self.strip_entity else query
            scores = self._score(rq, cands)
            ranked = sorted(zip(scores, cands), key=lambda t: -t[0])
            # before/after ranks of the final top-5: the single most useful
            # debugging attribute in the whole trace.
            before = {h.chunk_id: i for i, h in enumerate(cands, 1)}
            sp["moves"] = [{"chunk_id": h.chunk_id, "was": before[h.chunk_id],
                            "now": i} for i, (_, h) in enumerate(ranked[:5], 1)]
        # Reorder only - never drop. Anything beyond the re-ranked window keeps
        # its original order so recall@50 is provably unchanged.
            head = [Hit(h.chunk_id, float(s), h.text, h.metadata) for s, h in ranked]
            return head[:k]
