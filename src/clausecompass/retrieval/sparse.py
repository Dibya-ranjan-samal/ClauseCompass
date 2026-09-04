"""BM25 sparse retrieval - exact-term matching, where dense embeddings are weak.

Day 3 predicted this matters here: the golden questions name parties ("Under the
Adams Golf Endorsement agreement..."), and proper nouns are exactly what dense
embeddings blur and lexical matching nails.
"""

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from clausecompass.obs.trace import tracer

from .base import Hit

# Keep hyphenated legal terms intact: "non-compete" must not become two useless
# tokens, and "most-favoured-nation" must stay one.
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


class BM25Retriever:
    name = "bm25"
    CACHE = Path("data/interim/bm25.pkl")

    def __init__(self, chunks_path="data/interim/chunks_enriched.json"):
        if self.CACHE.exists():
            with self.CACHE.open("rb") as f:
                self.bm25, self.rows = pickle.load(f)
        else:
            chunks = json.loads(Path(chunks_path).read_text())
            # Cache only what retrieval needs. Keeping full payloads would put
            # several hundred MB in memory on an 8 GB machine for no benefit.
            self.rows = [(c["chunk_id"], c["doc_id"], c["text"]) for c in chunks]
            self.bm25 = BM25Okapi([tokenize(t) for _, _, t in self.rows])
            self.CACHE.parent.mkdir(parents=True, exist_ok=True)
            with self.CACHE.open("wb") as f:
                pickle.dump((self.bm25, self.rows), f)
        self.by_doc: dict[str, list[int]] = {}
        for i, (_, did, _) in enumerate(self.rows):
            self.by_doc.setdefault(did, []).append(i)

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        with tracer.span("retrieve.sparse", doc_id=doc_id, k=k):
            return self._search(query, k, doc_id)

    def _search(self, query, k, doc_id):
        scores = self.bm25.get_scores(tokenize(query))
        idx = np.asarray(self.by_doc.get(doc_id, [])) if doc_id else None
        if idx is not None:
            if not len(idx):
                return []
            order = idx[np.argsort(-scores[idx])][:k]
        else:
            order = np.argsort(-scores)[:k]
        return [Hit(self.rows[i][0], float(scores[i]), self.rows[i][2],
                    {"doc_id": self.rows[i][1]}) for i in order]
