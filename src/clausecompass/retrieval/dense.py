"""Dense retrieval: BGE-small embeddings over Qdrant."""

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from clausecompass.obs.trace import tracer

from .base import Hit

# BGE is asymmetric: queries get this prefix, documents do not. Omitting it
# costs 5-10% retrieval quality silently.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseRetriever:
    name = "dense"

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None,
                 url="http://localhost:6333", collection="clauses"):
        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        self.client = QdrantClient(url=url)
        self.collection = collection

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]:
        with tracer.span("retrieve.dense", doc_id=doc_id, k=k) as sp:
            return self._search(query, k, doc_id, sp)

    def _search(self, query, k, doc_id, sp):
        qv = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        qfilter = None
        if doc_id:
            qfilter = models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=doc_id))])
        # client.search is deprecated in qdrant-client >=1.19
        res = self.client.query_points(self.collection, query=qv.tolist(), limit=k,
                                       with_payload=True, query_filter=qfilter).points
        sp["n_hits"] = len(res)
        sp["top_score"] = float(res[0].score) if res else None
        return [Hit(r.payload["chunk_id"], float(r.score), r.payload["text"],
                    {"doc_id": r.payload["doc_id"]}) for r in res]
