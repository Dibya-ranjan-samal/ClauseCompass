"""Task 4 - embed the chunks with BGE-small and index them into Qdrant."""

import json
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

COLLECTION = "clauses"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE = 384
BATCH = 32
UPSERT_BATCH = 256

chunks = json.loads(Path("data/interim/chunks.json").read_text())
print(f"Loaded {len(chunks)} chunks")

model = SentenceTransformer(MODEL_NAME, device="mps")
client = QdrantClient(url="http://localhost:6333")

# recreate_collection is deprecated in qdrant-client >=1.19; do it explicitly.
if client.collection_exists(COLLECTION):
    client.delete_collection(COLLECTION)
client.create_collection(
    COLLECTION,
    vectors_config=models.VectorParams(
        size=VECTOR_SIZE, distance=models.Distance.COSINE
    ),
)

# NOTE: documents are embedded with no prefix. Only queries get the BGE
# "Represent this sentence..." prefix - see ask.py.
texts = [c["text"] for c in chunks]
vectors = model.encode(
    texts, normalize_embeddings=True, batch_size=BATCH, show_progress_bar=True
)

for start in range(0, len(chunks), UPSERT_BATCH):
    window = list(zip(chunks[start:start + UPSERT_BATCH],
                      vectors[start:start + UPSERT_BATCH]))
    client.upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=start + i, vector=v.tolist(), payload=c)
            for i, (c, v) in enumerate(window)
        ],
    )

count = client.count(COLLECTION, exact=True).count
print(f"\nIndexed {len(chunks)} chunks into Qdrant (collection '{COLLECTION}' "
      f"reports {count} points)")
