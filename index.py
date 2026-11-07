"""Task 6 - embed the enriched chunks and index them into Qdrant with filters.

Two changes over Day 1:

1. CONTEXTUAL PREFIX. Chunking strips the surrounding context a clause needs to
   be interpretable, so each chunk is embedded with a short header naming its
   contract type and section. "Termination for convenience" is ambiguous on its
   own; "Sponsorship agreement | Section 12 TERMINATION" is not.

2. PAYLOAD INDEXES. Filtered vector search needs an index on the filtered field
   or Qdrant falls back to a full scan.

The prefix is included in the stored `embed_text` so the exact string that
produced each vector is auditable after the fact.
"""

import argparse
import json
from pathlib import Path

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

ap = argparse.ArgumentParser()
ap.add_argument("--collection", default="clauses")
ap.add_argument("--no-prefix", action="store_true",
                help="embed chunk text only, for the Day 4 prefix ablation")
ap.add_argument("--docs-from", help="index only chunks of doc_ids in this golden .jsonl "
                "(CI: doc-scoped results depend only on that document's chunks)")
ARGS = ap.parse_args()

COLLECTION = ARGS.collection
MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
VECTOR_SIZE = 384
BATCH = 32
UPSERT_BATCH = 256
INDEXED_FIELDS = ("contract_type", "governing_law", "source", "doc_id",
                  "chunking_strategy", "section_heading")


def embed_text(c: dict) -> str:
    """Contextual prefix. Kept short: it competes with the clause for the
    model's 512-token budget.

    --no-prefix embeds the bare clause, which is the Day 4 ablation arm: Day 2
    measured that the prefix RAISES mean pairwise chunk similarity by 0.022
    (every chunk gains the same boilerplate header), but whether that costs
    retrieval quality was never tested.
    """
    if ARGS.no_prefix:
        return c["text"]
    bits = [f"Contract type: {c.get('contract_type') or 'Unknown'}"]
    if c.get("section_heading"):
        num = c.get("section_number")
        bits.append(f"Section {num} {c['section_heading']}" if num
                    else str(c["section_heading"]))
    if c.get("governing_law"):
        bits.append(f"Governing law: {c['governing_law']}")
    return " | ".join(bits) + "\n\n" + c["text"]


def main() -> int:
    src = Path("data/interim/chunks_enriched.json")
    if not src.exists():
        print("missing chunks_enriched.json - run extract_metadata.py first")
        return 1
    chunks = json.loads(src.read_text())
    if ARGS.docs_from:
        keep = {json.loads(line)["doc_id"] for line in open(ARGS.docs_from)}
        chunks = [c for c in chunks if c["doc_id"] in keep]
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    client = QdrantClient(url="http://localhost:6333")

    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(
            size=VECTOR_SIZE, distance=models.Distance.COSINE
        ),
    )

    texts = [embed_text(c) for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True,
                           batch_size=BATCH, show_progress_bar=True)

    for start in range(0, len(chunks), UPSERT_BATCH):
        window = list(zip(chunks[start:start + UPSERT_BATCH],
                          vectors[start:start + UPSERT_BATCH],
                          texts[start:start + UPSERT_BATCH]))
        client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(id=start + i, vector=v.tolist(),
                                   payload={**c, "embed_text": t})
                for i, (c, v, t) in enumerate(window)
            ],
        )

    for field in INDEXED_FIELDS:
        client.create_payload_index(
            COLLECTION, field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"payload indexes: {', '.join(INDEXED_FIELDS)}")

    count = client.count(COLLECTION, exact=True).count
    print(f"\nIndexed {len(chunks)} chunks into Qdrant "
          f"(collection '{COLLECTION}' reports {count} points)")
    if count != len(chunks):
        print("WARNING: point count does not match chunk count")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
