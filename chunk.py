"""Task 3 - naive character-window chunking.

Deliberately naive: Day 3 replaces this with structural chunking on section
numbers. Keep this version as the baseline to beat.
"""

import json
from pathlib import Path

docs = json.loads(Path("data/interim/documents.json").read_text())

CHUNK_SIZE = 1000
OVERLAP = 150


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


all_chunks = []
per_doc = {}
for doc in docs:
    pieces = chunk_text(doc["text"], CHUNK_SIZE, OVERLAP)
    per_doc[doc["doc_id"]] = len(pieces)
    for i, piece in enumerate(pieces):
        all_chunks.append({
            "chunk_id": f"{doc['doc_id']}::{i}",
            "doc_id": doc["doc_id"],
            "text": piece,
        })

Path("data/interim/chunks.json").write_text(json.dumps(all_chunks, indent=2))
print(f"Created {len(all_chunks)} chunks from {len(docs)} documents")
counts = sorted(per_doc.values())
print(f"chunks/doc: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")
thin = [d for d, c in per_doc.items() if c < 10]
if thin:
    print(f"WARNING: thin docs (<10 chunks): {', '.join(thin)}")
