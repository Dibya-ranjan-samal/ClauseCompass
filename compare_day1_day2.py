"""Task 7 - qualitative Day 1 vs Day 2 comparison.

This is NOT a quality measurement. There is no eval harness until Day 3, so the
only honest claims here are structural ones that can be checked directly:

  - does the retrieved chunk begin at a clause boundary, or mid-sentence?
  - does it carry a section number and heading?
  - how many distinct documents do the top-5 span?

Answer quality is deliberately not scored. Resisting that is the point of the
task: any "it looks better" claim today would be guessing.
"""

import json
import re
import sys

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

QUESTIONS = [
    "What is the governing law of this agreement?",
    "Is there an indemnification clause?",
    "What are the termination provisions?",
    "Does the agreement include a limitation of liability?",
]
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# A clause-boundary start looks like a section number, a capitalised heading, or
# an ALL-CAPS heading. A mid-sentence start begins lowercase or with punctuation.
CLEAN_START = re.compile(r"^\s*(?:\d+(?:\.\d+)*\s*[.\)]?\s*)?(?:[A-Z][A-Z ]{3,}|[A-Z][a-z])")


def main() -> int:
    model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="mps")
    client = QdrantClient(url="http://localhost:6333")

    totals = {"clean": 0, "n": 0, "headed": 0, "preamble": 0}
    rows = []
    for q in QUESTIONS:
        qv = model.encode(QUERY_PREFIX + q, normalize_embeddings=True)
        hits = client.query_points("clauses", query=qv.tolist(), limit=5,
                                   with_payload=True).points
        docs = {h.payload["doc_id"] for h in hits}
        print(f"\n=== {q}")
        print(f"    top-5 spans {len(docs)} distinct documents")
        for h in hits:
            p = h.payload
            clean = bool(CLEAN_START.match(p["text"]))
            totals["clean"] += clean
            totals["n"] += 1
            totals["headed"] += bool(p.get("section_heading"))
            totals["preamble"] += bool(p.get("is_preamble"))
            head = p.get("section_heading") or "-"
            print(f"    {h.score:.4f} {'OK ' if clean else 'MID'} "
                  f"[{p.get('section_number') or '-':>6}] {head[:26]:<28} "
                  f"{p['source']:<6} {p['text'][:58].replace(chr(10),' ')}")
        rows.append({"question": q, "distinct_docs": len(docs),
                     "chunk_ids": [h.payload["chunk_id"] for h in hits]})

    n = totals["n"]
    print(f"\n=== structural summary over {n} retrieved chunks ===")
    print(f"  start at a clause boundary : {totals['clean']}/{n} "
          f"({100*totals['clean']//n}%)")
    print(f"  carry a section heading    : {totals['headed']}/{n} "
          f"({100*totals['headed']//n}%)")
    print(f"  are preamble chunks        : {totals['preamble']}/{n}")
    json.dump(rows, open("data/interim/day2_retrieval_sample.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
