"""Task 5 - retrieve relevant clauses and answer with citations.

Uses a local Ollama model, so no API key is needed. Requires:
    ollama serve
    ollama pull qwen2.5:3b
"""

import sys

import httpx
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "clauses"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"
TOP_K = 5

# Required for BGE models: queries get this prefix, documents do not.
# Omitting it silently costs 5-10% retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

PROMPT = """Answer the question using ONLY the excerpts below.
Cite the excerpt ID after every factual statement, like [contract_01::3].
If the excerpts don't contain the answer, say "Not found in the provided excerpts."

<excerpts>
{context}
</excerpts>

Question: {question}"""


def main() -> int:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        question = input("Ask a question about the contracts: ").strip()
    if not question:
        print("No question given.", file=sys.stderr)
        return 1

    model = SentenceTransformer(EMBED_MODEL, device="mps")
    client = QdrantClient(url="http://localhost:6333")

    qv = model.encode(QUERY_PREFIX + question, normalize_embeddings=True)
    # client.search is deprecated in qdrant-client >=1.19; query_points replaces it.
    hits = client.query_points(
        COLLECTION, query=qv.tolist(), limit=TOP_K, with_payload=True
    ).points

    if not hits:
        print("No chunks retrieved - is the 'clauses' collection populated?")
        return 1

    print("\nRetrieved:")
    for h in hits:
        print(f"  {h.score:.4f}  [{h.payload['chunk_id']}]")

    context = "\n\n".join(
        f"[{h.payload['chunk_id']}]\n{h.payload['text']}" for h in hits
    )

    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": PROMPT.format(context=context, question=question),
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=300.0,
    )
    resp.raise_for_status()

    print("\n--- Answer ---")
    print(resp.json()["response"].strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
