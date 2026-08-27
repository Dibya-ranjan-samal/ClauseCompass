"""Task 5 - retrieve relevant clauses and answer with citations.

Uses a local Ollama model, so no API key is needed. Requires:
    ollama serve
    ollama pull qwen2.5:3b
"""

import re
import sys

import httpx
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "clauses"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen2.5:3b"
TOP_K = 5

# Ollama defaults num_ctx to 2048 tokens regardless of what the model supports
# (qwen2.5:3b declares 32768), and it TRUNCATES an over-long prompt silently -
# no error, no warning. Measured: a 20,316-char prompt (~5,079 tokens) had only
# 2,050 tokens evaluated, i.e. 60% of the retrieved excerpts were discarded
# before the model ever saw them. TOP_K=5 costs ~1,200 tokens so it fits inside
# the default, but with no margin. Set it explicitly.
NUM_CTX = 8192

# Required for BGE models: queries get this prefix, documents do not.
# Omitting it silently costs 5-10% retrieval quality.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Excerpts are labelled [1]..[k] rather than by chunk_id. Day 1 showed that
# chunk_id-shaped labels collide with the contracts' own section numbering:
# given a chunk containing the heading "27.15 Governing Law", the model emitted
# the citation [Governing Law::27] - an ID that exists nowhere in the corpus -
# and then blended text from two different contracts under it. Opaque integer
# labels have no counterpart in the contract text, so there is nothing to
# pattern-match onto. chunk_ids are restored after generation.
#
# Keep this instruction minimal. Adding prohibitions ("do not invent labels",
# "never put words inside brackets") measurably backfired on qwen2.5:3b - it
# named the forbidden thing instead, emitting [Governing Law] and [27.15].
# Over the 4 stock questions: minimal = 0/4 fabricated citations, the two
# variants with prohibitions = 1/4 each.
PROMPT = """Answer the question using ONLY the excerpts below.
Cite the excerpt label after every factual statement, like [3].
If the excerpts don't contain the answer, say "Not found in the provided excerpts."

<excerpts>
{context}
</excerpts>

Question: {question}"""

CITATION = re.compile(r"\[([^\]\n]{1,60})\]")


def restore_chunk_ids(answer: str, label_to_chunk: dict[str, str]) -> str:
    """Rewrite [1] -> [contract_09::8]. Unknown labels are left untouched so
    they stay visible to the caller rather than being silently dropped."""
    return CITATION.sub(
        lambda m: f"[{label_to_chunk[m.group(1)]}]"
        if m.group(1) in label_to_chunk
        else m.group(0),
        answer,
    )


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

    label_to_chunk = {str(i): h.payload["chunk_id"] for i, h in enumerate(hits, 1)}

    print("\nRetrieved:")
    for label, h in zip(label_to_chunk, hits):
        print(f"  [{label}] {h.score:.4f}  {h.payload['chunk_id']}")

    context = "\n\n".join(
        f"[{label}]\n{h.payload['text']}" for label, h in zip(label_to_chunk, hits)
    )

    resp = httpx.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "prompt": PROMPT.format(context=context, question=question),
            "stream": False,
            "options": {"temperature": 0.0, "num_ctx": NUM_CTX},
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    body = resp.json()
    answer = body["response"].strip()

    cited = set(CITATION.findall(answer))
    bogus = sorted(cited - set(label_to_chunk))

    print("\n--- Answer ---")
    print(restore_chunk_ids(answer, label_to_chunk))

    # Day 1 F4: citation compliance is inconsistent, so report it rather than
    # assuming it. Enforcement (retry, then refuse) is a Day 5 job.
    if bogus:
        print(f"\nWARNING: {len(bogus)} citation(s) not in the retrieved set, "
              f"left unmapped: {', '.join(f'[{b}]' for b in bogus)}", file=sys.stderr)
    if not cited & set(label_to_chunk):
        print("WARNING: answer contains no valid citation.", file=sys.stderr)

    # Guard against the silent truncation described at NUM_CTX above.
    used = body.get("prompt_eval_count")
    if used and used >= NUM_CTX * 0.9:
        print(f"WARNING: prompt used {used}/{NUM_CTX} context tokens - excerpts "
              f"may have been truncated. Raise NUM_CTX or lower TOP_K.",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
