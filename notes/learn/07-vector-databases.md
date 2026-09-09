# 07 — Vector Databases, ANN & Payload Filters

---

## 1. What it is

A **vector database** stores embeddings alongside arbitrary metadata and answers
nearest-neighbour queries efficiently, with optional filtering on that metadata.

This project uses **Qdrant**, running as a Docker container, holding **18,280
points**.

## 2. Why it exists — and the honest answer to "why not NumPy?"

Module 00 showed the whole of dense retrieval as one line:

```python
scores = E @ q          # (18280, 384) @ (384,) -> 18,280 cosines
```

That is ~7 million multiply-adds — **a few milliseconds**. So at this scale a
vector database buys you *nothing* for the search itself, and you should say so
plainly in an interview rather than pretending otherwise.

What it does buy:

| Capability | Why NumPy alone doesn't give it |
|---|---|
| **Persistence** | your array dies with the process; re-embedding 18,280 chunks costs 5 minutes |
| **Payload storage** | retrieval returns the *text*, not just an index into an array you must keep in sync |
| **Metadata filtering** | "search only within `doc_id = X`" — see §5, this turned out decisive |
| **Sub-linear search** | at 10M vectors, brute force is no longer milliseconds |
| **Concurrent access** | API and eval runner querying the same index safely |
| **Incremental updates** | insert one contract without rebuilding everything |

**The order matters.** For this project, the top three are the real reasons; the
fourth is the one people quote and the one that mattered least.

## 3. The data model

| Term | Meaning | In this project |
|---|---|---|
| **Collection** | a table of vectors sharing dimension and metric | `clauses` |
| **Point** | one row: `id`, `vector`, `payload` | one chunk |
| **Payload** | arbitrary JSON attached to the vector | `chunk_id`, `doc_id`, `text`, `section_heading`, `contract_type`, `governing_law` |
| **VectorParams** | declares `size` and `distance` | `size=384, distance=COSINE` |
| **Payload index** | a secondary index on a payload field | 6 of them |

```python
client.create_collection(
    "clauses",
    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
)
```

`size=384` must match your encoder's output, and `distance=COSINE` must match how
you normalise. **A mismatch here is silent** — Qdrant will happily compute dot
products on unnormalised vectors and call them cosine.

### Why store the full text in the payload

```python
models.PointStruct(id=i, vector=v.tolist(), payload={**chunk, "embed_text": t})
```

So a search returns the text in one round trip. The alternative — store an id,
then look the text up in `chunks.json` — introduces a second source of truth that
can drift out of sync with the index. Storing the text is a few tens of MB and
removes an entire class of bug.

Note it also stores `embed_text`: the *exact string that produced the vector*.
That's an audit trail. When you later ask "why did this chunk match?", you can
see precisely what was embedded, including the contextual prefix (module 06).

## 4. ANN and HNSW — what "approximate" means

Exact nearest-neighbour search is O(n) per query. At 10 million vectors that's
too slow, so production systems use **approximate** nearest neighbour: accept a
small chance of missing a true neighbour in exchange for large speedups.

**HNSW** (Hierarchical Navigable Small World) is the dominant algorithm and the
one Qdrant uses. The intuition:

```
Layer 2 (sparse):   A -------------------- F
                     \                    /
Layer 1:            A ---- C ---- D ---- F
                     \     |      |     /
Layer 0 (all):      A - B - C - D - E - F - G - H ...
```

- Vectors are nodes in a graph; edges connect near neighbours.
- Multiple layers: upper layers are sparse with long edges, layer 0 has everything.
- A search **starts at the top**, greedily walks toward the query, then **descends**
  a layer and refines — like a skip list in high dimensions.

Long edges up top cover distance fast; dense lower layers give precision. Search
touches O(log n) nodes instead of n.

**The tradeoff you must be able to name:** recall (fraction of true neighbours
found) versus latency, controlled by `ef_search` — how many candidates to keep
while walking. Larger `ef` means better recall, slower search. Build-time knobs
`m` (edges per node) and `ef_construct` trade index size and build time for
quality.

**Why it barely matters here:** with 18,280 points, exact search is already fast,
so the approximation buys little and costs a little recall. Worth knowing for
interviews and for scale; not worth tuning in this project.

## 5. Payload filters — the part that actually mattered

Metadata filtering turned out to be the most consequential feature, and it is
easy to underrate.

```python
qfilter = models.Filter(must=[models.FieldCondition(
    key="doc_id", match=models.MatchValue(value=doc_id))])

hits = client.query_points(collection, query=qv, limit=k,
                           query_filter=qfilter).points
```

### Why it needs an index

```python
for field in ["contract_type", "governing_law", "source", "doc_id",
              "chunking_strategy", "section_heading"]:
    client.create_payload_index("clauses", field_name=field,
                                field_schema=models.PayloadSchemaType.KEYWORD)
```

Without a payload index, filtering degenerates to scanning payloads. With one,
Qdrant can restrict the ANN walk to matching points. **Pre-filtering** (restrict
then search) versus **post-filtering** (search then discard) is the distinction:
post-filtering can return fewer than `k` results, or none, because everything in
the top-k was filtered out. Pre-filtering is what you want, and the index is what
enables it.

### The measured payoff

This is the number that reframed the project. Same generic question — *"Which
state's or country's law governs this agreement?"* — asked of 30 different
contracts:

| Retrieval | hit@5 | hit@50 |
|---|---|---|
| **Unscoped** over all 18,280 chunks | **0 / 30** | 2 / 30 |
| **Doc-scoped** via payload filter | **27 / 30** | 30 / 30 |

Zero versus twenty-seven, from adding one filter.

The reason is not that retrieval is bad. That query string is **byte-identical**
for all 530 contracts; ~68 documents contain a chunk literally headed "Governing
Law", all equally good matches. Asking dense retrieval to pick contract #217's
clause is asking it to guess. **The filter supplies the information the query
lacks.**

This is why `doc_id` is a parameter of the `Retriever` protocol rather than a
separate code path:

```python
class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]: ...
```

## 6. Verification instead of assumption

```python
count = client.count(COLLECTION, exact=True).count       # 18280
print(f"Indexed {len(chunks)} chunks ({count} points)")
```

Do **not** print `len(chunks)` and call it "indexed". Ask the database what it
actually holds. Those two numbers disagreeing is exactly the bug you want caught
— a failed upsert batch, a duplicate-id overwrite, a dimension rejection.

This is module 05's accounting-check pattern applied to indexing.

## 7. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| `size` mismatch | upsert rejected, or garbage scores | assert `len(vector) == declared size` | match encoder output |
| Metric mismatch (dot vs cosine) | plausible but wrong ranking | recompute one score by hand | normalise + declare COSINE |
| Duplicate point ids | silent overwrites; count < chunks | compare `client.count()` to `len(chunks)` | unique ids |
| No payload index | filtered search slow, or post-filtered | time a filtered vs unfiltered query | `create_payload_index` |
| Post-filtering | fewer than `k` results returned | check `len(hits) < k` on filtered queries | pre-filter with an index |
| Stale index after re-chunking | `chunk_id`s in results don't exist | validate returned ids against the store | re-index, or version collections |
| `recreate_collection` deprecated | warnings, future breakage | read warnings | `delete_collection` + `create_collection` |
| Volume not persisted | index empty after restart | mount a volume | `./data/qdrant:/qdrant/storage` |

## 8. Compared with

| | NumPy array | Qdrant / vector DB |
|---|---|---|
| Search at 18k vectors | ~ms (exact) | ~ms (approximate) |
| Search at 10M | seconds | ~ms |
| Persistence | none | disk-backed |
| Metadata + filtering | you build it | native, indexed |
| Concurrency | your problem | handled |
| Operational cost | zero | a container to run |

| | Pre-filter | Post-filter |
|---|---|---|
| Order | restrict candidates, then search | search, then drop non-matching |
| Returns k? | yes | often fewer |
| Needs payload index | yes | no |
| Cost | cheap with an index | wasted work on discarded results |

**Honest positioning:** for 18,280 vectors you do not need a vector database for
speed. You want it for persistence, payload filtering, and the operational
properties. Saying that plainly reads as stronger than claiming you needed ANN.

## 9. Interview angle

**"Why Qdrant and not NumPy?"** — Lead with the concession: at 18,280 vectors
brute force is a few milliseconds, so I did not need ANN. I needed persistence
(re-embedding costs 5 minutes), payload storage so retrieval returns the text
without a second source of truth, and indexed metadata filtering — which turned
out to be the single highest-impact feature, taking hit@5 from 0/30 to 27/30 on
document-scoped questions.

**"Explain HNSW."** — A multi-layer proximity graph. Upper layers are sparse with
long edges to traverse distance quickly; layer 0 contains every point. Search
descends greedily from the top, refining each layer. O(log n) nodes touched
instead of n. The knob is `ef_search`, trading recall against latency, and the
"approximate" means you may miss true neighbours — which is why you measure
recall rather than assume it.

**"What's the difference between pre- and post-filtering, and why care?"** —
Pre-filtering restricts the candidate set before the ANN walk and needs a payload
index; post-filtering searches then discards, so it can return fewer than k
results — and in the worst case zero, when the entire top-k belonged to other
documents. For per-document Q&A that's the difference between working and not.

## 10. Test yourself

**Q1.** You declare `VectorParams(size=384, distance=DOT)` but forget to
normalise your embeddings. What happens?

<details><summary>Answer</summary>

Everything runs. Qdrant computes raw dot products, which for unnormalised vectors
equals `|a||b|cos(a,b)` — so scores are inflated by vector magnitude. Since
magnitude in text embeddings correlates with passage length and model confidence,
**longer chunks are systematically favoured** regardless of relevance.

The ranking is wrong in a plausible-looking way: no error, scores in a reasonable
range, and the bias is toward verbose chunks that often *do* contain more
material — so it can even look defensible. Detect it by recomputing one cosine by
hand and comparing to the returned score.

</details>

**Q2.** Your generic question scored hit@5 = 0/30 unscoped and 27/30 doc-scoped.
Is this a retrieval failure? Explain precisely.

<details><summary>Answer</summary>

No — it is an **information failure in the query**. The question string is
identical for all 530 contracts, and roughly 68 of them contain a chunk headed
"Governing Law". Nothing in the query distinguishes contract #217's clause from
contract #4's, so no retriever, however good, can rank the right one first except
by luck. Expected hit@5 is roughly `5/68`.

The filter doesn't improve retrieval; it supplies the missing information — which
document. The general lesson: before tuning a retriever, check that the query
*contains enough information to identify the answer*. This is what motivated
entity-qualified questions in module 12.

</details>

**Q3.** Why store the chunk's full text in the payload rather than looking it up
by id afterwards?

<details><summary>Answer</summary>

Two reasons. **Operationally**, it returns the text in one round trip instead of
requiring a second lookup structure to be loaded and kept in memory.

**More importantly**, it removes a second source of truth. If the text lives in
`chunks.json` and the vectors live in Qdrant, the two can drift — re-chunk
without re-indexing and your `chunk_id`s now resolve to different text than was
embedded, silently. Storing the text with the vector makes the index
self-describing. The project goes further and stores `embed_text` too, so you can
see the exact string that produced each vector.

</details>

**Q4.** `client.count()` returns 18,275 but you upserted 18,280 chunks. Give
three plausible causes.

<details><summary>Answer</summary>

1. **Duplicate point ids** — five chunks were assigned ids that collided with
   earlier ones, so the later upserts overwrote rather than inserted. This is the
   most likely cause when ids are derived from content or truncated strings (the
   project hit exactly this class of bug in `ingest_cuad.py`, where truncating
   doc_ids collapsed two distinct contracts).
2. **A failed batch** — one `upsert` call errored or was partially applied and the
   exception was swallowed.
3. **Dimension rejection** — five vectors had the wrong length and were rejected
   individually.

The reason to check at all: all three are silent from the caller's perspective if
you print `len(chunks)` instead of querying the database.

</details>

**Q5.** Harder. At what corpus size would ANN start earning its keep here, and
what would you measure to decide?

<details><summary>Answer</summary>

Brute force is O(n·d). At 18,280 × 384 it's ~7M FLOPs — sub-millisecond to a few
ms. Exact search stays acceptable into the low millions of vectors on a single
machine, especially with SIMD/BLAS: at 1M × 384 that's ~400M FLOPs, tens of
milliseconds — still fine for many applications.

ANN earns its keep when **p95 exact-search latency exceeds your budget**, which
depends on your budget, not a universal threshold. So the thing to measure is not
corpus size but the pair: (exact search p95, your latency SLO). And crucially,
measure **ANN recall** against exact search on your own data before switching —
`ef_search` too low silently drops true neighbours, which is another instance of
the silent-quality-loss pattern.

Secondary consideration that often dominates: **memory**. 1M × 384 × 4 bytes ≈
1.5 GB just for vectors, before payloads and graph edges. On the 8 GB machine
this project runs on, memory becomes the binding constraint well before search
latency does.

</details>

---

**Previous:** [06 — Dense retrieval](06-dense-retrieval.md) ·
**Next:** [08 — BM25](08-bm25.md)
