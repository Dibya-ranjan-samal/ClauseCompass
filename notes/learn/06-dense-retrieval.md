# 06 — Dense Retrieval & the Query-Prefix Asymmetry

---

## 1. What it is

**Dense retrieval** is search by embedding similarity: embed every chunk once,
embed the query at request time, return the chunks whose vectors are closest.

"Dense" because the vector is fully populated — all 384 dimensions carry values.
Contrast module 08's **sparse** vectors, which have one dimension per vocabulary
term and are almost all zero.

## 2. Why it exists

Keyword search fails when the question and the answer use different words:

| Query | Clause it should find | Shared words |
|---|---|---|
| "how do we end this contract early" | "Termination for Convenience. Either party may terminate..." | *none* |
| "who pays if we get sued" | "Supplier shall indemnify and hold harmless..." | *none* |

Both should match. No amount of stemming gets you there. Dense retrieval works
because embedding similarity approximates *semantic* similarity (module 00).

## 3. The mechanism, end to end

### Indexing (once)

```
for each chunk:
    tokens  = tokenise(chunk_text)             # ~250 tokens for a 1000-char chunk
    vectors = transformer(tokens)              # one 384-d vector per token
    v       = pool(vectors)                    # CLS pooling -> single 384-d vector
    v       = v / |v|                          # unit length
    store(v, payload={chunk_id, doc_id, text, ...})
```

### Querying (per request)

```
qv     = normalise(pool(transformer(tokenise(PREFIX + query))))
scores = all stored vectors · qv               # cosine, because both are unit length
return top-k by score
```

The asymmetry hiding in those two blocks — `PREFIX` appears in one and not the
other — is the subject of §4 and is the most commonly botched detail in dense
retrieval.

## 4. The query-prefix asymmetry

```python
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# queries:   model.encode(QUERY_PREFIX + question)   <- prefixed
# documents: model.encode(chunk_text)                <- NOT prefixed
```

### Why this exists at all

BGE is trained with **contrastive learning** on (query, positive passage,
negative passages) triples. The objective pulls a query's vector toward its
correct passage and pushes it away from wrong ones — typically with InfoNCE:

```
              exp(sim(q, p⁺) / τ)
L = −log ─────────────────────────────
          Σⱼ exp(sim(q, pⱼ) / τ)
```

- `sim(q,p)` — cosine similarity between query and passage vectors
- `p⁺` — the correct passage; `pⱼ` ranges over the positive and the negatives
- `τ` — temperature, sharpening or softening the distribution

The key point: **queries and passages are different kinds of text.** A query is
short, interrogative, often a fragment. A passage is long, declarative, prose. If
you embed both with the same unmodified encoder, you ask one function to place
two distinct distributions in a shared space.

BGE's fix is to prepend an instruction to queries during training, so the model
learns "when you see this prefix, embed into the space where matching *passages*
live." The prefix is a **learned mode switch**.

### Why getting it wrong is so dangerous

Consider the ways to be wrong:

| Mistake | Effect |
|---|---|
| No prefix on queries | queries land in "passage space"; mild systematic mismatch |
| Prefix on documents too | you've undone the asymmetry the model was trained with |
| Prefix on documents only | actively inverted |

Every one of them **runs perfectly.** No exception, no warning. You lose roughly
5–10% retrieval quality and have no signal that anything is wrong. It is the same
class as module 02's offset drift and module 00's silent truncation:

> **Silent quality loss is worse than a crash.** A crash costs an hour. Silent
> loss costs you the credibility of every number downstream.

The only defence is an eval harness — which is the argument for building module
11 before optimising anything.

## 5. The embedding space is compressed — and it bounds what scores can tell you

Module 00 established the corpus similarity floor. Restating it because dense
retrieval is where it *bites*:

| Measurement (44,850 random pairs) | Cosine |
|---|---|
| Random chunk pair, mean | 0.6430 |
| Random pair, p99 | **0.7787** |
| Best relevant chunk, real query | **0.7670** |
| Nonsense query, top-1 | 0.5779 |

The relevant answer scores **below** the 99th percentile of random pairs. So on
this corpus:

- ✅ absolute score **can** detect an off-topic query (0.578 vs 0.767)
- ❌ absolute score **cannot** rank clause relevance or drive abstention

That second line is why plain dense retrieval alone plateaus, and why module 10
exists.

## 6. The contextual-prefix experiment — a properly ablated negative-ish result

Day 2 added metadata before each chunk at embed time:

```python
def embed_text(c):
    bits = [f"Contract type: {c.get('contract_type') or 'Unknown'}"]
    if c.get("section_heading"):
        bits.append(f"Section {c['section_number']} {c['section_heading']}")
    if c.get("governing_law"):
        bits.append(f"Governing law: {c['governing_law']}")
    return " | ".join(bits) + "\n\n" + c["text"]
```

Intuition: chunking strips context, so restore it. Plausible. **Never measured**
until Day 4, and then measured two ways.

### Effect on the similarity floor

Same 300 chunks, same model, only the embedded string differs:

| Embedded string | mean | p99 |
|---|---|---|
| Day 1 naive chunks, no prefix | 0.6430 | 0.7787 |
| Clause chunks, **text only** | **0.6180** | **0.7454** |
| Clause chunks, **with prefix** | 0.6403 | 0.7517 |

The prefix **raises** mean pairwise similarity by 0.022. Obvious in hindsight:
every chunk now begins with the same boilerplate shape, so all chunks look *more*
alike. The improvement over Day 1 came from **clause chunking**, and the prefix
partially undid it.

### Effect on actual retrieval — which is what matters

A separate collection was built without prefixes and evaluated:

| Scope | with prefix | without | verdict |
|---|---|---|---|
| corpus-wide recall@5 | **0.3632** | 0.3443 | prefix **helps** +0.019 |
| corpus-wide recall@50 | **0.5849** | 0.5582 | prefix **helps** +0.027 |
| doc-scoped recall@5 | 0.5751 | **0.5842** | prefix **hurts** −0.009 |
| doc-scoped MRR | **0.4421** | 0.4271 | prefix helps +0.015 |

**Kept**, because it helps where document identification is the hard part.

Two lessons worth more than the numbers:

1. **A metric can move the wrong way while the system improves.** The floor got
   worse; retrieval got better. Optimising the intermediate quantity (similarity
   spread) instead of the end metric would have led you to delete the prefix.
2. **The prefix's effect is scope-dependent, and in the same direction as the
   entity qualifier in module 10.** Two independent measurements pointing the
   same way is what makes it a *mechanism* rather than noise — see module 10 §7.

## 7. Implementation

```python
class DenseRetriever:
    name = "dense"

    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device="mps",
                 url="http://localhost:6333", collection="clauses"):
        self.model = SentenceTransformer(model_name, device=device)
        self.client = QdrantClient(url=url)
        self.collection = collection

    def search(self, query, k, doc_id=None):
        qv = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        qfilter = None
        if doc_id:
            qfilter = models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=doc_id))])
        res = self.client.query_points(self.collection, query=qv.tolist(), limit=k,
                                       with_payload=True, query_filter=qfilter).points
        return [Hit(r.payload["chunk_id"], float(r.score), r.payload["text"],
                    {"doc_id": r.payload["doc_id"]}) for r in res]
```

Points worth noting:

- **Model constructed once**, in `__init__`. Constructing it per search turns a
  40 ms call into a 10 s one — the classic FastAPI+ML mistake (module 18).
- **`normalize_embeddings=True`** so cosine is a dot product and matches the
  collection's `Distance.COSINE`.
- **`doc_id` is a filter, not a separate code path.** Module 07 explains why that
  one parameter turned out to matter more than any model choice.

## 8. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| Missing/incorrect query prefix | 5–10% quality loss, silent | eval harness; A/B the prefix | prefix queries only |
| Different model for index vs query | results effectively random | store the model name in the collection | re-index with one model |
| Model loaded per request | first-request latency on every request | compare 1st vs 2nd request timing | load at startup |
| Thresholding raw scores | wrong abstentions | measure the random-pair floor | rank logic, or a cross-encoder |
| Chunk exceeds 512 tokens | tail invisible in its own vector | tokenise and compare lengths | smaller chunks / windowing |
| Proper nouns blurred | entity queries retrieve wrong document | per-category recall breakdown | hybrid with BM25 (module 08) |

## 9. Compared with

| | Dense (bi-encoder) | Sparse (BM25, module 08) |
|---|---|---|
| Vector | 384 dense floats | ~30k dims, mostly zero |
| Matches | meaning | exact terms |
| Handles synonyms | **yes** | no |
| Handles rare proper nouns | **poorly** — blurs them | **exactly** |
| Out-of-vocabulary term | approximates via sub-words | scores 0 if absent |
| Needs training | yes, on paired data | no — statistics only |
| Index | HNSW graph | inverted index |
| Interpretability | none | you can point at the matching terms |

That "poorly" in row 4 is the whole reason module 08 exists — and this project's
questions name company entities, which is exactly dense retrieval's weak spot.

## 10. Interview angle

**"Explain dense retrieval."** — Embed chunks once offline, embed the query at
request time, return nearest neighbours by cosine. The non-obvious part is
asymmetry: BGE is contrastively trained with an instruction prefix on queries and
none on passages, so you must reproduce that at inference or you silently lose
5–10% — with no error to tell you.

**"You added metadata to your chunks before embedding. Did it help?"** — I
measured it twice and the answers disagreed in an interesting way. It *raised*
mean pairwise chunk similarity by 0.022, making chunks harder to distinguish,
because every chunk gained the same header shape. But on the actual retrieval
metric it helped corpus-wide (+0.019 recall@5, +0.027 recall@50) and slightly
hurt document-scoped. I kept it, and the reason it helps corpus-wide is
document identification — the same mechanism as the entity qualifier in my
re-ranker.

**"How would you debug 'retrieval quality dropped after a deploy'?"** — Check the
silent-failure candidates first, in order: is the query prefix still applied; is
the index built with the same model as the query encoder; is the model loaded
once or per request; did chunk size grow past the encoder's token limit. All four
run without error and all four degrade quality.

## 11. Test yourself

**Q1.** Why does BGE prefix queries but not documents? What breaks if you prefix
both?

<details><summary>Answer</summary>

Queries and passages are different text distributions — short interrogative
fragments versus long declarative prose. BGE is contrastively trained with an
instruction prefix on the query side only, so the prefix acts as a learned switch
telling the encoder "embed this into the space where matching passages live".

Prefixing both removes the asymmetry the model was trained to exploit: documents
now also get the "this is a search query" signal, shifting them within the space
and degrading the alignment the training established. It runs fine and costs
quality silently.

</details>

**Q2.** You inherit a system whose retrieval is mediocre. Name four silent
failures to check before touching the model, and the check for each.

<details><summary>Answer</summary>

1. **Query prefix** — grep the query path for the prefix string; A/B with and
   without on the eval set.
2. **Model mismatch between index and query** — compare the model name recorded
   at index time with the one in the query path; if unrecorded, that's the first
   thing to fix.
3. **Distance metric mismatch** — verify vectors are normalised and the
   collection uses cosine; recompute one cosine by hand and compare to the
   returned score.
4. **Silent truncation** — tokenise the longest chunks and compare to the
   encoder's `max_seq_length`; anything over is partially unembedded.

All four run without raising, which is why they're first.

</details>

**Q3.** Your corpus has random-pair mean similarity 0.643. A colleague wants to
abstain from answering whenever the top score is below 0.75. Evaluate.

<details><summary>Answer</summary>

It fails in both directions simultaneously, because 0.75 sits *inside* the noise
distribution.

- **False abstentions:** relevant chunks scored around 0.767 with spread below
  that, so many correct answers fall under 0.75 and get refused.
- **False acceptances:** random pairs average 0.643 and reach 0.779 at p99, so a
  substantial mass of irrelevant chunks clears 0.75.

The threshold is uninformative because the score measures *register*, not
relevance, on a corpus where everything shares register. What *does* work is
detecting off-topic queries — nonsense scored 0.578, well clear of 0.767 — so a
much lower threshold could gate "is this question even about contracts", which is
a different question from "is this chunk relevant".

</details>

**Q4.** The contextual prefix raised mean pairwise similarity by 0.022 (bad) but
improved corpus-wide recall@5 by 0.019 (good). Reconcile these, and say what it
implies about optimising intermediate metrics.

<details><summary>Answer</summary>

They measure different things. Mean pairwise similarity measures how
*distinguishable chunks are from each other* — a property of the corpus geometry.
Recall@5 measures whether the *right* chunk outranks the others for a given query.

The prefix adds a shared boilerplate header, which raises chunk-to-chunk
similarity across the board. But it also adds **discriminative signal for the
query** — contract type and governing law help identify *which document* the
question is about, and corpus-wide that is the hard part. The floor rose and the
signal rose more.

The implication: **optimising an intermediate proxy can move you away from the
goal.** Had I treated "lower similarity floor" as the objective, I would have
deleted a change that improves the metric I actually care about. Intermediate
metrics are for *diagnosis*; end metrics are for *decisions*.

</details>

**Q5.** Harder. Dense retrieval blurs proper nouns. Explain *why* from the
training objective, and predict what that implies for a corpus of contracts whose
questions name companies.

<details><summary>Answer</summary>

Contrastive training optimises for *semantic* discrimination: it must tell
"termination clause" from "indemnity clause" across many documents. Two company
names — "Adams Golf" and "Bravatek Solutions" — occupy nearly the same semantic
role: both are *company-shaped entities* in the same syntactic position. Nothing
in the objective rewards separating them, and sub-word tokenisation makes them
share pieces. So they collapse toward the same region.

Prediction for such a corpus: dense retrieval will find *the right kind of
clause* reliably but *in the wrong document*. Precisely what happened — the
corpus-wide config retrieved genuine governing-law clauses from five different
contracts, and the model correctly refused because the question said "this
agreement" (module 09 §6). The implied fixes are lexical matching on the entity
(module 08), metadata filtering by document (module 07), or scoping the query.

</details>

---

**Previous:** [05 — Span coverage](05-span-coverage.md) ·
**Next:** [07 — Vector databases](07-vector-databases.md)
