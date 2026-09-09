# 10 — Cross-Encoders & Re-ranking

The biggest measured win in the project (recall@5 0.575 → 0.762) and the biggest
measured *loss* (0.363 → 0.158), from the same component. Understanding why is
the most valuable thing in this course.

---

## 1. What it is

A **cross-encoder** takes a query and a document **together** as one input and
outputs a single relevance score.

```
bi-encoder  (module 06):   encode(query) · encode(doc)     -> two vectors, compared after
cross-encoder (this one):  encode(query ⊕ doc)             -> one score, computed jointly
```

**Re-ranking** is using it to reorder a shortlist that a cheap retriever
produced.

## 2. Why it exists

A bi-encoder must compress a document into 384 numbers **without knowing the
query**. That vector has to be simultaneously useful for every possible
question. It is a lossy summary computed in ignorance.

A cross-encoder sees both at once, so its attention layers can put query tokens
and document tokens in the same context window and let them interact directly:

```
[CLS] what is the liability cap ? [SEP] 8.2 the total liability of each party
      └──────────┬──────────┘          └───────────────┬───────────────────┘
                 └────── attention connects these directly ──────┘
```

The token `cap` can attend to `total liability`, and `liability` in the query can
attend to `8.2`. Nothing in the bi-encoder architecture permits that — the two
texts never meet.

**The cost:** you cannot precompute. There is no reusable "document vector",
because the representation depends on the query. Every (query, document) pair
needs its own forward pass.

## 3. Intuition and the arithmetic that forces the two-stage design

Suppose one cross-encoder forward pass takes 20 ms.

| Approach | Passes per query | Time |
|---|---|---|
| Cross-encoder over the whole corpus | 18,280 | **366 seconds** |
| Bi-encoder retrieve 50, cross-encode those | 50 (+ 1 embed) | **1 second** |

366 seconds per query is not a product. So:

```
18,280 chunks ──[bi-encoder, cheap, approximate]──> 50 candidates
                                                       │
                        ──[cross-encoder, expensive, accurate]──> top 5
```

**This is the standard two-stage retrieval pattern**, and the reason it works is
that the two stages have complementary error profiles: the bi-encoder is good at
*recall* over huge sets, the cross-encoder is good at *precision* over small
ones.

## 4. The ceiling — the single most important property

A re-ranker **only reorders what retrieval already found.**

```
recall@5 after re-ranking  ≤  recall@candidates before re-ranking
```

Two consequences, and both were load-bearing in this project:

### (a) It bounds the achievable gain

| Scope | recall@50 (the pool) | Ceiling for re-ranking |
|---|---|---|
| doc-scoped | 0.987 | plenty of headroom above 0.575 |
| corpus-wide | **0.604** | almost nothing above 0.363 |

Corpus-wide, 40% of gold chunks are absent from the pool. The re-ranker cannot
promote a chunk that was never fetched. **You cannot rank your way out of a
recall problem.**

### (b) It gives you a free correctness assertion

Since re-ranking permutes a fixed set, `recall@50` must be **numerically
identical** before and after:

```python
assert hybrid["recall@50"] == rerank["recall@50"]
```

Measured: `0.6044 == 0.6044` corpus-wide, `0.9867 == 0.9867` doc-scoped. ✅

If it moved, you'd have a bug — dropping or adding candidates — not a result.
**Design your components so that some invariant must hold, then assert it.**

## 5. Implementation, and the truncation problem

```python
class RerankingRetriever:
    def __init__(self, base, model_name="BAAI/bge-reranker-base", device="cpu",
                 candidates=50, max_length=512, split_long=True, strip_entity=True):
        self.model = CrossEncoder(model_name, max_length=max_length, device=device)
        self.max_chars = max_length * 3        # rough chars-per-token estimate

    def search(self, query, k, doc_id=None):
        cands = self.base.search(query, max(self.candidates, k), doc_id)
        rq = ENTITY_PREFIX.sub("", query) if self.strip_entity else query
        scores = self._score(rq, cands)
        ranked = sorted(zip(scores, cands), key=lambda t: -t[0])
        return [Hit(h.chunk_id, float(s), h.text, h.metadata) for s, h in ranked][:k]
```

`device="cpu"` is deliberate: `CrossEncoder` on Apple's MPS backend is flaky
across `sentence-transformers` versions, and 50 pairs on CPU is tolerable.

### Truncation, and why splitting in half was not enough

The re-ranker's window is 512 tokens ≈ 1,536 characters. Measured over 18,280
chunks:

| | count |
|---|---|
| Chunks longer than the window | **6,679 (36%)** |
| Chunks whose **halves still overflow** | **3,815 (20%)** |
| Longest chunk | 4,192 chars = **2.7× the window** |

Splitting a 4,192-character chunk in two gives 2,096-character halves — still
over. So the first implementation windows the whole chunk and keeps the best
window:

```python
for start in range(0, len(h.text), self.max_chars):
    pairs.append((query, h.text[start:start + self.max_chars]))
    owner.append(i)
...
raw = self.model.predict(pairs, batch_size=16)
best = [float("-inf")] * len(hits)
for score, i in zip(raw, owner):
    best[i] = max(best[i], float(score))     # max-pooling over windows
```

**Max-pooling** is the right aggregation: a clause is relevant if *any* part of it
answers the query, so the best window should represent it. Mean-pooling would
punish a long clause with one relevant paragraph.

## 6. The result — and the finding that reframed the project

### Document-scoped

| Configuration | recall@5 | recall@50 | MRR | p95 |
|---|---|---|---|---|
| Dense only | 0.575 | 0.979 | 0.442 | 29 ms |
| Hybrid RRF | 0.481 | 0.987 | 0.394 | 87 ms |
| Hybrid + cross-encoder | 0.606 | 0.987 | 0.435 | 12,046 ms |
| **+ entity-stripped query** | **0.762** | 0.987 | **0.657** | 11,599 ms |

### Corpus-wide

| Configuration | recall@5 | recall@50 | MRR |
|---|---|---|---|
| **Dense only** | **0.363** | 0.585 | **0.315** |
| Hybrid + cross-encoder | 0.331 | 0.604 | 0.263 |
| + entity-stripped query | **0.158** | 0.604 | 0.136 |

Look at the last row of each table. **The same code change is worth +0.156 in
one scope and −0.173 in the other.**

## 7. Why: document identification vs clause relevance

This is the deepest idea in the course, so build it carefully.

The eval questions are **entity-qualified** (module 12):

> *"Under the **Adams Golf Endorsement** agreement, which state's law governs it?"*

The party name is there so that corpus-wide retrieval has something to identify
the document with. But measured on the gold chunks:

> **Only 21% (26/121) of gold clauses contain the party name the question
> states.**

Of course they don't — the party name lives in the *preamble*, not in the
governing-law clause.

Now trace what the cross-encoder does with that:

**Corpus-wide.** Candidates come from many documents. The query says "Adams
Golf". A chunk *containing* "Adams Golf" (say, the preamble of the right
contract, or any chunk of a similarly-named company) gets rewarded for a term
overlap the correct governing-law clause lacks. The cross-encoder attends to the
mismatch and **demotes the correct clause**. Stripping the entity removes the
only signal telling it which document to prefer — hence −0.173.

Wait — both effects can't be right at once. And they aren't: stripping *hurts*
corpus-wide precisely because the name is doing document-identification work
there. Keeping it is the lesser evil.

**Document-scoped.** The payload filter has already fixed the document. Every
candidate is from the right contract. The party name now carries **zero
information** — it cannot discriminate between candidates that all share it or
all lack it — while still consuming attention and competing with the clause
semantics that actually matter. Removing it lets the model judge clause relevance
cleanly: **+0.172**.

### The unifying statement

> **Retrieval over a many-document corpus is two problems wearing one coat:**
> **(1) which document?  (2) which clause within it?**
> Identity signals help the first and hurt the second.

Three independent measurements agree:

| Signal | corpus-wide | doc-scoped |
|---|---|---|
| Entity qualifier in the rerank query | **+0.173** | **−0.172** |
| Day 2 contextual prefix (module 06) | +0.019 | −0.009 |
| BM25 lexical matching (module 08) | raises the ceiling +0.019 | +0.001 |

Two independent effects with the same sign pattern is what turns a curiosity into
a **mechanism**. This is why `config.py` documents `strip_entity_for_rerank` as
scope-dependent and unsafe to set globally.

## 8. Attribution — and being wrong twice

I shipped two changes together (windowing + entity-stripping), got +0.156, and
could not say which caused it. One extra run with `--no-strip-entity` separated
them:

| Arm | recall@5 | MRR |
|---|---|---|
| rerank, halves only, keep entity | 0.6062 | 0.4351 |
| + full windowing (keep entity) | 0.5906 | 0.4796 |
| + windowing + strip entity | **0.7622** | **0.6567** |
| **windowing alone** | **−0.0156** | +0.0445 |
| **stripping alone** | **+0.1716** | +0.1771 |

I had predicted the wrong sign **twice**:

1. I expected entity-stripping to help corpus-wide. It collapsed recall@5 to 0.158.
2. I was confident the *windowing* fix — fixing a provable truncation defect
   affecting 20% of chunks — would carry the gain. It was **slightly negative** on
   recall@5.

The fix I believed in was worth −0.016. The fix I had backwards was worth
**eleven times more**.

> **Two lessons.** First, "change one thing, measure" is not pedantry — I broke
> it and it cost a 20-minute run to untangle, and would have cost a *confident
> wrong explanation* in a README and an interview. Second, a defect being
> *provable* (20% of chunks demonstrably truncated) does not make fixing it
> *impactful*. Correctness and impact are different axes.

Windowing was kept anyway: +0.04 MRR and it removes a real defect. But
`config.py` labels it honestly as "~neutral on recall@5".

## 9. Where the cross-encoder genuinely shone: near-miss

| Type | dense | BM25 | hybrid+rerank+strip |
|---|---|---|---|
| single_hop | 0.577 | 0.421 | **0.756** |
| **near_miss** | 0.500 | **0.250** | **0.917** |
| multi_hop | 0.625 | 0.354 | 0.708 |

Near-miss questions ask about clause A in a contract that also contains adjacent
clause B ("cap on liability" vs "uncapped liability"). Module 08 showed BM25
*lost* there — adjacent legal language shares vocabulary.

The cross-encoder reaches **0.917** because it reads the pair jointly and can
distinguish *"liability shall not exceed $1,000,000"* from *"liability is uncapped
for gross negligence"* — sentences with nearly identical bags of words and
opposite meanings. **This is the case that only a joint model can solve**, and
the measurement says so. (Caveat: n=6. Directional.)

## 10. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **recall@50 changes after re-ranking** | — | assert equality | you're dropping candidates, not reordering |
| Re-ranking a poor pool | tiny or negative gain | compare recall@candidates to recall@5 | fix retrieval first |
| Silent truncation | long clauses' tails ignored | `% chunks > max_length·3` | window + max-pool |
| Query contains scope noise | helps one scope, hurts another | ablate per scope | scope-dependent config |
| MPS instability | crashes or wrong scores | run on CPU and compare | `device="cpu"` |
| Latency shock | p95 explodes 300× | measure p95 per config | fewer candidates, smaller model, quantise |
| Two changes at once | unattributable gain | one change per run | separate runs |

## 11. Compared with

| | Bi-encoder | Cross-encoder |
|---|---|---|
| Input | each text alone | the pair, jointly |
| Query-document interaction | none | full attention |
| Output | a vector | a scalar score |
| Precomputable | ✅ embed corpus once | ❌ per pair |
| Cost per query | 1 embed + dot products | `candidates` forward passes |
| p95 here | 29 ms | 11,599 ms (**400×**) |
| Scales to | millions | tens per query |
| Best at | recall over huge sets | precision over shortlists |
| Near-synonym discrimination | weak (0.500) | strong (**0.917**) |

## 12. Interview angle

**"What does a cross-encoder do that a bi-encoder can't?"** — Let query and
document tokens attend to each other. A bi-encoder compresses a document into a
fixed vector *without seeing the query*, so it's a lossy summary computed in
ignorance. The price is that nothing is precomputable — 400× the latency here,
29 ms to 11.6 s.

**"Why not use it for everything?"** — Arithmetic. 18,280 chunks × ~20 ms is 366
seconds per query. Two-stage retrieval exists because the bi-encoder is good at
recall over huge sets and the cross-encoder is good at precision over shortlists.

**"How do you know your re-ranker isn't broken?"** — recall@50 is invariant under
re-ranking by construction, so I assert it. 0.6044 before and after corpus-wide,
0.9867 doc-scoped. If it moved I'd have a bug rather than a result.

**"Your re-ranker made one config worse. Explain."** — Best question to be asked,
because it's the real finding. My eval questions name the contract, but only 21%
of gold clauses contain that party name. Corpus-wide the cross-encoder needs the
name to identify the document; document-scoped the payload filter already did
that, so the name is pure noise competing with clause semantics. The same
one-line change was +0.172 in one scope and −0.173 in the other. The Day 2
contextual prefix showed the same sign pattern, which is what convinced me it was
a mechanism and not noise.

**"Tell me about a time you were wrong."** — I shipped a truncation fix and a
query fix together and got +0.156 I couldn't attribute. One extra run showed the
truncation fix — which addressed a provable defect in 20% of chunks — was worth
−0.016, and the query fix I'd had *backwards* was worth +0.172. A defect being
provable doesn't make fixing it impactful.

## 13. Test yourself

**Q1.** Why must `recall@50` be unchanged by re-ranking with 50 candidates, and
what does it mean if it isn't?

<details><summary>Answer</summary>

Re-ranking is a **permutation** of a fixed candidate set. `recall@50` asks "what
fraction of gold chunks are in the top 50?" — and the top 50 *is* the whole
candidate set, just in a different order. Set membership is invariant under
permutation, so the value cannot change.

If it changes, you are not permuting: you're dropping candidates (returning fewer
than you received, or filtering by score), or adding them (re-fetching). Either
is a bug, and this assertion is why you'd catch it immediately rather than
attributing the movement to the model.

</details>

**Q2.** Corpus-wide recall@50 is 0.604 and recall@5 is 0.363. Estimate the
maximum recall@5 a perfect re-ranker over 50 candidates could achieve, and say
what that implies.

<details><summary>Answer</summary>

**0.604** — the re-ranker can at best promote every gold chunk that is in the
pool into the top 5 (assuming ≤5 gold per question, which mostly holds).

So the entire available gain is 0.363 → 0.604, and 39.6% of gold chunks are
permanently unreachable because they were never fetched. That implies re-ranking
is *not* the highest-value work corpus-wide: the binding constraint is retrieval
coverage. Any effort should go to chunking, lexical matching, query rewriting, or
two-stage document-then-clause retrieval — not to a better re-ranker.

</details>

**Q3.** Only 21% of gold clauses contain the party name the question states.
Explain why removing that name from the query helps document-scoped retrieval but
devastates corpus-wide.

<details><summary>Answer</summary>

The name is a **document-identification** signal, not a clause-relevance signal.

*Corpus-wide*, candidates span many contracts, so the model needs some way to
prefer the right document's clauses. The party name is the only such cue in the
query. Remove it and the cross-encoder ranks purely on clause semantics — finding
the most "governing-law-ish" clause in the corpus regardless of which contract
it belongs to. Recall collapses to 0.158.

*Document-scoped*, the payload filter has already selected the document. Every
candidate shares (or equally lacks) the party name, so it can no longer
discriminate anything — but it still occupies attention and competes with the
clause semantics that do discriminate. Removing it is pure gain: +0.172.

Same token, opposite value, because the *job* changed.

</details>

**Q4.** 36% of your chunks exceed the re-ranker's window and 20% still overflow
after being halved. Why is **max-pooling** over windows the right aggregation
rather than mean or first-window?

<details><summary>Answer</summary>

Relevance is **existential**, not average: a clause answers the query if *any*
part of it does. A 4,000-character clause whose final paragraph contains the
liability cap is fully relevant even though 80% of it is about something else.

- **Mean-pooling** would dilute that — averaging one highly relevant window with
  two irrelevant ones produces a mediocre score, systematically penalising long
  clauses.
- **First-window** is what naive truncation does, and it discards the tail
  entirely, which is the defect being fixed.
- **Max-pooling** takes the best evidence the chunk offers, matching the semantics
  of "is this chunk relevant".

The caveat worth knowing: max-pooling is optimistic and can be gamed by a long
chunk containing one coincidentally-matching sentence — a precision cost paid for
a recall gain.

</details>

**Q5.** Harder. Your doc-scoped p95 is 11.6 s against 29 ms for dense. Give three
ways to reduce it and the quality cost of each.

<details><summary>Answer</summary>

1. **Fewer candidates** (50 → 20). Cost scales roughly linearly, so ~2.5× faster.
   Quality cost: lowers the ceiling to `recall@20` instead of `recall@50` — here
   0.987 is already near-saturated at 50, and recall@10 was 0.857, so the ceiling
   loss is real but modest. Measure `recall@k` at the candidate depth to price it
   exactly.
2. **A smaller cross-encoder** (`bge-reranker-base` → a MiniLM cross-encoder,
   ~90 MB vs 1.1 GB). Several times faster. Quality cost: unknown until measured —
   and this is the point, you *ablate* it against the golden set rather than
   assuming the smaller model is adequate.
3. **Quantisation / ONNX / GPU.** int8 quantisation typically gives 2–4× on CPU
   with small accuracy loss; ONNX Runtime removes Python overhead; a GPU removes
   the problem. Quality cost near zero for ONNX, small and measurable for
   quantisation, none for GPU — but each adds deployment complexity.

A fourth, structural option: **only re-rank when it pays**. This project already
does a version of this — the API applies the cross-encoder *only* when a `doc_id`
is supplied, because corpus-wide it is both slower and worse. Conditional
application is often better than uniform optimisation.

</details>

---

**Previous:** [09 — Rank fusion](09-rank-fusion.md) ·
**Next:** [11 — Retrieval metrics](11-retrieval-metrics.md)
