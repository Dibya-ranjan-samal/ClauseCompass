# 09 — Reciprocal Rank Fusion

---

## 1. What it is

**RRF** combines several ranked lists into one, using only each item's **rank**
in each list — never its score.

```
                    1
RRF(d) =  Σ    ───────────
        lists   k + rank(d, list)
```

- `rank(d, list)` — 1-based position of document `d` in that list (absent → no term)
- `k` — a constant, conventionally 60, controlling how steeply rank matters

## 2. Why it exists

Module 08 ended with two numbers that cannot be combined:

| Retriever | Typical top score | Range |
|---|---|---|
| Dense (cosine) | 0.81 | bounded [0, 1] |
| BM25 | 13.85 | unbounded, corpus-dependent |

**These live on different scales with different meanings.** Averaging them is
arithmetic on incommensurable units — like averaging a temperature in Celsius
with a distance in miles.

### Why not just normalise the scores?

The obvious fix — min-max normalise each list to [0,1], then average — fails for
reasons worth knowing:

1. **Min-max depends on the result set.** The same document gets a different
   normalised score depending on which *other* documents were retrieved. Change
   `k` from 50 to 100 and every normalised score shifts.
2. **The distributions have different shapes.** Cosine scores on this corpus are
   compressed into ~[0.6, 0.8] (module 00's anisotropy); BM25 scores are
   long-tailed with most at 0. Linearly mapping both to [0,1] does not make them
   comparable — it just makes them *look* comparable.
3. **BM25's scale moves with the query.** A two-term query and a six-term query
   produce different score magnitudes, so per-query normalisation is required and
   still doesn't fix (2).

**Ranks avoid all of this.** "Third best" means the same thing in both lists,
regardless of scale, distribution, or query. You discard magnitude information —
that is a real cost — in exchange for a comparison that is actually valid.

## 3. Intuition: agreement beats confidence

```
dense:  1. chunk_A    2. chunk_B    3. chunk_C
bm25:   1. chunk_D    2. chunk_B    3. chunk_E
```

`chunk_B` is second in both. `chunk_A` and `chunk_D` are each first in one list
and absent from the other. With `k = 60`:

```
chunk_B: 1/(60+2) + 1/(60+2) = 0.01613 + 0.01613 = 0.03226
chunk_A: 1/(60+1)            = 0.01639
chunk_D: 1/(60+1)            = 0.01639
```

`chunk_B` wins. **Two retrievers agreeing at rank 2 beats one retriever's
top pick.** That is the design principle: independent methods agreeing is
stronger evidence than one method being confident.

## 4. The algebra of `k` — and a test that caught my own error

I wrote a unit test asserting that a *small* `k` would let a lone rank-1 hit beat
a doc ranked 2nd in both lists. **It failed.** Here's why, and it's worth
deriving because it's a genuine property of the formula.

**Claim:** two rank-2 hits always beat one rank-1 hit, for every `k ≥ 0`.

```
two at rank 2:   2/(k+2)
one at rank 1:   1/(k+1)

2/(k+2) > 1/(k+1)
⟺  2(k+1) > (k+2)        [cross-multiply; both denominators positive]
⟺  2k + 2 > k + 2
⟺  k > 0
```

True for all `k > 0`. So **`k` cannot change that ordering.** My test premise was
mathematically impossible.

### So what *does* `k` control?

How steeply a *better rank* is rewarded. Compare rank 1 against rank 5 in a
single list:

```
ratio = [1/(k+1)] / [1/(k+5)] = (k+5)/(k+1)
```

| `k` | ratio | interpretation |
|---|---|---|
| 1 | 6/2 = **3.00** | rank 1 worth 3× rank 5 — rank matters enormously |
| 10 | 15/11 = 1.36 | moderate |
| 60 | 65/61 = **1.066** | rank 1 barely beats rank 5 — ranks nearly flat |
| 1000 | 1005/1001 ≈ 1.004 | effectively "appeared in the list at all" |

**Small `k` trusts each retriever's ordering. Large `k` trusts only membership.**

As `k → ∞`, RRF degenerates to counting how many lists contained the document.

The corrected test, which asserts something true:

```python
def test_rrf_k_controls_how_steeply_rank_matters():
    lst = [h("first"), h("b"), h("c"), h("d"), h("fifth")]
    ratio = lambda kk: (lambda o: o["first"] / o["fifth"])(
        {x.chunk_id: x.score for x in rrf_fuse([lst], k=kk)})
    assert ratio(1) > 2.5      # rank matters a lot
    assert ratio(60) < 1.1     # ranks nearly flat
    assert ratio(1) > ratio(60)
```

> **The lesson:** a unit test on pure logic caught a misconception I would
> otherwise have carried into an interview. Test the *properties* of a formula,
> not just that the code runs.

## 5. Implementation

```python
def rrf_fuse(rankings: list[list[Hit]], k: int = 60, top_n: int = 50) -> list[Hit]:
    scores: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for lst in rankings:
        for rank, hit in enumerate(lst, start=1):      # 1-based
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit.chunk_id, hit)        # keep first-seen text
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]
    return [Hit(cid, s, by_id[cid].text, by_id[cid].metadata) for cid, s in ordered]
```

Four details:

- **`enumerate(lst, start=1)`** — ranks are 1-based. Starting at 0 makes the first
  item `1/(k+0)`, over-weighting it and dividing by zero when `k=0`.
- **`scores.get(..., 0.0) +`** — accumulation is what implements "agreement".
- **`by_id.setdefault`** — deduplication. A chunk in both lists must appear once.
- **Complexity** O(total items), trivially cheap: fusion took 2 ms against 94 ms
  for the retrievals it fuses.

## 6. In this system — hybrid made recall@5 *worse*

| Config (doc-scoped) | recall@5 | recall@50 | MRR |
|---|---|---|---|
| Dense only | **0.575** | 0.979 | **0.442** |
| BM25 only | 0.411 | 0.980 | 0.319 |
| **Hybrid RRF** | 0.481 | **0.987** | 0.394 |

And corpus-wide:

| Config | recall@5 | recall@50 |
|---|---|---|
| Dense only | **0.363** | 0.585 |
| Hybrid RRF | 0.338 | **0.604** |

**Hybrid is worse than dense on recall@5 in both scopes, and better on
recall@50 in both.**

### Interpreting that, precisely

Recall@50 measures the **candidate pool**; recall@5 measures the **ordering**.
So:

- BM25 contributes *new* gold chunks dense missed → pool improves (0.585 → 0.604)
- BM25's ordering is noisier → fused top-5 degrades (0.363 → 0.338)

Module 08 §10 Q5 predicted exactly this from the recall@50/recall@5 decomposition.

**Why keep it?** Because a better pool is worth having *if* something downstream
can re-order it. That something is the cross-encoder — and doc-scoped, hybrid +
re-ranking reached **0.762**, far above dense alone, even though hybrid alone was
worse. Fusion is a means, not an end.

## 7. The sweep, and a bug it exposed

### `rrf_k`

| `rrf_k` | recall@5 | recall@50 |
|---|---|---|
| **10** | **0.3564** | 0.6044 |
| 20 | 0.3382 | 0.6044 |
| 60 (standard) | 0.3382 | 0.6044 |
| 120 | 0.3382 | 0.6044 |

`k = 10` wins. Consistent with §4: dense has the better ordering here, so a
smaller `k` — which trusts ordering more — lets dense's good ranks dominate
BM25's noisy ones. Note recall@50 is identical everywhere, because `k` reorders
the union without changing it.

### `bm25_candidates` — and F24

The standard advice when hybrid underperforms is "BM25 is contributing noise,
retrieve fewer BM25 candidates". Tested:

| `bm25_candidates` | recall@5 |
|---|---|
| 5 | 0.2974 |
| 10 | 0.3043 |
| 25 | 0.3538 |
| 50 | **0.3564** |

**Monotonic: more BM25 candidates is better.** The advice is wrong for this
corpus. The problem was never BM25's *volume*; it was its *ordering* — a
different disease needing a different cure.

### The bug that nearly hid this

The first run of that sweep returned **byte-identical metrics** for
`bm25cand10`, `bm25cand25` and `rrf_k60` — 0.3382 / 0.6044 / 0.2755 across all
three. Four decimal places of coincidence is not coincidence.

```python
per = max(self.per_k, k)                    # k = 50 from the runner
d = self.dense.search(query, per, doc_id)
s = self.sparse.search(query, per, doc_id)  # per_k clamped away!
```

`max(10, 50) = 50`. A guard added so fusion could always fill `k` slots had
**silently disabled the knob**. The sweep ran, reported, and measured nothing.

The fix makes the fetch asymmetric, because the two retrievers need different
things:

```python
d = self.dense.search(query, max(self.per_k, k), doc_id)   # must be able to fill k
s = self.sparse.search(query, self.per_k, doc_id)          # this is the knob
```

**How it was caught:** identical numbers across supposedly different configs.
Same detection mechanism as F20 ("kept 120, deleted 2") and F9 (character totals
not reconciling) — *the output contradicted itself and someone read it carefully*.

> A parameter sweep that reports suspiciously stable results is not stable. It is
> broken.

## 8. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| Averaging raw scores | one retriever dominates arbitrarily | print both score ranges | fuse on ranks |
| Min-max normalising | scores shift when `k` changes | vary `k`, watch scores move | fuse on ranks |
| 0-based ranks | first item over-weighted; div-by-zero at `k=0` | unit-test the top item's score | `start=1` |
| No dedup | one chunk occupies several slots | assert output ids are unique | `setdefault` |
| **Parameter clamped by a guard** | identical metrics across configs | compare runs bit-for-bit | asymmetric fetch; assert the knob changes output |
| Fusing before filtering | filtered results < k | check `len(hits)` | pre-filter in each retriever |
| Assuming hybrid always helps | recall@5 drops | measure both recall@5 and @50 | keep it only if a re-ranker follows |

## 9. Compared with

| | RRF | Score normalisation + weighted sum | Learned fusion (LTR) |
|---|---|---|---|
| Input | ranks | normalised scores | features (scores, ranks, metadata) |
| Needs training | no | no | yes, with labels |
| Scale-invariant | ✅ | ❌ | depends |
| Tunable | one `k` (plus per-list weights) | weights per retriever | fully |
| Discards magnitude | ✅ (a real cost) | no | no |
| Robustness | high | brittle to distribution shape | needs enough labels |

RRF's popularity is precisely its **robustness with no training**. Its cost is
throwing away confidence information: a dense hit at 0.95 and one at 0.71 are
both "rank 1" if they top their lists.

## 10. Interview angle

**"How do you combine two retrievers?"** — Fuse on ranks, not scores, because
cosine ~0.8 and BM25 ~14 are incommensurable and normalising them is illusory —
min-max depends on the result set and the distributions have different shapes.
RRF sums `1/(k+rank)`, so agreement between independent retrievers beats one
retriever's confidence.

**"What does `k` do in RRF?"** — Controls how steeply better ranks are rewarded.
Rank 1 is worth 3× rank 5 at `k=1` and only 1.07× at `k=60`. As `k → ∞` it
becomes "did this appear at all". What `k` cannot change is the agreement rule:
two rank-2 hits always beat one rank-1 hit, because `2/(k+2) > 1/(k+1)` for all
`k > 0` — I know because a unit test rejected my assumption otherwise.

**"Did hybrid retrieval help?"** — Not the way I expected, and that's the useful
part. Recall@50 rose (0.585 → 0.604) but recall@5 *fell* (0.363 → 0.338). BM25
contributed new gold chunks with noisier ordering, so the candidate pool improved
while the top-5 degraded. It only became a win once a cross-encoder re-ordered
that better pool — doc-scoped 0.762 versus 0.575 dense-only.

**"Tell me about a bug in your evaluation code."** — My `bm25_candidates` sweep
returned byte-identical metrics for three different values. A `max(per_k, k)`
guard I'd added so fusion could fill `k` slots had clamped the parameter away, so
the sweep measured nothing. I caught it by noticing four-decimal agreement across
configs that should differ. The re-run then contradicted the standard advice:
more BM25 candidates was monotonically better, so the issue was BM25's ordering,
not its volume.

## 11. Test yourself

**Q1.** Two lists. List A: `[X, Y, Z]`. List B: `[Y, W, X]`. Compute RRF with
`k = 60` and give the final order.

<details><summary>Answer</summary>

```
X: 1/(60+1) + 1/(60+3) = 0.016393 + 0.015873 = 0.032266
Y: 1/(60+2) + 1/(60+1) = 0.016129 + 0.016393 = 0.032522
Z: 1/(60+3)            = 0.015873
W: 1/(60+2)            = 0.016129
```

Order: **Y, X, W, Z**.

Y edges out X because Y is ranked 1st and 2nd while X is 1st and 3rd — a small
difference at `k=60`, since ranks are nearly flat at that value. At `k=1` the gap
would widen substantially.

</details>

**Q2.** Prove that two rank-2 hits always beat one rank-1 hit under RRF.

<details><summary>Answer</summary>

Two at rank 2 score `2/(k+2)`; one at rank 1 scores `1/(k+1)`.

```
2/(k+2) > 1/(k+1)
⟺ 2(k+1) > (k+2)      both denominators positive, so cross-multiplication is safe
⟺ 2k + 2 > k + 2
⟺ k > 0
```

Holds for every `k > 0`. So `k` cannot invert this ordering — it only changes
*how much* better ranks are worth within a list. Any intuition of the form "small
`k` lets a strong single hit win" is false.

</details>

**Q3.** Your hybrid retriever has higher recall@50 but lower recall@5 than dense
alone. Explain what each metric is telling you, and decide whether to ship it.

<details><summary>Answer</summary>

Recall@50 describes the **candidate pool**: the sparse retriever found gold chunks
the dense retriever missed, so the union is richer. Recall@5 describes the
**ordering**: BM25's ranks are noisier, so mixing them degrades the top of the
fused list.

Ship decision depends entirely on what comes next. Standalone — **don't ship**,
you'd be serving a worse top-5. With a cross-encoder downstream — **ship**,
because re-ranking can only reorder what was fetched, so a better pool raises the
achievable ceiling. Exactly what happened: hybrid alone 0.481, hybrid +
cross-encoder 0.762, dense alone 0.575.

</details>

**Q4.** You sweep a parameter and three values give identical metrics to four
decimal places. What do you conclude and what do you check?

<details><summary>Answer</summary>

Conclude that **the parameter is not reaching the computation** — identical
floating-point aggregates across genuinely different configurations essentially
never happen by chance.

Checks, in order:
1. Trace the parameter from the CLI to its use site, looking for a `max`, `min`,
   `or`, or default that overrides it.
2. Log the effective value inside the function that consumes it, not where it's
   set.
3. Assert that changing it changes *something* observable — e.g. the retrieved id
   list differs — before trusting any metric from the sweep.

This is what F24 was: `max(self.per_k, k)` clamped `per_k` to `k=50` for every
configured value.

</details>

**Q5.** Harder. RRF discards score magnitude. Construct a case where that loses
real information, and say what you'd do instead.

<details><summary>Answer</summary>

Case: a query where dense retrieval is *confidently* right and BM25 has no
matching terms at all. Dense returns its top hit at cosine 0.91 — far above the
0.643 corpus floor, so genuinely exceptional. BM25's query terms are absent, so
all its scores are 0.0 and `argsort` returns an **arbitrary** order (module 08).

RRF sees "rank 1 in list A, rank 1 in list B" and treats BM25's arbitrary top
item as equally strong evidence as dense's confident one. The noise is promoted to
parity with the signal, because rank alone cannot express "this list has no
opinion".

What to do instead:
1. **Gate on degenerate lists** — if `max(bm25_scores) == 0`, drop that list from
   the fusion entirely rather than letting arbitrary order contribute.
2. **Weight the lists**, giving dense a higher coefficient
   (`w·1/(k+rank)`) when you have evidence it's the stronger retriever — which
   the ablation provides.
3. **Learned fusion**, if you have enough labelled queries: feed both scores,
   both ranks, and list-level features to a small ranker. This recovers magnitude
   without hand-tuning, at the cost of needing labels and re-training as the
   corpus shifts.

The honest summary: RRF's scale-invariance is bought with an inability to express
confidence, and the fix is either explicit gating or a model that consumes the
scores you discarded.

</details>

---

**Previous:** [08 — BM25](08-bm25.md) ·
**Next:** [10 — Cross-encoders & re-ranking](10-cross-encoders.md)
