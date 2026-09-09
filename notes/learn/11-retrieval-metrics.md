# 11 — Retrieval Metrics: recall@k, hit@k, MRR, nDCG

---

## 1. What they are

Four functions, all taking the same two inputs:

- **`gold`** — the set of chunk ids that *should* be retrieved
- **`retrieved`** — the ranked list of chunk ids that *were* retrieved

They differ in what question they answer. Using the wrong one gives you a number
that doesn't mean what you think.

**Running example for the whole module:**

```
gold      = {a, b}
retrieved = [x, a, y, z, b]        # a at rank 2, b at rank 5
```

## 2. Why they exist — and why four

"Did retrieval work?" is ambiguous. Four different reasonable readings:

| Question | Metric |
|---|---|
| What fraction of what I needed did I find? | **recall@k** |
| Did I find *anything* useful? | **hit@k** |
| How high up was the first useful thing? | **MRR** |
| Did I find everything, weighted by position? | **nDCG@k** |

They are not interchangeable, and the gaps between them are diagnostic.

## 3. recall@k — coverage

```
             |gold ∩ retrieved[:k]|
recall@k = ───────────────────────
                   |gold|
```

```python
def recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    if not gold:
        return 1.0                       # see §7 — this line is a trap
    return len(gold & set(retrieved[:k])) / len(gold)
```

**Worked:**

- `recall@5` → top-5 is `[x,a,y,z,b]`, contains both `a` and `b` → `2/2 = ` **1.0**
- `recall@2` → top-2 is `[x,a]`, contains only `a` → `1/2 = ` **0.5**
- `recall@1` → top-1 is `[x]` → `0/2 = ` **0.0**

Note: **position within the top-k is invisible.** Gold at ranks 1 and 2 scores the
same as gold at ranks 4 and 5. That's the metric's design, not a flaw — but it's
why you need MRR too.

## 4. hit@k — did anything land

```
hit@k = 1 if |gold ∩ retrieved[:k]| > 0 else 0
```

**Worked:** `hit@2` → `a` is in `[x,a]` → **1.0**, even though `recall@2 = 0.5`.

Binary, and often what a user actually experiences: one correct clause on screen
is usually enough to answer the question.

### Why hit@5 > recall@5 in this project

| | corpus-wide | doc-scoped |
|---|---|---|
| recall@5 | 0.3632 | 0.5751 |
| **hit@5** | **0.4375** | **0.6797** |

hit@k ≥ recall@k always, and the gap tells you about **multi-gold questions**.
When a question has 3 gold chunks and you retrieve 1, recall credits 0.33 but hit
credits 1.0. A large gap means many questions have several gold chunks and you're
finding some but not all — which for a *user* may be fine and for a *metric* looks
mediocre.

Report both, and know which one your product cares about.

## 5. MRR — how high was the first hit

**Reciprocal rank** of the first gold hit, averaged over queries:

```
          1                                1  N     1
RR = ──────────────         MRR = ── Σ  ────────
      rank_first_gold              N i=1  rank_i
```

```python
def mrr(gold: set[str], retrieved: list[str]) -> float:
    if not gold:
        return 1.0
    for i, cid in enumerate(retrieved, 1):
        if cid in gold:
            return 1.0 / i
    return 0.0
```

**Worked:** first gold is `a` at rank 2 → `1/2 = ` **0.5**.

The reciprocal shape is aggressive:

| rank of first hit | RR |
|---|---|
| 1 | 1.000 |
| 2 | 0.500 |
| 3 | 0.333 |
| 5 | 0.200 |
| 10 | 0.100 |

Moving a result from rank 2 to rank 1 is worth **+0.5**. From rank 9 to rank 8,
**+0.014**. MRR cares almost exclusively about the top of the list — which makes
it the metric most sensitive to exactly what a re-ranker changes.

That's visible in your data: re-ranking took doc-scoped MRR from **0.442 → 0.657**
(+49%) while recall@5 went 0.575 → 0.762 (+33%). MRR moved more because
re-ranking's whole job is promoting the best result to rank 1.

**MRR ignores everything after the first hit.** Gold at ranks 1, 2 and 3 scores
identically to gold at rank 1 alone.

## 6. nDCG@k — position-weighted completeness

The only metric that accounts for *all* gold chunks *and* their positions.

### Step 1: DCG (Discounted Cumulative Gain)

```
          k     relᵢ
DCG@k =  Σ   ──────────
         i=1  log₂(i+1)
```

With binary relevance, `relᵢ = 1` if `retrieved[i]` is gold, else 0. The
`log₂(i+1)` **discount** shrinks the credit for later positions:

| rank `i` | `log₂(i+1)` | weight `1/log₂(i+1)` |
|---|---|---|
| 1 | log₂2 = 1.000 | **1.000** |
| 2 | log₂3 = 1.585 | 0.631 |
| 3 | log₂4 = 2.000 | 0.500 |
| 4 | log₂5 = 2.322 | 0.431 |
| 5 | log₂6 = 2.585 | 0.387 |

Why logarithmic rather than reciprocal like MRR? It's gentler — position 5 is
worth 39% of position 1, whereas MRR would say 20%. The log discount encodes "a
user will plausibly read a few results", not "only the first matters".

### Step 2: IDCG — the best achievable

```
           min(|gold|,k)      1
IDCG@k =      Σ         ──────────
              i=1        log₂(i+1)
```

That is: all gold chunks packed into the top positions.

### Step 3: normalise

```
nDCG@k = DCG@k / IDCG@k
```

Range [0,1], where 1.0 means perfect ordering. Normalising makes the metric
comparable across queries with different numbers of gold chunks.

### Worked, fully

`gold = {a,b}`, `retrieved = [x,a,y,z,b]`, `k = 5`.

```
DCG  = 0/log₂2 + 1/log₂3 + 0/log₂4 + 0/log₂5 + 1/log₂6
     = 0 + 0.6309 + 0 + 0 + 0.3869
     = 1.0178

IDCG = 1/log₂2 + 1/log₂3        (2 gold, best case ranks 1 and 2)
     = 1.0000 + 0.6309
     = 1.6309

nDCG@5 = 1.0178 / 1.6309 = 0.624
```

Compare: `recall@5 = 1.0` (found everything) but `nDCG@5 = 0.624` (found it in
poor positions). **That difference is exactly the information recall@k throws
away.**

```python
def ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    if not gold:
        return 1.0
    dcg   = sum(1.0/math.log2(i+1) for i, cid in enumerate(retrieved[:k], 1) if cid in gold)
    ideal = sum(1.0/math.log2(i+1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0
```

## 7. The empty-gold footgun — and what it would have cost

Every one of those four functions begins:

```python
if not gold:
    return 1.0
```

Defensible in isolation: "nothing was expected, so nothing was missed."

**Dangerous in aggregate.** Your golden set has 148 questions, of which **20 have
no gold chunks**:

- 18 **unanswerable** questions (the correct response is refusal)
- 2 **aggregation** questions (no single chunk contains a corpus-wide count)

The natural aggregation is "average over answerable questions". Both aggregation
questions are marked `answerable: true`. So a naive mean would include them, plus
anyone filtering only on `answerable` would include all 20 in a slightly
different setup — each contributing a **free 1.0**.

### Price it

Corpus-wide recall@5 over the 128 scored questions is **0.3632**. Add 20 free
1.0s:

```
(0.3632 × 128 + 1.0 × 20) / 148 = (46.49 + 20) / 148 = 0.449
```

**0.363 reported as 0.449** — a 24% relative inflation, in the flattering
direction, with no error anywhere.

### The fix

```python
scored = [r for r in results if r["n_gold"] > 0]
agg = {m: mean(r[m] for r in scored) for m in METRICS}
agg["n_excluded_no_gold"] = len(results) - len(scored)     # 20, reported
```

Filter on **`n_gold > 0`**, not on `answerable` — the two differ, and that
difference is where the bug lives.

And pin the hazard in a test so nobody "simplifies" it away:

```python
def test_empty_gold_is_free_score():
    """Documents the footgun: no gold means 1.0, so the runner must filter these."""
    assert recall_at_k(set(), [], 5) == 1.0
    assert hit_at_k(set(), [], 5) == 1.0
    assert mrr(set(), []) == 1.0
```

> **A test that documents a hazard is worth more than one that re-checks
> arithmetic.**

This is the same pathology as module 04's blended median: **a mean over a mixture
of populations describes neither.**

## 8. Reading the gaps — the diagnostic that drove Days 4–6

Individual values matter less than the *differences* between them.

### recall@k vs recall@50 — is it retrieval or ranking?

| Scope | recall@5 | recall@50 | gap | Diagnosis |
|---|---|---|---|---|
| doc-scoped | 0.575 | 0.979 | **0.404** | it's *retrieved*, badly *ranked* → **ranking problem** |
| corpus-wide | 0.363 | 0.585 | 0.222 | recall@50 itself is low → **retrieval problem** |

These were written down on Day 3, *before* Day 4's work, and then tested:

- **Ranking prediction: correct.** Re-ranking took doc-scoped 0.575 → 0.762
  against the 0.987 ceiling.
- **Retrieval prediction: correct in direction, small in size.** BM25 lifted the
  corpus-wide ceiling only 0.585 → 0.604.

**This is the whole reason the eval harness comes before the optimisation.**
Without it you cannot even form the hypothesis, let alone check it.

### hit@k vs recall@k — how many gold chunks per question?

Large gap → many multi-gold questions, and you're finding some but not all.

### MRR vs recall@5 — is the top of the list good?

MRR much lower than recall@5 → you find gold within 5 but rarely at rank 1.
Prime re-ranker territory.

## 9. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **Empty-gold questions in the mean** | metrics inflated ~24% | report `n_excluded_no_gold` | filter on `n_gold > 0` |
| Filtering on `answerable` not `n_gold` | 2 aggregation questions leak in | compare the two counts | filter on gold |
| Averaging over mixed populations | a mean describing nobody | group and report per type | per-type breakdown |
| `k` larger than list length | silently scores as if truncated | assert `len(retrieved) >= k` | fetch enough |
| 0-based ranks in MRR | RR of 1/0 → `ZeroDivisionError` or inflated | unit-test rank 1 = 1.0 | `enumerate(..., 1)` |
| Reporting one metric | can't diagnose anything | — | report the family and the gaps |
| Micro vs macro averaging | large-gold questions dominate | check per-question variance | average per query (macro) |
| Comparing across eval sets | numbers move for the wrong reason | pin the golden set with a hash | version the eval set |

## 10. Compared with

| Metric | Sees positions? | Sees all gold? | Range | Best for |
|---|---|---|---|---|
| **recall@k** | no (within k) | ✅ | [0,1] | coverage; ceilings |
| **hit@k** | no | ❌ (any is enough) | {0,1} | user-felt success |
| **MRR** | ✅ heavily top-weighted | ❌ (first only) | (0,1] | re-ranker sensitivity |
| **nDCG@k** | ✅ log-discounted | ✅ | [0,1] | overall ranking quality |

Also worth knowing (not used here, but interview-relevant):

- **Precision@k** — fraction of the top-k that is gold. Rarely used in RAG because
  `k` is chosen by you, so precision is mostly a function of `|gold|/k`.
- **MAP** — mean average precision; like nDCG but with a different discount.
- **Recall@∞ / span coverage** (module 05) — the ceiling above all of these.

**The hierarchy to keep in your head:**

```
span coverage  ≥  recall@∞  ≥  recall@50  ≥  recall@5  ≥  answer quality
```

Debug in that order. Fixing a stage that isn't the constraint cannot help.

## 11. Interview angle

**"Which retrieval metric do you report and why?"** — All four, because the *gaps*
are the diagnosis. recall@5 for coverage, hit@5 for what a user feels, MRR
because it's most sensitive to re-ranking, nDCG@5 because it's the only one
seeing both completeness and position. Reporting one number means you can't tell
a retrieval problem from a ranking problem.

**"Your recall@5 is 0.36. Is that bad?"** — On its own, meaningless. The
informative number is the gap to recall@50: doc-scoped mine was 0.575 vs 0.979, so
the answer is retrieved and mis-ranked — a ranking problem, which re-ranking then
fixed to 0.762. Corpus-wide recall@50 is only 0.585, so 41% is never retrieved
and re-ranking cannot help; that needs a retrieval-side fix.

**"Tell me about a bug in your metrics."** — Every metric returns 1.0 for empty
gold, which is defensible per-question and disastrous in aggregate. 20 of my 148
questions have no gold chunks — 18 unanswerable, 2 aggregation — so a naive mean
would hand out 20 free 1.0s and report corpus-wide recall@5 as ~0.45 instead of
0.363. I filter on `n_gold > 0`, report the excluded count, and pinned the hazard
in a unit test. The subtlety is that filtering on `answerable` is *not* the same
filter — the two aggregation questions are answerable but have no gold.

**"Why is MRR more sensitive to re-ranking than recall@5?"** — Because MRR is
`1/rank` of the first hit, so promoting a result from rank 2 to rank 1 is worth
+0.5 while recall@5 doesn't move at all — both are "within the top 5". Re-ranking
reorders within the retrieved set, which is precisely what MRR measures and
recall@k is blind to. My data: MRR +49% versus recall@5 +33%.

## 12. Test yourself

**Q1.** `gold = {a}`, `retrieved = [x, y, z, a]`. Compute recall@3, recall@5,
hit@3, MRR.

<details><summary>Answer</summary>

- `recall@3`: top-3 is `[x,y,z]`, no gold → **0.0**
- `recall@5`: top-5 is `[x,y,z,a]` (list has only 4), contains `a` → `1/1 = ` **1.0**
- `hit@3`: no gold in top-3 → **0.0**
- `MRR`: first gold at rank 4 → `1/4 = ` **0.25**

Note recall@3 = 0 and recall@5 = 1.0 from one position of difference — which is
why reporting a single `k` hides a lot.

</details>

**Q2.** `gold = {a,b,c}`, `retrieved = [a,b,c,x,y]`. Compute nDCG@5 and explain
the result.

<details><summary>Answer</summary>

```
DCG  = 1/log₂2 + 1/log₂3 + 1/log₂4 = 1.0000 + 0.6309 + 0.5000 = 2.1309
IDCG = same, since 3 gold in the top 3 is optimal                = 2.1309
nDCG = 2.1309 / 2.1309 = 1.0
```

**1.0** — perfect. All gold chunks occupy the best possible positions, so DCG
equals IDCG. This shows nDCG's normalisation working: it doesn't punish you for
having only 3 gold chunks rather than 5.

</details>

**Q3.** Your metric functions return 1.0 for empty gold. 20 of 148 questions have
no gold and your true score over the other 128 is 0.3632. What does a naive mean
report, and why is the error especially insidious?

<details><summary>Answer</summary>

`(0.3632 × 128 + 1.0 × 20) / 148 = 66.49/148 = ` **0.449**.

0.363 becomes 0.449 — a 24% relative inflation.

Insidious for three reasons:
1. **It flatters.** Errors that make you look worse get investigated; errors that
   make you look better get shipped.
2. **It's silent.** No exception, and 0.449 is a perfectly plausible number for
   this task, so nothing looks wrong.
3. **It scales with the thing you're proud of.** The more unanswerable questions
   you add — and adding them is *good* eval practice — the more the metric
   inflates. Doing the right thing makes the bug worse.

</details>

**Q4.** Why must the aggregation filter be `n_gold > 0` rather than
`answerable == True`? Be specific about what differs.

<details><summary>Answer</summary>

They partition the set differently. The 148 questions break down as:

- 130 `answerable: true`, of which **2 are aggregation questions with
  `gold_chunk_ids: []`**
- 18 `answerable: false` (unanswerable), all with empty gold

Filtering on `answerable` keeps 130 questions, **2 of which have no gold** and
therefore score a free 1.0 each. Filtering on `n_gold > 0` keeps exactly the 128
that can be meaningfully scored.

The conceptual point: `answerable` is a statement about whether the *contract*
answers the question; `n_gold > 0` is a statement about whether *your index can be
scored* on it. Aggregation questions are answerable in principle and unscoreable
by retrieval metrics — a distinction only visible if you look.

</details>

**Q5.** Harder. Two systems: A has recall@5 = 0.70, MRR = 0.40. B has
recall@5 = 0.65, MRR = 0.60. Which would you ship, and what would you check first?

<details><summary>Answer</summary>

**It depends on what consumes the results, and the gap itself is the finding.**

- A finds more gold within the top 5 but ranks it poorly — MRR 0.40 means the
  first hit averages around rank 2.5.
- B finds slightly less but puts it near the top — MRR 0.60 implies first hit
  around rank 1.7.

**Ship B if a human reads the results** — users read top-down and stop early, so
rank-1 quality dominates perceived quality. **Ship A if an LLM consumes all 5
chunks**, since the generator sees the whole set and order matters much less than
whether the evidence is present.

What I'd check first: **recall@50 for both.** If A's recall@50 is much higher, A
has the better candidate pool and is the better *base* retriever — so the right
answer is neither, it's A plus a re-ranker, which is exactly the shape of this
project's doc-scoped result (hybrid alone was worse than dense alone, but hybrid +
re-ranking beat both).

Second check: per-type breakdown. If B's advantage comes entirely from
single-hop questions while A wins on near-miss, the aggregate is hiding two
different systems.

</details>

---

**Previous:** [10 — Cross-encoders](10-cross-encoders.md) ·
**Next:** [12 — Building a golden set](12-building-a-golden-set.md)
