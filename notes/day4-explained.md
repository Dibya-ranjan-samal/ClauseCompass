# ClauseCompass Day 4 — Explained From Scratch

Hybrid retrieval and re-ranking, measured against Day 3's instrument. Companion
to [day1](day1-explained.md), [day2](day2-explained.md), [day3](day3-explained.md).

**The discipline:** change one thing, run the eval, write down the number. I
broke that rule once today and it cost an extra 20-minute run to untangle — the
story is in F23.

**The headline, and it is not the one the brief predicted:** the winning
configuration **differs by scope**. Document-scoped retrieval improved a lot
(recall@5 0.575 → **0.762**, MRR 0.442 → **0.657**). Corpus-wide retrieval
**did not improve at all** — nothing beat the plain dense baseline on recall@5.

---

## Contents

1. [Part 0 — What we are adding and why](#part-0)
2. [Part 1 — The three techniques, explained](#part-1)
3. [Part 2 — The ablation](#part-2)
4. [Part 3 — Failures and fixes](#part-3)
5. [Part 4 — Glossary](#part-4)
6. [Part 5 — Interview questions](#part-5)
7. [Part 6 — Numbers and honest status](#part-6)

---

<a name="part-0"></a>
## Part 0 — What we are adding and why

Day 3 produced a diagnosis, and today tests it. That distinction matters: these
were written down *before* any of today's runs.

> **Day 3 prediction 1.** `doc_scoped` has a 0.404 gap between recall@5 (0.575)
> and recall@50 (0.979). The chunk is retrieved and mis-ranked, so re-ranking
> should lift recall@5 toward a ceiling near 0.98.
>
> **Day 3 prediction 2.** `corpus_wide` recall@50 is only 0.585, so 41% of gold
> is never retrieved. Re-ranking cannot help. BM25 should raise that ceiling,
> because the questions name parties and lexical matching is strong on proper
> nouns where dense embeddings are weak.

**Both were directionally right, and prediction 2 was right for the wrong
reason** — BM25 did raise the ceiling (0.585 → 0.608), but by far less than
hoped, and its ranking was bad enough to *lower* recall@5.

---

<a name="part-1"></a>
## Part 1 — The three techniques, explained

### BM25 — lexical retrieval

Dense embeddings compare *meaning*. BM25 compares *words*. It scores a document
by how many query terms it contains, weighted so that

- **rare terms count more** (IDF — "indemnification" is more informative than "the"),
- **repetition saturates** (the 10th "liability" adds little over the 3rd),
- **long documents are penalised**, so they don't win by sheer size.

**Why it should complement dense retrieval:** dense models blur proper nouns.
"Adams Golf" and "Bravatek Solutions" are near-identical to an embedding model —
both are just *company-shaped*. BM25 treats them as distinct tokens.

**Tokenization matters more than it looks.** Legal text is full of hyphenated
compounds, so the tokenizer keeps them intact:

```python
TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
```

That is a genuine two-edged trade, and measuring it took one minute:

| Query | Top BM25 score | Why |
|---|---|---|
| `non-compete` | 7.77 ✅ | contracts write it hyphenated |
| `most favored nation` | 22.79 ✅ | contracts write it unhyphenated |
| `most-favored-nation` | **0.00** ❌ | one token, appears nowhere in the corpus |

A hyphenated query term absent from the corpus scores **exactly zero** — BM25
has no partial credit. Splitting compounds into parts as well would fix this;
it was left alone because no scored question depends on it, and the aggregation
question that mentions MFN has no gold chunks. **Recorded rather than fixed,
because the eval cannot see the difference.**

### Reciprocal Rank Fusion — merging two ranked lists

Dense scores are cosine similarities around 0.8. BM25 scores are unbounded
(13.85 for "force majeure"). **You cannot average them** — different scales,
no shared meaning. RRF throws the scores away and uses only **rank**:

```
score(chunk) = Σ over lists  1 / (k + rank_in_that_list)
```

A chunk at rank 2 in both lists scores `1/62 + 1/62 = 0.032`, beating one at
rank 1 in a single list (`1/61 = 0.016`). **Agreement between retrievers beats a
strong opinion from one.**

`rrf_k` controls how steeply rank matters — and it is worth understanding that
it does *not* change the agreement rule:

> Two rank-2 hits always beat one rank-1 hit, because `2/(k+2) > 1/(k+1)` for
> every `k ≥ 0`. What `k` changes is how much a *better* rank is worth: rank 1
> over rank 5 is worth 3× at `k=1` and only 1.06× at `k=60`.

I originally wrote a unit test asserting that a small `k` would let a lone
rank-1 hit win. It failed, and the algebra above is why — **the test caught my
own misunderstanding**, which is what tests on pure logic are for.

### Cross-encoder re-ranking

| | Bi-encoder (retrieval) | Cross-encoder (re-ranking) |
|---|---|---|
| Input | query and chunk **separately** | the **pair**, together |
| Sees the interaction? | no | yes |
| Precomputable? | yes — embed the corpus once | no — per query-chunk pair |
| Cost here | 37 ms p95 | **~12,000 ms p95** |

The critical property: **a re-ranker only reorders what retrieval already
found.** It can never exceed the base retriever's recall@candidates. That makes
`recall@50` a hard ceiling *and* a built-in correctness check:

> If `recall@50` changes after re-ranking, you have a bug — you are dropping or
> adding candidates, not reordering them.

Mine held exactly (0.6044 → 0.6044, 0.9867 → 0.9867) in every run, which is why
I trust the rest of the numbers.

---

<a name="part-2"></a>
## Part 2 — The ablation

### Task 0 — the refactor gate

Everything implements one interface:

```python
class Retriever(Protocol):
    name: str
    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]: ...
```

`doc_id` is part of the interface rather than a separate code path, because Day 3
showed the two scopes have different bottlenecks — every retriever has to support
both or neither prediction can be tested.

**The gate:** re-run the dense baseline through the new interface and demand the
*same numbers as yesterday*.

| | recall@5 | recall@50 | MRR | nDCG@5 |
|---|---|---|---|---|
| Day 3 corpus_wide | 0.3632 | 0.5849 | 0.3148 | 0.2892 |
| Day 4 refactored | **0.3632** | **0.5849** | **0.3148** | **0.2892** |

Identical to four decimals, both scopes. A refactor that changes behaviour
invalidates every comparison after it, so this is not a formality.

### The headline tables

**`corpus_wide` — search all 18,280 chunks**

| Configuration | R@5 | R@10 | R@50 | MRR | nDCG@5 | p95 ms |
|---|---|---|---|---|---|---|
| **dense_baseline** | **0.363** | 0.453 | 0.585 | **0.315** | **0.289** | **37** |
| bm25_only | 0.223 | 0.317 | 0.547 | 0.207 | 0.172 | 58 |
| hybrid_rrf | 0.338 | 0.431 | **0.604** | 0.276 | 0.252 | 94 |
| hybrid_rerank | 0.331 | 0.472 | 0.604 | 0.263 | 0.243 | 11,080 |
| hybrid_rerank + strip entity | 0.158 | 0.249 | 0.604 | 0.136 | 0.106 | 11,750 |

**Nothing beat the baseline.** The only thing that improved was the ceiling
(recall@50, +0.019 from BM25) — and 300× the latency bought a *worse* top-5.

**`doc_scoped` — filter to the named contract first**

| Configuration | R@5 | R@10 | R@50 | MRR | nDCG@5 | p95 ms |
|---|---|---|---|---|---|---|
| dense_baseline | 0.575 | 0.711 | 0.979 | 0.442 | 0.425 | 29 |
| bm25_only | 0.411 | 0.632 | 0.980 | 0.319 | 0.285 | 56 |
| hybrid_rrf | 0.481 | 0.686 | 0.987 | 0.394 | 0.356 | 87 |
| hybrid_rerank | 0.606 | 0.818 | 0.987 | 0.435 | 0.431 | 12,046 |
| **hybrid_rerank + strip entity** | **0.762** | **0.857** | 0.987 | **0.657** | **0.646** | 11,599 |

**+0.187 recall@5 and +0.215 MRR over the baseline**, against a 0.987 ceiling.

### Per-type recall@5 — where the story lives

| Configuration | single_hop | near_miss | multi_hop |
|---|---|---|---|
| dense_baseline-corpus_wide | 0.356 | 0.333 | 0.625 |
| bm25_only-corpus_wide | 0.219 | **0.250** | 0.312 |
| dense_baseline-doc_scoped | 0.577 | 0.500 | 0.625 |
| bm25_only-doc_scoped | 0.421 | **0.250** | 0.354 |
| hybrid_rerank+strip-doc_scoped | 0.756 | **0.917** | 0.708 |

**The brief predicted BM25 would beat dense on near-miss questions. It did not —
it lost, in both scopes (0.250 vs 0.333 and 0.500).** That is the day's cleanest
negative result, and the mechanism makes sense in hindsight: a near-miss trap is
*legally adjacent* language, so the trap clause shares vocabulary with the
question. "What is the monetary cap on liability?" lexically matches an uncapped-
liability clause about as well as a cap clause. Exact-term matching cannot
separate near-synonyms; that is precisely what it is bad at.

**What did solve near-miss: the cross-encoder, at 0.917.** It reads the
query-clause pair jointly, so it can tell "liability is capped at $X" from
"liability is uncapped for gross negligence". That is the correct tool, and the
data says so.

### Parameter sweep

| Knob | Values tried | Result |
|---|---|---|
| `rrf_k` | 10, 20, 60, 120 | **10 best** (0.356 vs 0.338); 20/60/120 identical |
| `bm25_candidates` | 5, 10, 25, 50 | monotonic — **more is better** (0.297 → 0.356) |

The BM25-candidate result contradicts the brief, which suggests *reducing* BM25
candidates when hybrid underperforms ("BM25 noise crowding out good hits").
Measured here, restricting BM25 makes things **worse at every step**. The problem
was never that BM25 contributed too many candidates; it is that its *ordering*
is poor, which is a different disease and needs a different cure.

**Not run: the `rerank_candidates` sweep (20/50/100).** Three cross-encoder runs
at ~20 minutes each, to tune a component that is a net *loss* on the headline
scope and already near its ceiling on the other. Skipped deliberately, recorded
here rather than quietly omitted.

### Task 5 — the contextual-prefix ablation (Day 2's F16, finally settled)

Day 2 added `Contract type: … | Section 7.2 Indemnification | Governing law: …`
before each chunk at embed time, and measured that it **raised** mean pairwise
chunk similarity by 0.022 — chunks became harder to tell apart. Whether that cost
retrieval quality was never tested. A second collection without prefixes settles it:

| Scope | with prefix | without | verdict |
|---|---|---|---|
| corpus_wide R@5 | **0.3632** | 0.3443 | prefix **helps** +0.019 |
| corpus_wide R@50 | **0.5849** | 0.5582 | prefix **helps** +0.027 |
| doc_scoped R@5 | 0.5751 | **0.5842** | prefix **hurts** −0.009 |
| doc_scoped MRR | **0.4421** | 0.4271 | prefix helps +0.015 |

**Kept.** It helps where document identification is the hard part and is roughly
neutral where the document is already known. Note this is the *same shape* as the
entity-qualifier result below — and that is not a coincidence.

---

<a name="part-3"></a>
## Part 3 — Failures and fixes

### F21 — The unifying finding: document identification vs clause relevance

Not a bug — the day's actual insight, arrived at by being wrong first.

Three independent measurements point the same way:

| Signal | corpus_wide | doc_scoped |
|---|---|---|
| Entity qualifier in rerank query | **+0.173** (removing it costs this) | **−0.172** (removing it gains this) |
| Contextual prefix at embed time | +0.019 | −0.009 |
| BM25 lexical matching | raises ceiling +0.019 | +0.001 |

**Retrieval over a many-document corpus is two problems wearing one coat:**

1. **Which document?** Helped by identity signals — party names, contract type,
   lexical proper-noun matching.
2. **Which clause within it?** Helped by *removing* those signals, because they
   compete for attention with the clause semantics that actually matter.

`doc_scoped` solves problem 1 with a payload filter, so every identity signal
becomes pure noise. `corpus_wide` must solve both at once with one query string,
and the two needs pull in opposite directions.

This is why `strip_entity_for_rerank` is documented in `config.py` as
**scope-dependent and unsafe to set globally**.

> **Interview line:** "My cross-encoder made corpus-wide retrieval worse and
> document-scoped retrieval much better, from the same code change. The eval
> questions name the contract — 'Under the Adams Golf agreement, …' — and only
> 21% of gold clauses contain that party name. Corpus-wide, the cross-encoder
> needs the name to pick the right document. Document-scoped, the filter already
> did that, so the name is noise competing with clause semantics. Stripping it
> was worth +0.17 recall@5 in one scope and −0.17 in the other."

---

### F22 — I predicted the wrong sign, twice

**Prediction A:** stripping the entity qualifier would help, because only 21% of
gold clauses contain the party name so it looked like pure noise.
**Reality:** it helped doc-scoped (+0.172) and *destroyed* corpus-wide (−0.173),
taking recall@5 to 0.158 — worse than BM25 alone.

**Prediction B:** the windowing fix would be the important one. 36% of chunks
exceed the re-ranker's ~1,536-char window and **20% still overflow after being
split in half**, so my Day 2 halving was provably insufficient. Fixing it looked
like the substantive change.

**Reality, after the attribution run:**

| Arm | R@5 | MRR |
|---|---|---|
| rerank, halves only, keep entity | 0.6062 | 0.4351 |
| + full windowing (keep entity) | 0.5906 | 0.4796 |
| + windowing + strip entity | **0.7622** | **0.6567** |
| **windowing alone contributes** | **−0.0156** | +0.0445 |
| **stripping entity contributes** | **+0.1716** | +0.1771 |

The fix I was confident about was **slightly negative** on recall@5. The fix I
had backwards was worth **eleven times more** than the one I believed in.

Kept anyway, because it is +0.04 MRR and it removes a provable defect — but it is
labelled honestly in `config.py` as "~neutral on recall@5".

---

### F23 — I broke my own one-change-at-a-time rule

I shipped windowing and entity-stripping in the same commit and ran the eval
once. The result was unambiguous (0.762 doc-scoped) and **completely
unattributable** — two changes, one number.

Untangling it cost one extra 20-minute run with `--no-strip-entity`. Without it I
would have written "fixing re-ranker truncation was worth +19 points", which the
data says is false.

**The cost of the shortcut was larger than the discipline would have been**, and
the failure mode is that a wrong causal story is *more* damaging than no story —
it survives into a README, a CV bullet, and an interview answer.

---

### F24 — A parameter sweep where the parameter did nothing

`bm25cand10` and `bm25cand25` returned **byte-identical metrics** to `rrf_k60` —
0.3382 / 0.6044 / 0.2755 across all three. Four decimal places of coincidence is
not coincidence.

**Cause.** My own guard:

```python
per = max(self.per_k, k)          # k=50 from the runner
d = self.dense.search(query, per, doc_id)
s = self.sparse.search(query, per, doc_id)
```

With `k=50`, `max(10, 50) = 50`. The knob was clamped away by a line I added so
fusion could always fill `k` slots. **The sweep ran, reported, and measured
nothing.**

**Fix — make it asymmetric, since the two retrievers need different things:**

```python
d = self.dense.search(query, max(self.per_k, k), doc_id)   # must fill k
s = self.sparse.search(query, self.per_k, doc_id)          # the knob
```

Re-run gave a clean monotonic curve (0.297 → 0.304 → 0.354 → 0.356).

**How it was caught:** identical numbers across supposedly different configs.
Same class as Day 3's F20 ("kept 120, deleted 2") — **the output contradicted
itself, and reading it carefully was the whole detection mechanism.** A sweep
that reports suspiciously stable results is not stable, it is broken.

---

<a name="part-4"></a>
## Part 4 — Glossary

| Term | Definition |
|---|---|
| **BM25** | Lexical ranking by weighted query-term overlap; IDF-weighted, saturating, length-normalised |
| **IDF** | Inverse document frequency — rare terms are more informative |
| **Sparse vs dense** | Sparse matches words; dense matches meaning |
| **RRF** | Reciprocal Rank Fusion — merge ranked lists by `Σ 1/(k+rank)`, ignoring incomparable scores |
| **`rrf_k`** | How steeply rank matters; small `k` sharpens rank differences |
| **Cross-encoder** | Scores the query-passage pair jointly; accurate, not precomputable |
| **Re-ranking ceiling** | Re-ranking only reorders, so `recall@candidates` bounds it |
| **Ablation** | Change one component, hold the rest, measure the delta |
| **Attribution** | Determining *which* of several changes produced an effect |
| **Refactor gate** | Demanding identical numbers after a behaviour-neutral refactor |
| **Negative result** | A measured non-improvement. Publishable, and cheaper than a wrong belief |

---

<a name="part-5"></a>
## Part 5 — Interview questions

**"What did hybrid retrieval and re-ranking buy you?"**
> Document-scoped, recall@5 went 0.575 to 0.762 and MRR 0.442 to 0.657, against a
> 0.987 ceiling. Corpus-wide, honestly, nothing — plain dense stayed the best on
> recall@5 at 0.363. BM25 raised the candidate ceiling from 0.585 to 0.608 but
> ranked badly enough that fusion lowered the top-5. I kept the negative result
> because it's the more useful half.

**"Your re-ranker made one config worse. Why ship it?"**
> I didn't ship it there. `config.py` records a different winner per scope, with
> the reason. Corpus-wide the cross-encoder is a 300× latency cost for a worse
> top-5, because it's reordering a candidate pool that's only 60% correct — you
> can't rank your way out of a recall problem.

**"What was the most interesting thing you found?"**
> That corpus-wide retrieval is two problems at once. The questions name the
> contract, and only 21% of gold clauses contain that party name. Corpus-wide the
> cross-encoder needs the name to identify the document; document-scoped the
> filter already did that, so the name competes with clause semantics. The same
> one-line change was worth +0.17 recall@5 in one scope and −0.17 in the other.
> The contextual prefix from Day 2 showed the same pattern, which is what
> convinced me it was a real mechanism and not noise.

**"Did BM25 help with the near-miss traps like you expected?"**
> No, and that was the prediction I was most confident about. BM25 scored 0.250 on
> near-miss against dense's 0.333 — it lost in both scopes. In hindsight it's
> obvious: a near-miss trap is legally adjacent language, so it shares vocabulary
> with the question. Lexical matching can't separate "cap on liability" from
> "uncapped liability". The cross-encoder could, at 0.917.

**"How do you know your re-ranker isn't dropping candidates?"**
> recall@50 is invariant under re-ranking by construction, so I assert it. It was
> 0.6044 before and after corpus-wide, 0.9867 before and after doc-scoped. If it
> had moved, I'd have had a bug rather than a result.

**"Tell me about a mistake you made today."**
> I changed two things in the re-ranker at once — a truncation fix and a query
> fix — and got a large unattributable gain. One extra run showed the truncation
> fix I was confident about was worth −0.016 recall@5, and the query fix I'd had
> backwards was worth +0.172. If I'd shipped the combined result I'd have written
> a confident, wrong explanation into my README.

---

<a name="part-6"></a>
## Part 6 — Numbers and honest status

### The CV numbers

| Configuration | recall@5 | recall@50 | MRR | p95 ms |
|---|---|---|---|---|
| Dense only, corpus-wide (Day 3 baseline) | 0.3632 | 0.5849 | 0.3148 | 37 |
| BM25 only, corpus-wide | 0.2229 | 0.5467 | 0.2070 | 58 |
| Hybrid RRF, corpus-wide | 0.3382 | 0.6044 | 0.2755 | 94 |
| **Best corpus-wide (still dense)** | **0.3632** | 0.5849 | 0.3148 | 37 |
| Dense only, doc-scoped (Day 3 baseline) | 0.5751 | 0.9789 | 0.4421 | 29 |
| **Best doc-scoped (hybrid + rerank + strip)** | **0.7622** | 0.9867 | **0.6567** | 11,599 |

**The defensible CV bullet:**
> *"Built a 148-question golden eval set from attorney-annotated contract spans,
> then improved document-scoped retrieval recall@5 from 0.58 to 0.76 (MRR 0.44 to
> 0.66) with hybrid BM25/dense fusion and cross-encoder re-ranking — and measured
> that the same pipeline gave no gain on corpus-wide retrieval, which the recall
> ceiling showed was a retrieval problem rather than a ranking one."*

The second clause is what makes the first one credible.

### Do not claim

- ❌ "Hybrid retrieval improved recall" — it improved the *ceiling* (recall@50),
  and lowered recall@5 in both scopes.
- ❌ "BM25 recovered the near-miss cases" — it lost to dense on near-miss. The
  cross-encoder recovered them (0.917).
- ❌ "recall@5 0.68 → 0.84" — that is the brief's illustrative example, not a
  measurement of this system.
- ❌ "Fixed re-ranker truncation for a big gain" — worth −0.016 recall@5;
  the query fix carried it.
- ❌ Any single global "best config" — the winner differs by scope.

### Status of every failure so far

| # | Failure | Status |
|---|---|---|
| F1–F2 | Fabricated citations, cross-contract splice | ✅ Day 1 |
| F3 | Scores can't express relevance | ✅ **Resolved** — cross-encoder gives a usable signal doc-scoped |
| F4 | Citations sometimes missing | ⚠️ detected → **Day 5** |
| F5 | Near-duplicates waste top-k | ✅ Day 2 |
| F6 | Single-doc question, many-doc corpus | ✅ Day 2 — and today showed *why* it matters (F21) |
| F7 | Chunks cut mid-clause | ✅ Day 2 |
| F8 | Ollama truncated 60% of the prompt | ✅ Day 1 |
| F9 | Preamble dropped, answers unreachable | ✅ Day 2 |
| F10–F15 | Metadata extraction and loader bugs | ✅ Day 2 |
| F16 | **Contextual prefix unmeasured** | ✅ **Day 4** — kept; helps corpus-wide, neutral doc-scoped |
| F17 | Good retrieval, refused answer | ✅ Day 2 |
| F18 | Generic questions make corpus-wide eval impossible | ✅ Day 3 |
| F19 | Empty-gold questions inflate metrics | ✅ Day 3 |
| F20 | Review decisions silently stopped applying | ✅ Day 3 |
| F21 | **Identity signals help retrieval, hurt re-ranking** | ✅ **Day 4** — scope-dependent config |
| F22 | Predicted the wrong sign twice | ✅ measured and attributed |
| F23 | Two changes, one number | ✅ untangled with an extra run |
| F24 | Sweep parameter clamped to a no-op | ✅ **Day 4** — asymmetric fetch |

### Open, going into Day 5

1. **Corpus-wide recall@50 is 0.604 and that is the binding constraint.** 40% of
   gold chunks are never retrieved. No amount of ranking work can fix it — it
   needs better chunking, query rewriting, or two-stage document-then-clause
   retrieval. **That is the highest-value remaining work in the project.**
2. **12-second p95 is not shippable** for the doc-scoped winner. Options: a
   smaller cross-encoder, fewer candidates, ONNX/quantisation, or GPU.
3. **BM25 tokenization** drops hyphenated query terms absent from the corpus
   (`most-favored-nation` → 0.00). Harmless to the current eval; a real bug.
4. **near_miss n=6, multi_hop n=4.** The 0.917 near-miss result rests on six
   questions. Directional.

### Tomorrow

Day 5 is grounded generation: structured answers with validated citations,
faithfulness scoring, and refusal accuracy on the 18 unanswerable questions.
Today fixed *finding* the right clause, document-scoped. Tomorrow is about the
model not lying about what it found — and F4 (missing citations, detected on
Day 1 but never enforced) finally gets closed.
