# ClauseCompass Day 3 — Explained From Scratch

The day that turns opinion into measurement. Companion to
[day1-explained.md](day1-explained.md) and [day2-explained.md](day2-explained.md).

**The mental shift:** today does not improve the system. It builds the
**instrument** that says whether tomorrow's change helped. Everything from Day 4
on is "change one thing → did the number move?" Without today, that sentence has
no meaning.

---

## Contents

1. [Part 0 — Why an eval set is the hard part](#part-0)
2. [Part 1 — The offset-mapping problem](#part-1)
3. [Part 2 — The tasks, explained](#part-2)
4. [Part 3 — Failures and fixes](#part-3)
5. [Part 4 — Glossary](#part-4)
6. [Part 5 — Interview questions](#part-5)
7. [Part 6 — Numbers and honest status](#part-6)

---

<a name="part-0"></a>
## Part 0 — Why an eval set is the hard part

Days 1 and 2 produced plenty of plausible claims: clause chunks look better,
metadata coverage is high, the contextual prefix probably helps. **Not one of
them was a measurement.** Two were actively wrong when tested — the prefix
raised the similarity floor (Day 2 F16), and 98% governing-law accuracy became
86% on held-out data (Day 2 F12).

A **golden eval set** is a fixed list of questions where someone has already
decided what the right answer is. Given one, "did retrieval improve?" becomes
arithmetic.

**Why it's hard:** the questions and the right answers have to come from
somewhere. Reading 500 contracts to write them by hand is roughly 40 hours.
This is where CUAD pays for itself — attorneys already did that labelling, and
crucially they recorded **character offsets**, so the exact position of every
answer in every contract is known.

**The trap that makes today dangerous:** a corrupted golden set does not throw
an error. It produces stable, plausible numbers that mean nothing, and every
decision for the rest of the week rests on them. So today's discipline is:
**every generated question must be independently verified, and anything that
cannot be verified gets discarded.**

---

<a name="part-1"></a>
## Part 1 — The offset-mapping problem

CUAD says: *"the governing-law answer starts at character 4,120 and is 380
characters long."* My chunks each cover a character range of that same document.
Mapping one onto the other identifies the **gold chunk** — the chunk retrieval
must find.

```
document text   [.......................................................]
CUAD answer                        ^start=4120  ────────►  end=4500
my chunks       [ c0 ][ c1 ][ c2 ][ c3        ][ c4 ][ c5 ]
                                   ^char_start=3980, char_end=4600
                                   → c3 overlaps the answer → c3 is gold
```

Overlap is arithmetic: `chunk.char_start < answer_end AND chunk.char_end > answer_start`.

An answer can straddle two chunks — then **both** are gold, and retrieving
either should count.

### The three ways this breaks silently

| Bug | Symptom | The catch |
|---|---|---|
| Offsets computed on differently-normalised text | Gold chunk doesn't contain the answer string | Assert `answer_text in chunk_text` |
| Whitespace collapsing shifted all offsets | Consistent small drift; answers land one chunk early | Spot-check by hand |
| Offsets relative to a different field | Nothing matches at all | Print 5 examples immediately |

**The rule, and it is not optional:** overlap alone is *not* enough. Overlap only
proves the arithmetic lines up. It does not prove the text survived chunking.
So `gold_for()` requires **both**:

```python
overlaps = c["char_start"] < end and c["char_end"] > start
contains = norm(answer)[:120] in norm(c["text"])
if overlaps and contains:        # both, always
    gold.append(c["chunk_id"])
```

### Why this went well — Day 2 paid for it

Measured across all 6,702 answerable annotations before writing a line of the
generator: **6,697 / 6,702 = 99%** of answers were found inside a chunk that
overlaps their span.

That is a direct dividend of Day 2's F9 fix. The chunker had been discarding
every contract preamble — 1.8M characters — leaving 16.6% of answers in no chunk
at all. Had that survived, today's golden set would have quietly lost a sixth of
its questions, and the ones about dates and parties (which live in preambles)
would have vanished disproportionately.

**Only 3 of 4,256 questions were dropped for a missing gold chunk.** When a
tutorial warns "this is where it usually breaks" and it doesn't break, that's
usually because an earlier day's bug was already found.

---

<a name="part-2"></a>
## Part 2 — The tasks, explained

### Task 2 — Generating questions, and the discovery that reshaped the day

CUAD's own questions are unusable as-is: *"Highlight the parts (if any) of this
contract related to 'Governing Law' that should be reviewed by a lawyer."* So
each of 20 categories gets a natural-language template.

Then, before building anything, one check: **can retrieval actually answer these?**

```
Question: "Which state's or country's law governs this agreement?"
```

Asked of 30 different contracts, unscoped over all 530:

| Retrieval | hit@5 | hit@50 |
|---|---|---|
| Unscoped, generic question | **0/30** | 2/30 |
| Doc-scoped, generic question | 27/30 | 30/30 |

**Zero.** And it is not a tuning problem — it is arithmetic. That question string
is *byte-identical* for all 530 contracts. It contains no information about which
contract is meant. Around 68 documents have a chunk literally headed "Governing
Law", all of them equally good matches. Asking dense retrieval to pick contract
#217's clause is asking it to guess.

The task brief expects recall@5 between 0.55 and 0.78. Built as written, the
harness would have reported ~0.00 and stayed there no matter what Day 4 changed.

**The fix: entity-qualified questions.** Name the contract in the question, the
way anyone actually asks:

```
"Under the Adams Golf Endorsement agreement, which state's or country's law governs it?"
```

Measured again, unscoped: **hit@5 4/30, hit@50 13/30**. Low, but *non-zero and
improvable* — which is the entire requirement for a baseline. And it produces a
concrete Day 4 prediction: BM25 does exact-term matching, and party names are
exact terms, so BM25 should help here far more than it helps a generic query.

Two guards went into the generator alongside it:

- **Leak guard.** If the question text would restate the answer, drop the
  question. This is why the `Document Name` and `Parties` categories are excluded
  entirely — their answer *is* the contract title or the party names, which the
  entity-qualified question already says out loud. A question that contains its
  own answer measures nothing.
- **Fragment guard.** Answers under 15 characters, or with no letters, are
  discarded — 282 of them.

**Result: 4,256 questions from 20,910 annotations.**

| Rejected | Count | Why this is healthy |
|---|---|---|
| `no_answer` | 14,208 | CUAD asks all 41 categories of every contract; most don't apply. **These are not waste** — they become Day 5's refusal tests |
| `no_template` | 2,161 | 21 categories left untemplated |
| `fragment` | 282 | quality filter |
| `no_gold_chunk` | **3** | offset mapping essentially exact |

Quality-filter rejection rate: **6.3%**.

---

### Task 3 — Sampling to 120, and reviewing them

4,256 questions is too many (slow) and too repetitive (near-duplicates). The
target is ~120 *diverse* ones, so sampling is **stratified by category**: 6 per
category × 20 categories. Without stratification, `Governing Law` (437
annotations) would swamp `Warranty Duration` (75).

Review is what makes the set "golden". Rather than skim 120 items uniformly,
automated heuristics flagged the suspicious ones:

| Flag | Count | What it looks for |
|---|---|---|
| `SHORT` | 10 | answer under 30 chars |
| `MANY_GOLD` | 7 | more than 3 gold chunks |
| `NO_CATEGORY_VOCAB` | 4 | gold chunk lacks vocabulary the category implies |
| `STARTS_LOWER` | 3 | answer begins mid-sentence |
| **clean** | **98/120** | |

All 22 flagged items were read in full, plus 12 random unflagged ones to check
that bucket too.

**Most flags were false positives, and that is worth knowing:**

- `SHORT` fired on ten dates. "February 20, 2020" *is* a complete answer to "on
  what date was it signed?" — the heuristic was wrong, not the question.
- `NO_CATEGORY_VOCAB` fired on *"no less than 50% of the Joint Venture Products
  should be sold overseas"* for Minimum Commitment. That is textbook minimum
  commitment; my keyword list just didn't include "no less than".
- `MANY_GOLD` is not a defect at all — a long annotated span legitimately crosses
  several chunks, and `recall@k` already gives partial credit.

**Two genuine deletions:**

| Question | Why deleted |
|---|---|
| GARMAN Software Licence — "on what date was it signed?" | The answer is `"____ day of May, 2000"` — an **unfilled blank**. The contract was never dated, so there is nothing to score against |
| MediWound Supply — "how long does any warranty last?" | The annotated answer is a non-conforming-product **rejection window**, which does not answer a question about warranty duration |

**Two edits rather than deletions:** answers beginning mid-sentence
(`"after Nantz Communications has received notice…"`) and mid-*word*
(`"n case of an order volume"` — the offset lands one character inside "In").
These are fine for scoring *retrieval* (the gold chunks are right) but unusable
as answer references, so they carry `answer_is_fragment: true` for Day 5 to
exclude.

**Review-stage rejection: 2/120 = 1.7%.**

That is far below the brief's ~12% expectation, and the honest reason is not that
the review was lenient — it is that **the filtering already happened upstream**.
285 questions were rejected at generation. Reporting one number would be
misleading, so both are recorded: **6.3% generation-stage, 1.7% review-stage.**

---

### Task 4 — The hard cases, which matter more than the other 118

30 hand-authored questions. This is where taste shows.

#### 4a. Unanswerable — 18 questions, and they are *verified*

The correct response is refusal. This is the hallucination stress test.

The brief says write plausible topics. But "plausible" asserted is worthless — if
one of these contracts *does* mention escrow, the question isn't unanswerable and
the test is backwards. So each topic ships with the terms that must be **absent**,
and the builder searches for a contract where none of them appear:

```python
("a source-code escrow arrangement", ["escrow"]),
("pandemic or epidemic force-majeure relief", ["pandemic", "epidemic", "covid"]),
("ITAR or export-control compliance", ["itar", "export control", "export administration"]),
```

Then, in `build_hard()`:

```python
low = docs[doc_id]["text"].lower()
if any(t in low for t in absent):
    continue                       # this contract does mention it - try another
```

**Unanswerability is a verified property, not an assumption.** An independent
check re-confirms all 18 against the document text: **0 leaks**.

Note the topics are deliberately *adjacent* to commercial contracts — GDPR,
SOC 2, biometric data, FCPA, liquidated damages. "What is the recipe for pizza?"
would test nothing.

#### 4b. Near-miss traps — 6 questions

Each asks about clause A in a contract that **also contains** clause B, where B
is the obvious dense-retrieval confusion:

| Asked | Trap |
|---|---|
| Cap On Liability | Uncapped Liability |
| Termination For Convenience | Post-Termination Services |
| IP Ownership Assignment | License Grant |
| Non-Compete | Exclusivity |
| Anti-Assignment | Change Of Control |
| Audit Rights | Insurance |

Both clauses must genuinely exist, so the builder only accepts a document with
CUAD annotations for both. The trap's chunk ids are stored as `trap_chunk_ids`,
and the runner reports whether the trap outranked the answer — a diagnostic the
brief doesn't ask for but which makes the category actionable.

#### 4c. Multi-hop — 4 questions

Two clauses needed for one answer: *"what is its term, and how much notice stops
it renewing?"* — gold is the union of both categories' chunks.

#### 4d. Aggregation — 2 questions, expected to fail

*"How many contracts in the corpus are governed by New York law?"*

No single chunk answers this. It is included **on purpose**, as a measured known
weakness that motivates the Day 6 agent. Measuring something you expect to fail,
and saying so up front, is a stronger move than quietly excluding it.

**Final set: 148 questions, 12% unanswerable, spanning 127 distinct documents.**

---

### Task 5 — The metrics, explained properly

Four metrics, all comparing the **gold chunk ids** to the **ranked list** of
retrieved ids. Worked example throughout: gold = `{a, b}`, retrieved =
`[x, a, y, z, b]`.

#### recall@k — how much of the answer did we get?

```python
len(gold & set(retrieved[:k])) / len(gold)
```

- `recall@5` → top 5 is `[x,a,y,z,b]`, contains both a and b → **1.0**
- `recall@2` → top 2 is `[x,a]`, contains only a → **0.5**

Answers *"what fraction of what I needed did I find?"*

#### hit@k — did we get anything useful at all?

```python
1.0 if gold & set(retrieved[:k]) else 0.0
```

- `hit@2` → a is there → **1.0** (even though recall@2 was only 0.5)

Binary. Often what a user actually feels: one good clause on screen is usually
enough. **This is why hit@5 (0.44) exceeds recall@5 (0.36)** — partial credit
versus any credit.

#### MRR — how high up was the first good result?

Reciprocal rank of the first gold hit, `1/rank`:

- first gold is `a` at rank 2 → **1/2 = 0.5**
- rank 1 → 1.0; rank 3 → 0.33; rank 10 → 0.1

Punishes burying the answer. Sensitive to exactly what a re-ranker fixes.

#### nDCG@k — recall, weighted by position

Each gold hit at rank *i* earns `1/log₂(i+1)`, divided by the best achievable
arrangement. Unlike MRR it accounts for *all* gold chunks, not just the first.

#### The footgun, and why there is a test for it

Every metric returns **1.0 when `gold` is empty** — "nothing was expected, so
nothing was missed". That is defensible in isolation and dangerous in aggregate:
the 18 unanswerable and 2 aggregation questions have no gold chunks, so including
them would hand out **20 free 1.0s** and inflate every score.

So the runner filters on `n_gold > 0`, reports `n_excluded_no_gold: 20`, and
`tests/test_metrics.py` pins the behaviour explicitly:

```python
def test_empty_gold_is_free_score():
    """Documents the footgun: no gold means 1.0, so the runner must filter these."""
    assert recall_at_k(set(), [], 5) == 1.0
```

A test that documents a hazard is worth more than one that re-checks arithmetic.
**8 tests, all passing.**

---

### Task 6 — The runner, and why there are two configs

The runner embeds each question, retrieves top-50, computes all four metrics,
and writes a timestamped JSON with the git SHA — so a number can always be traced
back to the code that produced it.

Two configs, because the honest answer depends on the product:

| Config | What it does | Product it represents |
|---|---|---|
| `corpus_wide` | search all 18,280 chunks | "ask my whole contract library" |
| `doc_scoped` | filter to the question's `doc_id` first | per-contract Q&A |

`doc_scoped` is also an **upper bound**: it shows what retrieval could reach if
document identification were perfect.

---

### Task 7 — The baseline, and the diagnosis

```
                recall@5   recall@10   recall@50    hit@5     MRR    nDCG@5   p95
corpus_wide      0.3632      0.4526      0.5849    0.4375   0.3148   0.2892  39.9ms
doc_scoped       0.5751      0.7107      0.9789    0.6797   0.4421   0.4246  36.6ms
```

By type:

| Type | n | corpus recall@5 | corpus recall@50 | scoped recall@5 | scoped recall@50 |
|---|---|---|---|---|---|
| single_hop | 118 | 0.356 | 0.579 | 0.577 | 0.977 |
| near_miss | 6 | 0.333 | 0.417 | 0.500 | 1.000 |
| multi_hop | 4 | 0.625 | 1.000 | 0.625 | 1.000 |

`doc_scoped` recall@5 of 0.575 sits inside the brief's healthy 0.55–0.78 band.
`near_miss` at 0.33–0.50 sits inside its 0.30–0.50 band — the traps are trapping,
but only mildly: **1 of 6 pulled the trap into top-5, and 0 of 6 beat the answer.**

### The recall@5 vs recall@50 gap — the whole point of today

**The two configs give two different diagnoses, and that is the interesting part.**

**`doc_scoped`: gap of 0.404** (0.575 → 0.979). Nearly every gold chunk *is*
retrieved in the top 50 — it's just ranked badly. This is textbook
**re-ranking territory**. A cross-encoder reorders a shortlist it did not have to
find, and here the shortlist already contains the answer 98% of the time.

> **Prediction for Day 4:** the cross-encoder re-ranker lifts `doc_scoped`
> recall@5 substantially, toward a ceiling around 0.98. Re-ranking cannot exceed
> recall@50, because it only reorders what was already retrieved.

**`corpus_wide`: recall@50 is only 0.585.** So **41% of gold chunks are never
retrieved at all**, even at depth 50. Re-ranking is powerless there — you cannot
reorder a chunk you never fetched.

> **Prediction for Day 4:** BM25 helps `corpus_wide` more than re-ranking does,
> because the questions name parties and dense embeddings are weak on proper
> nouns while exact-term matching is strong on them. The number to watch is
> `corpus_wide` **recall@50** — if hybrid retrieval doesn't move *that*, the
> ceiling doesn't move either.

One sentence: **doc_scoped is a ranking problem, corpus_wide is a retrieval
problem, and they need different fixes.** Tomorrow tests whether that's right.

---

<a name="part-3"></a>
## Part 3 — Failures and fixes

### F18 — Generic questions make corpus-wide eval mathematically impossible ✅ FIXED before building

**Symptom.** hit@5 = 0/30 for unscoped retrieval on the brief's question format.

**Cause.** The question string is identical across all 530 contracts. There is no
signal identifying the target document, and ~68 documents hold an equally good
"Governing Law" chunk.

**Why it mattered so much.** The brief's runner searches unscoped. Built as
written, the baseline would have read ~0.00 and been *insensitive to every Day 4
change* — BM25, RRF and re-ranking would all have moved a number pinned at the
floor, and the week's central claim ("I measured the improvement") would have
been unsupportable.

**Fix.** Entity-qualified questions (0/30 → 4/30 hit@5; 2/30 → 13/30 hit@50) plus
two reported configs.

**Found by** checking feasibility *before* writing the generator — the cheapest
30 lines of the day.

> **Interview line:** "Before building the harness I checked whether the
> questions were answerable at all, and found the template format made
> corpus-wide retrieval mathematically impossible — hit@5 was zero out of thirty,
> because the question text was byte-identical across all 530 contracts. I
> re-posed the questions to name the contract and reported both a corpus-wide and
> a document-scoped config. Had I skipped that check I'd have spent Day 4
> optimising against a number that couldn't move."

---

### F19 — Empty-gold questions would have handed out 20 free points ✅ FIXED

Every metric returns 1.0 for empty gold. 18 unanswerable + 2 aggregation
questions have no gold chunks. Averaging naively over all `answerable`
questions — which is what the brief's runner does — would have inflated every
metric with 20 free 1.0s.

**Fix.** Aggregate over `n_gold > 0`, report `n_excluded_no_gold`, and pin the
hazard in a unit test.

**Scale of the error avoided:** 20 of 148 questions. On `corpus_wide` recall@5
that is the difference between the true **0.3632** and a flattering ~0.45.

---

### F20 — My review decisions silently stopped applying ✅ FIXED

**Symptom.** `reviewed 120 -> kept 120, deleted 2` — self-contradictory. 120
kept *and* 2 deleted cannot both be true.

**Cause.** Review decisions were keyed on question ids (`g0755`). Then I made
`stratify()` renumber the sample `g0001…g0120`, so the keys referenced ids that
no longer existed. Both deletions matched nothing and were skipped.

**Cause underneath the cause:** the ids were never stable. They are positional,
and any change to generation or sampling renumbers everything.

**Fix.** Key decisions on `doc_id|category`, which is stable by construction —
plus a guard that **fails the build** when a decision matches nothing:

```python
if (hit["delete"], hit["flag"]) != want:
    raise SystemExit(f"review keys stale: applied {hit['delete']} deletes / "
                     f"{hit['flag']} flags, expected {want[0]}/{want[1]}")
```

**Why this one is worth telling.** It is the same failure class as Day 1's F8 and
Day 2's F9 — *silent*. The set would have shipped with two known-bad questions
in it, and nothing would have complained. Note also that the print statement is
what exposed it: printing both numbers made the contradiction visible. **A log
line that can contradict itself is a cheap assertion.**

---

<a name="part-4"></a>
## Part 4 — Glossary

| Term | Definition |
|---|---|
| **Golden set** | Fixed questions with known correct answers; the measuring instrument |
| **Gold chunk** | The chunk retrieval is supposed to find for a question |
| **Answer offset** | Character position of an answer inside the document — what makes CUAD automatable |
| **recall@k** | Fraction of gold chunks appearing in the top k |
| **hit@k** | 1 if *any* gold chunk is in the top k, else 0 |
| **MRR** | Mean Reciprocal Rank — 1/(rank of first gold hit), averaged |
| **nDCG@k** | Recall weighted by position, normalised by the ideal ordering |
| **Stratified sampling** | Sample per category so no category dominates |
| **Unanswerable question** | Correct response is refusal; the hallucination test |
| **Near-miss trap** | A legally adjacent clause that is the plausible wrong answer |
| **Ablation** | Change one component, hold everything else, measure the delta |
| **Ceiling** | recall@50 bounds what re-ranking can achieve; it only reorders |
| **Dev vs held-out** | A number tuned against is not a measurement (Day 2 F12) |

---

<a name="part-5"></a>
## Part 5 — Interview questions

**"How do you know your system works?"**
> I have a 148-question golden eval set with per-question gold chunk ids, and a
> runner that reports recall@k, hit@k, MRR and nDCG with the git SHA of the code
> that produced them. Dense baseline is recall@5 0.36 corpus-wide and 0.58
> document-scoped. Before that existed I couldn't answer this question, and I'd
> already been wrong twice about changes I assumed were improvements.

**"Where did the questions come from?"**
> 118 are template-generated from CUAD's attorney-reviewed annotations, using
> character offsets mapped onto my chunk ranges — and only kept when the chunk
> both overlaps the span *and* contains the answer text. 30 are hand-authored
> hard cases: 18 unanswerable, 6 near-miss traps, 4 multi-hop, 2 aggregation.
> Everything was reviewed, and the review was done by me with AI assistance, not
> by an attorney — the file records that.

**"Your recall@5 is 0.36. Isn't that bad?"**
> It's a dense-only baseline on the hard config, and it's deliberately the hard
> config. The diagnostic that matters is the gap to recall@50. Document-scoped,
> recall@50 is 0.98 against recall@5 of 0.58 — the answer is retrieved and
> mis-ranked, which is a re-ranking problem. Corpus-wide, recall@50 is only 0.58,
> so 41% is never retrieved at all and re-ranking can't touch it — that needs
> hybrid retrieval. Same system, two different bottlenecks.

**"What's the most interesting bug you found?"**
> Before building the harness I checked whether the questions were answerable and
> got hit@5 of zero out of thirty. The template questions were byte-identical
> across all 530 contracts, so no retriever could know which contract was meant.
> The brief expected 0.55 to 0.78. I'd have built an instrument that read zero and
> stayed zero through every improvement I made afterwards.

**"How do you know your unanswerable questions are actually unanswerable?"**
> Each one ships with the terms that must be absent, and the builder only assigns
> it to a contract where none of them appear in the full text. A separate
> validation re-checks all 18 — zero leaks. If I'd just written plausible-sounding
> topics, some would have been answerable and I'd have been scoring refusals as
> correct when they were wrong.

**"What would you do differently?"**
> Build this on Day 1. Days 1 and 2 produced claims I couldn't check, and two of
> them were wrong. Also: my review decisions initially keyed on unstable question
> ids and silently stopped applying — I'd key on content-derived identifiers from
> the start, and assert that every recorded decision actually matched something.

---

<a name="part-6"></a>
## Part 6 — Numbers and honest status

### Record these

| Metric | Value |
|---|---|
| Golden set size | **148** questions |
| Composition | 118 single_hop, 18 unanswerable, 6 near_miss, 4 multi_hop, 2 aggregation |
| % unanswerable | **12%** |
| Distinct documents covered | 127 |
| Generation-stage rejection | **6.3%** |
| Review-stage rejection | **1.7%** (2 of 120) |
| **Baseline recall@5 (corpus_wide)** | **0.3632** |
| Baseline recall@50 (corpus_wide) | 0.5849 |
| Baseline MRR (corpus_wide) | 0.3148 |
| **Baseline recall@5 (doc_scoped)** | **0.5751** |
| Baseline recall@50 (doc_scoped) | 0.9789 |
| Baseline MRR (doc_scoped) | 0.4421 |
| p95 retrieval latency | 39.9 ms / 36.6 ms |

**Write the "before" numbers down now.** Tomorrow's will be higher and the
temptation to forget these is real.

### Honest limitations — put these in the README

- **CUAD-shaped.** 118 of 148 questions come from CUAD's 41 annotation
  categories, so the set reflects what attorneys chose to annotate, not the full
  space of contract questions. 20 categories are templated; 21 are not.
- **AI-reviewed, not attorney-verified.** CUAD's *clause spans* are
  attorney-reviewed. The sampling review, the 30 hard cases and the review
  decisions are mine. `review_decisions.json` records this in `_provenance`.
- **Entity-qualified questions are an intervention.** Real users would not always
  name the contract. The corpus-wide number measures retrieval given a named
  contract, which is easier than a truly cold query and harder than doc-scoped.
- **Small hard-case samples.** near_miss n=6, multi_hop n=4, aggregation n=2.
  Directional only — do not quote a per-type number from n=4 as a finding.
- **Traps are mild.** 0 of 6 near-miss traps beat the gold chunk, so the category
  is not yet discriminating much. Worth strengthening if Day 4 shows no movement.

### Status of every failure so far

| # | Failure | Status |
|---|---|---|
| F1–F2 | Fabricated citations, cross-contract splice | ✅ Day 1 |
| F3 | Scores can't express relevance | ⚠️ **now measurable** — Day 4 |
| F4 | Citations sometimes missing | ⚠️ detected → Day 5 |
| F5 | Near-duplicates waste top-k | ✅ improved Day 2 |
| F6 | Single-doc question, many-doc corpus | ✅ `--doc` scoping |
| F7 | Chunks cut mid-clause | ✅ 13% → 62% clause starts |
| F8 | Ollama truncated 60% of the prompt | ✅ Day 1 |
| F9 | Preamble dropped, 16.6% answers unreachable | ✅ Day 2 — **paid off today** |
| F10–F11 | Governing-law extraction errors | ✅ Day 2 |
| F12 | Accuracy overfit to its own dev set | ✅ held-out 86% quoted |
| F13 | "Reply NONE" made the model refuse everything | ✅ Day 2 |
| F14–F15 | Loader dropped a contract; MAX_CHARS unenforced | ✅ Day 2 |
| F16 | Contextual prefix worsens the similarity floor | ⚠️ **ablate it tomorrow** — now possible |
| F17 | Good retrieval, refused answer | ✅ Day 2 |
| F18 | Generic questions make corpus-wide eval impossible | ✅ **Day 3** |
| F19 | Empty-gold questions inflate every metric | ✅ **Day 3** |
| F20 | Review decisions silently stopped applying | ✅ **Day 3** |

### The theme, three days running

**Silent failures dominate.** F8, F9, F14, F18, F19, F20 — six of twenty — threw
no error, printed no warning, and produced plausible output. Every one was caught
by an **accounting check**: do the totals reconcile? Does this number contradict
that one? Would this metric be 1.0 for the wrong reason?

That is the transferable skill from this project, and it is worth more than any
individual number in the table above.

### Tomorrow

Day 4 adds BM25, fuses with Reciprocal Rank Fusion, adds the cross-encoder
re-ranker, and runs the ablation against today's baseline. Three things are now
testable that weren't yesterday:

1. Does re-ranking lift `doc_scoped` recall@5 toward its 0.98 ceiling?
2. Does BM25 lift `corpus_wide` **recall@50** above 0.585? If not, the ceiling
   holds and nothing downstream can help.
3. Does the Day 2 contextual prefix actually help (F16)? Ablate it now that a
   number exists to move.
