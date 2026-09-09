# 12 — Building a Golden Set

The module with the least glamour and the highest leverage. Every number in
modules 06–18 is only as trustworthy as this.

---

## 1. What it is

A **golden set** is a fixed list of questions where the correct answer is already
decided. Given one, "did retrieval improve?" becomes arithmetic.

Yours: `evals/golden/golden.jsonl` — **148 questions over 127 distinct
contracts.**

```json
{"id": "q0042",
 "question": "Under the Adams Golf Endorsement agreement, which state's or country's law governs it?",
 "gold_chunk_ids": ["cuad_ADAMSGOLFINC_...::c018::s25::p0"],
 "gold_answer": "This Agreement shall be governed by the laws of the State of Kansas",
 "doc_id": "cuad_ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT_AGREEMENT",
 "type": "single_hop", "answerable": true, "reviewed": true}
```

## 2. Why it exists

Days 1–2 of this project produced plenty of plausible claims: clause chunks look
better, metadata coverage is high, the contextual prefix probably helps.

**Not one was a measurement.** And two turned out to be wrong when finally tested
(module 06 §6, module 10 §8).

Without a golden set, you cannot distinguish improvement from motion. With one,
every subsequent decision becomes falsifiable.

### The trap that makes this dangerous

A corrupted golden set does not raise an error. It produces **stable, plausible
numbers that mean nothing**, and every decision for the rest of the project rests
on them. So the discipline is:

> **Every generated question must be independently verified, and anything that
> cannot be verified is discarded.**

## 3. The core problem: manufacturing ground truth

Reading 530 contracts to write questions by hand is roughly 40 hours. So you need
a source of pre-existing labels.

**CUAD** provides them: 510 contracts, 13,823 clause annotations across 41
categories, labelled by law students and reviewed by attorneys — and crucially,
each answer recorded as a **character offset**:

```json
{"question": "Highlight the parts related to \"Governing Law\"...",
 "answers": [{"text": "governed by the laws of the State of Kansas",
              "answer_start": 41203}],
 "is_impossible": false}
```

Those offsets are what make automation possible. They pin each answer to an exact
position in an exact string.

## 4. Offset mapping — turning an annotation into a gold chunk

```
document   [.....................................................]
CUAD answer                    ^start=4120 ──────► end=4500
my chunks  [ c0 ][ c1 ][ c2 ][ c3        ][ c4 ][ c5 ]
                                ^3980     ^4600
                                └── c3 is gold
```

```python
overlaps = c["char_start"] < answer_end and c["char_end"] > answer_start
contains = norm(answer)[:120] in norm(c["text"])
if overlaps and contains:
    gold.append(c["chunk_id"])
```

**Both conditions, always.** Module 05 develops why: overlap proves your
*numbers* agree, containment proves the *text* is there. Overlap alone passes
even when normalisation has shifted the text out from under your recorded
ranges.

An answer straddling two chunks makes **both** gold — and `recall@k` then gives
partial credit for finding either, which is the right semantics.

### It worked, because of module 05

**Only 3 of 4,256** generated questions were dropped for a missing gold chunk.
That is a direct dividend of fixing F9 (the discarded preambles). When a brief
warns "this is where it usually breaks" and it doesn't, it's usually because an
earlier bug was already found.

## 5. Question phrasing — the check that reshaped the project

CUAD's own questions are unusable verbatim: *"Highlight the parts (if any) of this
contract related to 'Governing Law' that should be reviewed by a lawyer."* So each
category gets a natural-language template.

Then, **before building anything**, one feasibility check: *can retrieval even
answer these?*

```
Question: "Which state's or country's law governs this agreement?"
```

Asked of 30 different contracts, searching all 530:

| Retrieval | hit@5 | hit@50 |
|---|---|---|
| Unscoped, generic question | **0 / 30** | 2 / 30 |
| Doc-scoped, generic question | 27 / 30 | 30 / 30 |

**Zero.** And not from bad retrieval — that string is **byte-identical** for all
530 contracts, and ~68 of them contain a chunk literally headed "Governing Law".
Expected hit@5 is about `5/68`. It is *arithmetically* unanswerable.

The brief expected recall@5 of 0.55–0.78. Built as written, the harness would
have reported **~0.00 and stayed there through every subsequent improvement** —
an instrument insensitive to the thing it exists to measure.

### The fix: entity-qualified questions

```
"Under the Adams Golf Endorsement agreement, which state's or country's law governs it?"
```

Measured unscoped: **hit@5 4/30, hit@50 13/30.** Low, but *non-zero and
improvable* — which is the entire requirement for a baseline.

**And it is an intervention you must disclose.** Real users would not always name
the contract. So the corpus-wide number measures retrieval *given a named
contract* — easier than a cold query, harder than doc-scoped. That caveat belongs
in the README, and it does.

> **The transferable move:** before you build an instrument, check that it can
> register the phenomenon. Thirty lines of feasibility check saved the week.

## 6. Two leak guards

A question containing its own answer measures nothing.

```python
if norm(answer)[:40] in norm(question):
    rejected["answer_leaked_into_question"] += 1
    continue
```

And structurally: the `Document Name` and `Parties` categories are **excluded
entirely**, because their answer *is* the contract title or the party names —
which the entity-qualified question already states out loud. Templating them
would have produced 1,020 questions that score near-perfectly and prove nothing.

## 7. Sampling: stratify, don't take the first N

4,256 questions is too many (slow) and too repetitive (near-duplicates). Target
~120 *diverse* ones.

```python
by_cat = defaultdict(list)
for g in generated:
    by_cat[g["category"]].append(g)
per = max(1, TARGET // len(by_cat))          # 120 // 20 = 6
for items in by_cat.values():
    sample.extend(random.sample(items, min(per, len(items))))
```

**Why stratify?** Category counts are wildly unequal:

| Category | answerable annotations |
|---|---|
| Governing Law | 437 |
| Cap On Liability | 275 |
| Audit Rights | 214 |
| Warranty Duration | 75 |

Uniform random sampling would give you ~6× more governing-law questions than
warranty questions, so the aggregate metric would mostly measure governing-law
retrieval. Stratification makes the aggregate mean something.

`random.seed(SEED)` makes it reproducible — non-negotiable for an eval set.

## 8. Review — what makes it "golden"

Rather than skim 120 items uniformly, flag the suspicious ones automatically:

| Flag | Count | What it looks for |
|---|---|---|
| `SHORT` | 10 | answer under 30 characters |
| `MANY_GOLD` | 7 | more than 3 gold chunks |
| `NO_CATEGORY_VOCAB` | 4 | gold chunk lacks vocabulary the category implies |
| `STARTS_LOWER` | 3 | answer begins mid-sentence |
| **clean** | **98/120** | |

All 22 flagged were read in full, plus 12 random unflagged ones to check that
bucket too.

### Most flags were false positives — which is itself the finding

- `SHORT` fired on ten **dates**. "February 20, 2020" *is* a complete answer to
  "on what date was it signed?"
- `NO_CATEGORY_VOCAB` fired on *"no less than 50% of the Joint Venture Products
  should be sold overseas"* for Minimum Commitment. Textbook minimum commitment —
  my keyword list just didn't include "no less than".
- `MANY_GOLD` is not a defect: a long annotated span legitimately crosses several
  chunks, and `recall@k` already gives partial credit.

**Heuristics find candidates. They do not make decisions.** A flag is a request
for attention, not a verdict.

### Two genuine deletions

| Question | Why deleted |
|---|---|
| GARMAN Software Licence — "on what date was it signed?" | answer is `"____ day of May, 2000"` — an **unfilled blank**. Nothing to score against |
| MediWound Supply — "how long does any warranty last?" | annotated answer is a non-conforming-product **rejection window**, which doesn't answer warranty duration |

### Two edits rather than deletions

Answers beginning mid-sentence (`"after Nantz Communications has received
notice…"`) or mid-*word* (`"n case of an order volume"` — the offset lands one
character inside "In"). Fine for scoring **retrieval** (the gold chunks are
correct) but unusable as answer references, so they carry
`answer_is_fragment: true` for module 15 to exclude.

### Report *both* rejection rates

```
generation-stage rejection: 6.3%   (282 fragments + 3 no-gold-chunk of 4,541)
review-stage rejection:     1.7%   (2 of 120)
```

The review number is far below the brief's ~12% expectation, and the honest
reason is **not** that the review was lax — it's that filtering already happened
upstream. Quoting only one of these would mislead. Quoting both is the credible
move.

## 9. Keying decisions on stable identifiers — F20

Review decisions were first keyed on question ids (`g0755`). Then `stratify()`
was changed to renumber the sample `g0001…g0120`. Result:

```
reviewed 120 -> kept 120, deleted 2
```

**Self-contradictory.** 120 kept *and* 2 deleted cannot both be true. Both
deletions matched nothing and were silently skipped.

The root cause: **the ids were never stable.** They're positional, so any change
to generation or sampling renumbers everything.

Fix — key on content, and fail loudly when a decision matches nothing:

```python
key = f"{q['doc_id']}|{q['category']}"           # stable by construction
...
if (hit["delete"], hit["flag"]) != want:
    raise SystemExit(f"review keys stale: applied {hit['delete']} deletes / "
                     f"{hit['flag']} flags, expected {want[0]}/{want[1]}")
```

**How it was caught:** the log line contradicted itself. *A print statement that
can disagree with itself is a free assertion* — same detection mechanism as F9's
character totals and F24's identical sweep metrics.

## 10. The hard cases — 30 questions that matter more than the other 118

### Unanswerable (18) — and they are *verified*

The correct response is refusal. This is the hallucination stress test.

"Plausible" **asserted** is worthless — if a contract *does* mention escrow, the
question isn't unanswerable and the test is backwards. So each topic ships with
the terms that must be **absent**:

```python
("a source-code escrow arrangement", ["escrow"]),
("pandemic or epidemic force-majeure relief", ["pandemic", "epidemic", "covid"]),
("ITAR or export-control compliance", ["itar", "export control", "export administration"]),
```

and the builder searches for a contract where none appear:

```python
low = docs[doc_id]["text"].lower()
if any(t in low for t in absent):
    continue                       # this contract does mention it - try another
```

**Unanswerability becomes a verified property, not an assumption.** Independently
re-checked: **0 leaks.**

Note the topics are deliberately *adjacent* to commercial contracts — GDPR,
SOC 2, biometric data, FCPA, liquidated damages. "What is the recipe for pizza?"
tests nothing, because refusing it requires no judgement.

### Near-miss traps (6)

Ask about clause A in a contract that **also contains** clause B, where B is the
obvious confusion:

| Asked | Trap |
|---|---|
| Cap On Liability | Uncapped Liability |
| Termination For Convenience | Post-Termination Services |
| IP Ownership Assignment | License Grant |
| Non-Compete | Exclusivity |

Both must genuinely exist, so the builder only accepts documents with CUAD
annotations for both. The trap's chunk ids are stored as `trap_chunk_ids`, and
the runner reports whether the trap outranked the answer.

### Multi-hop (4) and aggregation (2)

Multi-hop needs two clauses; gold is the union. Aggregation — *"How many contracts
are governed by New York law?"* — has **no gold chunks at all** and is included
**on purpose** as a measured known weakness that motivates the agent (module 17).

> Measuring something you expect to fail, and saying so up front, is stronger
> than quietly excluding it.

## 11. Final composition

| Type | n | % |
|---|---|---|
| single_hop | 118 | 80% |
| unanswerable | 18 | **12%** |
| near_miss | 6 | 4% |
| multi_hop | 4 | 3% |
| aggregation | 2 | 1% |
| **Total** | **148** | across **127** documents |

Independent validation of the finished file — all passing:

```
1 all gold chunk ids resolve          PASS  (0 dangling)
2 answer text inside a gold chunk     PASS  (0 failures)
3 unanswerable really unanswerable    PASS  (0 leaks)
4 ids and questions unique            PASS
5 no answer leaked into question      PASS
6 all reviewed                        PASS  148 questions
```

## 12. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **Questions can't identify the answer** | metrics pinned near 0 | feasibility-check before building | entity-qualify, or scope |
| Answer leaks into the question | suspiciously high scores | substring check | reject those questions |
| Trivially-templated categories | near-perfect scores on some categories | per-category breakdown | exclude Document Name / Parties |
| Unstratified sampling | aggregate reflects one category | count per category | stratify |
| Unstable decision keys | reviews silently stop applying | assert every decision matched | content-derived keys |
| Unverified "unanswerable" | scoring refusals as correct when wrong | check absent terms against full text | verify absence |
| Offset/normalisation drift | gold chunks don't contain answers | overlap **and** containment | module 02 |
| Single rejection rate quoted | misleadingly clean or dirty | report generation *and* review | both |
| Unversioned eval set | numbers move for the wrong reason | hash the file; record it per run | commit it, cite the SHA |

## 13. Provenance — the part that protects you

**CUAD's clause spans are attorney-reviewed.** Everything else here is not:

- the sampling review
- the 30 hand-authored questions
- every keep/edit/delete decision

`evals/golden/review_decisions.json` records the method in a `_provenance` field.
So the honest claim is **"AI-reviewed, CUAD-derived"**, never
"attorney-verified".

> A stated limitation is a strength. A hidden one found by an interviewer is
> fatal.

## 14. Interview angle

**"How did you build your eval set?"** — 118 questions templated from CUAD's
attorney-annotated clause spans by mapping character offsets onto chunk ranges,
keeping a chunk as gold only if it both overlaps the span *and* contains the answer
text. 30 hand-authored hard cases: 18 unanswerable with absence verified against
the full contract text, 6 near-miss traps where both clauses provably exist, 4
multi-hop, 2 aggregation. 148 total, 12% unanswerable, all reviewed — and it's
AI-reviewed, not attorney-verified, which is recorded in the file.

**"What was the most important thing you did before building it?"** — Checked
whether the questions were answerable at all. The generic template phrasing is
byte-identical across all 530 contracts, so corpus-wide retrieval scored hit@5 =
0/30 — arithmetically unanswerable, since ~68 documents have an equally good
governing-law clause. Built as specified, the harness would have read zero and
stayed there through every improvement I made afterwards.

**"How do you know your unanswerable questions are really unanswerable?"** — Each
one carries the terms that must be absent, and is only assigned to a contract
where none appear anywhere in its full text. Independently re-checked: zero
leaks. If I'd just written plausible-sounding topics, some would have been
answerable and I'd have been scoring refusals as correct when they were wrong.

**"Your review rejection rate is only 1.7% — was the review shallow?"** — No, and
the honest answer needs both numbers. 6.3% was rejected at *generation* by
automated quality filters — 282 fragment answers and 3 with no gold chunk — before
sampling. Of the 120 that survived into review, 2 were deleted and 2 edited. The
low review number reflects upstream filtering, not leniency, and quoting only one
of the two would mislead.

## 15. Test yourself

**Q1.** Why must a gold chunk satisfy *both* overlap and containment? Give a case
where overlap passes and containment fails.

<details><summary>Answer</summary>

Overlap compares two sets of numbers you control — the chunk's recorded
`char_start`/`char_end` and the annotation's span. It never inspects text.

Failing case: whitespace is normalised *after* `char_start` values are assigned.
The recorded ranges still tile the document contiguously, so every annotation
intersects some chunk's range and overlap passes for ~100% of answers. But the
chunk's actual `text` has shifted relative to those ranges, so it holds different
content than the range claims. Containment fails; overlap does not.

</details>

**Q2.** Explain why a generic question like "which state's law governs this
agreement?" scores hit@5 = 0/30 corpus-wide, and why this is not a retrieval
failure.

<details><summary>Answer</summary>

The string is identical for all 530 contracts, and ~68 of them contain a chunk
literally headed "Governing Law" — all equally good matches for that query.
Nothing in the query distinguishes contract #217's clause from contract #4's, so
expected hit@5 is roughly `5/68 ≈ 7%`, and 0/30 is consistent with that.

It isn't a retrieval failure because **no retriever could do better**. The
information required to pick the right document is absent from the input. It's an
*eval design* failure — the question doesn't specify the answer.

</details>

**Q3.** Why exclude CUAD's `Document Name` and `Parties` categories when your
questions are entity-qualified?

<details><summary>Answer</summary>

Because their answer *is* the contract title or the party names, and the
entity-qualified question already states those. "Under the **Adams Golf
Endorsement** agreement, what is the document name?" contains its own answer.

Those questions would score near-perfectly while measuring nothing about
retrieval — they'd measure whether the retriever can find a preamble containing a
string the query supplied. Since `Document Name` and `Parties` have 510 and 509
answerable annotations respectively, templating them would have added ~1,000
inflating questions to a 4,256-question pool.

</details>

**Q4.** Your review decisions were keyed on question id and silently stopped
applying. Diagnose the root cause and give the two-part fix.

<details><summary>Answer</summary>

**Root cause:** the ids were positional (`g0755` = the 755th generated question),
so they change whenever generation or sampling changes. When `stratify()` began
renumbering the sample `g0001…g0120`, the recorded keys referenced ids that no
longer existed.

**Fix, part 1 — stable keys.** Key on content-derived identifiers:
`f"{doc_id}|{category}"`, which is stable across regeneration by construction.

**Fix, part 2 — fail loudly.** Assert that every recorded decision matched
something:
```python
if (hit["delete"], hit["flag"]) != want: raise SystemExit(...)
```
Part 1 alone is insufficient — if the keys drift again for some other reason, you
want a crash, not a silent no-op. The symptom that exposed it was a log line
reading "kept 120, deleted 2", which cannot both be true.

</details>

**Q5.** Harder. You have no annotated dataset for your domain — only 2,000 of your
company's own contracts. Design a golden set, and say what you lose.

<details><summary>Answer</summary>

**Design:**

1. **Stratify by document type and clause category first**, so the set is
   diverse by construction rather than by luck. Enumerate the categories you
   actually care about (10–20), not everything.
2. **Hand-label a small, honest n.** 8–10 questions per category across 15–20
   documents gives 120–200 questions — the range this project used. Record the
   labeller and date in a `_provenance` field.
3. **Record character offsets while labelling**, not just the answer text. This
   is the step people skip and it's what buys span coverage (module 05), the
   overlap-and-containment check, and the ability to re-chunk without relabelling.
4. **Feasibility-check the phrasing** before mass-producing: does each question
   contain enough information to identify its answer document?
5. **Author the hard cases deliberately** — unanswerable questions with verified
   absent terms, and near-miss traps where both clauses provably exist. These are
   cheap to write and disproportionately informative.
6. **Double-label a subset** (say 20 questions by two people) and report
   inter-annotator agreement. Without it you have no idea whether your labels are
   reproducible — the same argument as validating an LLM judge (module 13).

**What you lose:**

- **Scale.** 150 hand-labelled questions instead of 4,256 generated candidates,
  so per-category n is small and per-type numbers are directional only.
- **Independent provenance.** Your labels are yours, so "attorney-reviewed" is
  unavailable and any bias in your reading is baked into the ground truth — you
  cannot detect a systematic misunderstanding of your own domain.
- **Free ground truth for derived tasks.** This project got agent evaluation
  labels *for free* from CUAD's `Cap On Liability` annotations (module 17); with
  only your own corpus, every new task needs new labelling.

**The pragmatic hybrid:** include a public annotated dataset alongside your own
corpus specifically to buy the correctness checks — span coverage in particular is
unmeasurable without offsets. That is a genuine and under-appreciated reason to
keep CUAD in a corpus even when your real documents are your own.

</details>

---

**Previous:** [11 — Retrieval metrics](11-retrieval-metrics.md) ·
**Next:** [13 — LLM-as-judge](13-llm-as-judge.md)
