# 05 — Span Coverage: Proving Your Chunker Didn't Lose the Answers

The most important module in Part B, and the one with no equivalent in most RAG
tutorials. It is also the story of this project's single most consequential bug.

---

## 1. What it is

**Span coverage** is the fraction of known answer spans that are fully contained
inside at least one chunk.

```
coverage = (# annotated answers reachable in some chunk) / (# annotated answers)
```

It is a property of your **chunker**, measured *before* you run any retrieval. It
is a hard upper bound on recall — and therefore on the whole system.

## 2. Why it exists

Every retrieval metric assumes something nobody checks: **that the answer is
somewhere in the index.**

If a chunk containing the answer does not exist, then:

- `recall@5` cannot find it
- `recall@50` cannot find it
- no re-ranker can promote it
- no larger LLM can reason about it
- **and no error is raised anywhere**

You would spend days tuning retrieval against a ceiling that isn't in the
retriever. Span coverage is the check that makes that ceiling visible.

## 3. Intuition

```
document:  [=========================================================]
answer:                    [## 380 chars ##]
                           ^4120          ^4500

chunks:    [ c0 ][ c1 ][ c2 ][ c3      ][ c4 ][ c5 ]
                             ^3980      ^4600
                             └── c3 covers [4120,4500] entirely → REACHABLE
```

Now the failure case:

```
chunks:    [ c0 ][ c1 ]                  [ c4 ][ c5 ]
                        ^^^^^^^^^^^^^^^^
                        no chunk here → answer UNREACHABLE
```

The second picture is what a buggy chunker produces, and nothing about it looks
wrong from the outside. You still have chunks. Retrieval still returns five of
them. They simply never contain that answer.

## 4. The two-part test — and why one part is not enough

```python
overlaps = chunk["char_start"] < answer_end and chunk["char_end"] > answer_start
contains = norm(answer)[:120] in norm(chunk["text"])
reachable = overlaps and contains
```

### Why `overlaps` alone is insufficient

Overlap is **arithmetic on your own bookkeeping**. It says: the range I recorded
for this chunk intersects the range the annotation claims. Both numbers are
yours; if your `char_start` values are computed wrongly, or the text was
normalised after offsets were assigned, overlap still passes.

> Overlap proves your *numbers* agree. Containment proves the *text* is there.

### Why `contains` alone is insufficient

A short answer string like `"New York"` appears in many chunks by coincidence.
Containment alone would credit the wrong chunk. Requiring overlap pins it to the
right *position* in the document.

### Why the interval condition is `<` and `>` and not `<=`

`chunk_start < answer_end and chunk_end > answer_start` is the standard
half-open interval intersection test. Work through the boundary case: a chunk
ending exactly at 4120 and an answer starting exactly at 4120 share **no
characters** — index 4120 belongs to the answer, not the chunk. With `<=` you'd
count that as overlap and credit a chunk containing none of the answer.

### Why `[:120]`

Long annotated spans may cross a chunk boundary, so the *whole* answer might not
sit in one chunk even when the answer's beginning clearly does. Matching the
first 120 normalised characters is a pragmatic compromise: strict enough to prove
you found the right passage, tolerant enough not to reject a 900-character span
whose head is unambiguously present. It's a **deliberate approximation you should
be able to defend**, not an accident.

## 5. The bug this check found — F9

This is the story to remember.

The Day 2 structural chunker began each document at its **first regex match** —
the first numbered section. Everything before that was never emitted:

```
CONSULTING AGREEMENT                              <-- discarded
This Agreement is made this 2nd day of January     <-- discarded
2020, by and between Global Technologies, Ltd      <-- discarded
("Company") and ...                                <-- discarded
WHEREAS the Company desires ...                    <-- discarded
                                                   
1.  DEFINITIONS                                   <-- first chunk starts HERE
```

Everything discarded is the **preamble**: the parties, the execution date, the
recitals, the contract title. Across the corpus that was **1,832,665 characters.**

Because CUAD supplies character-level offsets, the damage could be priced
exactly rather than guessed:

| | |
|---|---|
| Annotated answers reachable | **83.4%** |
| Answers in no chunk at all | **16.6%** |
| Retrieval recall ceiling, before any tuning | **0.834** |

And it was **biased**, not random. The categories living in preambles —
`Parties`, `Agreement Date`, `Effective Date`, `Document Name` — were
disproportionately destroyed, while `Indemnification` and `Governing Law` were
untouched. An eval built on that corpus would have systematically understated
performance on exactly one family of questions.

After the fix, coverage is **100.0%** and text loss is **0.00%**.

### What it would have cost

Had this survived one more day, Day 3's golden set would have been generated
against the damaged corpus. The generator discards questions whose gold chunk
can't be found — so roughly a sixth of candidate questions would have vanished
*quietly*, the date/party questions most of all. Day 4's ablation would then have
optimised retrieval against a ceiling of 0.834 while reporting numbers that
looked internally consistent.

**Nothing would have failed. The numbers would simply have been wrong.**

## 6. Two accounting checks worth having permanently

### Check 1 — text conservation

```python
stored = sum(len(c["text"]) for c in chunks)
source = sum(len(d["text"]) for d in docs)
loss   = (source - stored) / source     # negative means overlap inflation
```

For a non-overlapping chunker this must be ≈ 0. For an overlapping one it must
equal `-OVERLAP/stride` (module 03). **Any other value is a bug.** This is how
"0.00% text loss" was established.

### Check 2 — span coverage

```python
reachable = sum(1 for a in answers if any(
    c["char_start"] < a.end and c["char_end"] > a.start
    and norm(a.text)[:120] in norm(c["text"])
    for c in by_doc[a.doc_id]))
print(f"coverage {reachable}/{len(answers)} = {reachable/len(answers):.1%}")
```

Result before writing any golden-set code: **6,697 / 6,702 = 99%.** Only **3 of
4,256** generated questions were later discarded for a missing gold chunk — which
is how you know the chunker is sound.

## 7. The general pattern — this is the transferable skill

Span coverage is one instance of a family:

> **Accounting checks.** Ask whether the totals reconcile, rather than whether the
> code looks right.

Across this project, that pattern found:

| Check | Bug it caught |
|---|---|
| Do stored chars equal source chars? | preamble discarded (F9) |
| Are annotated answers reachable? | 16.6% unreachable (F9) |
| Does `max(len(chunk))` respect `MAX_CHARS`? | limit never enforced (F15) |
| Do 148 questions minus 2 deletions equal 148? | review decisions silently no-op'd (F20) |
| Is `recall@50` unchanged by re-ranking? | would have caught candidate dropping |
| Do tokens evaluated match tokens sent? | 60% of prompt truncated (F8) |
| Does the chunk count match `ceil(chars/stride)`? | disabled overlap |

Every one of those bugs was **silent**. None raised an exception. Each was found
by a reconciliation, not by reading code or watching it run.

**The instinct to build:** when you write a number, ask what *other* number it
must agree with — and assert it.

## 8. Failure modes of the check itself

| Failure | Symptom | Fix |
|---|---|---|
| Only checking `overlaps` | passes while text is missing | require containment too |
| Only checking `contains` | credits the wrong chunk | require overlap too |
| Normalising differently in the two places | spurious failures | one `norm()` used everywhere |
| Comparing full answer text | long spans fail spuriously | prefix match, documented |
| Running it after generating the eval | you've already baked in the damage | run it **before** |
| No annotated corpus to check against | can't measure coverage | hand-label a small sample; or check text conservation only |

That last row matters. Span coverage needs offsets. **Without an annotated
corpus you cannot measure it** — which is a strong practical argument for
including an annotated dataset like CUAD even when your target domain is your own
documents. It buys you a correctness check you otherwise cannot buy.

## 9. Compared with

| | Span coverage | recall@k |
|---|---|---|
| Measures | the **chunker** | the **retriever** |
| Needs a retriever? | no | yes |
| Needs an embedding model? | no | yes |
| Run when? | right after chunking | after indexing |
| Failure means | the answer is not in the index | the answer is in the index but not found |
| Bounds | recall@k, for every k | end-to-end answer quality |

They are strictly ordered: **span coverage bounds recall@∞, which bounds
recall@50, which bounds recall@5, which bounds answer quality.** Debug in that
order and you never optimise a stage that isn't the constraint.

## 10. Interview angle

**"How do you know your chunking is correct?"** — This is a question most
candidates cannot answer at all. The strong answer: I measure span coverage —
the fraction of known answer spans fully contained in some chunk — using the
character offsets from an annotated dataset. My structural chunker initially
scored 83.4% because it discarded every document preamble, 1.8 million
characters, which capped retrieval recall before any tuning. It's now 100% with
0.00% text loss.

**"What's the most valuable bug you found?"** — F9, and the reason it's valuable
is that it was *invisible*: no exception, no warning, chunks existed, retrieval
worked. It was caught by an accounting check — do the stored characters equal the
source characters — not by reading the code. And it was biased: it destroyed
date and party answers specifically, so the eval would have understated one
family of questions systematically.

**"Your recall@5 is low. Where do you start?"** — At the earliest stage that
bounds the outcome. First span coverage: is the answer in the index at all? Then
recall@50: is it retrieved at any depth? Then the recall@50-to-recall@5 gap: is
it retrieved but mis-ranked? Only then generation. Each stage bounds the next, so
fixing a downstream stage cannot recover information already discarded.

## 11. Test yourself

**Q1.** Why is checking `overlaps` alone insufficient? Describe a scenario where
overlap passes but the answer is genuinely absent from the chunk.

<details><summary>Answer</summary>

Overlap compares two sets of numbers that are both under your control — the
chunk's recorded `char_start`/`char_end` and the annotation's span. It never looks
at text.

Scenario: you normalise whitespace *after* assigning `char_start` values. The
recorded ranges still tile the document contiguously, so any annotation
intersects some chunk's range and overlap passes for ~100% of answers. But the
chunk's actual `text` has shifted relative to those ranges, so it contains
different content than the range claims. Containment fails; overlap does not.
This is precisely the offset-drift failure from module 02.

</details>

**Q2.** A chunker discards each document's first 200 characters. Span coverage
drops to 96%. Why is that 4% loss *worse* than a random 4%?

<details><summary>Answer</summary>

Because it is **systematically biased**, not randomly distributed. The first 200
characters of a contract contain the title, the execution date and the parties —
so the loss falls entirely on `Document Name`, `Agreement Date`, `Effective Date`
and `Parties` questions, while `Indemnification` and `Governing Law` are
untouched.

A random 4% loss adds noise and slightly depresses every metric. A biased 4% loss
makes one *category* of question look unanswerable, so your per-category
breakdown will show "the system is bad at dates" — a conclusion about retrieval
that is actually a fact about your chunker. Biased loss produces wrong
*explanations*, which is far more expensive than wrong numbers.

</details>

**Q3.** Your text-conservation check reports `-17.6%` (stored exceeds source).
Bug or expected?

<details><summary>Answer</summary>

Expected, if the chunker overlaps. Inflation equals `OVERLAP/stride`; at
`CHUNK_SIZE=1000, OVERLAP=150` the stride is 850 and `150/850 = 17.6%` — exactly
the reported figure.

The check is still valuable: it must equal *that specific number*. Anything else
— including 0% for an overlapping chunker, or −33% when you expect −17.6% — is a
bug in the stride logic. The point of the assertion is not "loss must be zero"
but "loss must equal what the parameters predict".

</details>

**Q4.** You have no annotated corpus — no character offsets anywhere. What can
you still check about your chunker, and what becomes unmeasurable?

<details><summary>Answer</summary>

**Still checkable:**
- Text conservation — do stored characters equal source characters (± the
  predicted overlap inflation)?
- Chunk-count identity — `ceil(chars/stride)` for fixed-width.
- `MAX_CHARS`/`MIN_CHARS` invariants.
- Clause-boundary start rate, via regex.
- Duplicate and near-duplicate chunk rates.

**Unmeasurable:** span coverage itself, because you have no ground-truth answer
positions. You cannot know whether the *answers* survived, only whether the
*text* did — and those differ, since text can be conserved while every answer
straddles a boundary.

The practical remedy is to hand-label a small sample (20–30 answers with their
character positions) purely as a chunker test, or to include an annotated public
dataset in the corpus specifically to buy this check. This is a genuine and
under-appreciated argument for using CUAD even when your real documents are your
own.

</details>

**Q5.** Harder. Span coverage is 100%. Recall@50 is 0.58. Are these consistent?
What does the combination tell you, and what is the next thing you measure?

<details><summary>Answer</summary>

Fully consistent — and the combination is highly informative.

100% coverage says every answer **exists in some chunk**, so the index is
complete. Recall@50 of 0.58 says the retriever **fails to fetch** 42% of those
chunks even at depth 50. The problem is therefore located precisely: not
chunking, not indexing, but the **matching function** — the embedding's ability
to connect this query to that chunk.

That rules out an entire class of fixes (re-chunking, re-indexing, larger
overlap) and points at another: lexical retrieval to catch exact terms the
embedding blurs, query rewriting, or a different embedding model. It also rules
out re-ranking as a fix for those 42%, since a re-ranker only reorders what was
fetched.

Next measurement: split recall@50 by question category and by retriever. In this
project that revealed the queries named party entities that dense embeddings
blur — which motivated BM25 (module 08), and the ceiling then moved only from
0.585 to 0.604, showing even that wasn't the whole story.

</details>

---

**Previous:** [04 — Structural chunking](04-structural-chunking.md) ·
**Next:** [06 — Dense retrieval](06-dense-retrieval.md)
