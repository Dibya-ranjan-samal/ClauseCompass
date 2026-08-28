# ClauseCompass Day 2 — Explained From Scratch

Companion to [day1-explained.md](day1-explained.md). Same rule: every number
here came from the actual run.

**What Day 2 is for.** Day 1 proved the pipeline connects. Day 2 makes the data
good enough that tomorrow's eval numbers mean something. If the corpus is 20
contracts and the chunks are arbitrary, a good eval score tells you nothing.

**The single most important thing that happened today** is buried in Part 3 as
F9: the chunker was silently discarding 16.6% of the annotated answers, which
would have capped tomorrow's retrieval recall at 83.4% with no error message and
no obvious symptom. Finding that *before* building the eval harness is worth
more than everything else on this page.

---

## Contents

1. [Part 0 — What changed, in one table](#part-0)
2. [Part 1 — CUAD: why this dataset is the whole project](#part-1)
3. [Part 2 — The tasks, explained](#part-2)
4. [Part 3 — Failures, causes, fixes](#part-3)
5. [Part 4 — New concepts](#part-4)
6. [Part 5 — Interview questions](#part-5)
7. [Part 6 — Honest status and CV numbers](#part-6)

---

<a name="part-0"></a>
## Part 0 — What changed, in one table

| | Day 1 | Day 2 |
|---|---|---|
| Documents | 20 (EDGAR) | **530** (20 EDGAR + 510 CUAD) |
| Corpus size | 2,728,031 chars | **29,535,164 chars** (10.8×) |
| Chunking | fixed 1,000-char windows | **clause boundaries**, 70% of docs |
| Chunks | 3,219 | **18,280** |
| Chunk metadata | `chunk_id`, `doc_id`, `text` | + section number/heading, char offsets, strategy, contract type, governing law, parties, dates |
| Ground truth | none | **13,823 attorney-reviewed answer spans** + 80 hand labels |
| Retrieval filters | none | payload indexes on 6 fields |

Note the chunk count grew only 5.7× while the corpus grew 10.8×. That is the
structural chunker doing its job: a real clause is longer than an arbitrary
window, so each chunk carries more source text (1,615 chars of source per chunk,
against Day 1's 850-char stride).

One statistic worth reading carefully. The *overall* median chunk is 1,000 chars,
which lands inside the brief's 400–1,200 target — but that number is an artifact
of mixing two strategies:

| Strategy | n | median | mean |
|---|---|---|---|
| structural | 13,237 | **1,554** | 1,900 |
| fallback | 5,043 | 1,000 | 986 |

The fallback chunks are *exactly* 1,000 because that is the window size, and they
drag the combined median onto the target. Structural chunks are genuinely larger
than the brief anticipated, because a real contract clause is. Reporting the
blended median would have hidden that — the same mistake as averaging EDGAR and
CUAD retrieval scores together.

---

<a name="part-1"></a>
## Part 1 — CUAD: why this dataset is the whole project

**CUAD** = Contract Understanding Atticus Dataset. 510 real commercial
contracts, annotated by law students and reviewed by attorneys, released
CC BY 4.0 by The Atticus Project.

*Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal Contract
Review", NeurIPS 2021.*

What we actually got:

| | Value |
|---|---|
| Contracts | 510 |
| Questions | 20,910 (41 categories × 510) |
| **Answer spans with character offsets** | **13,823** |
| Answerable questions | 6,702 |
| Unanswerable questions | 14,208 (68%) |
| Corpus size | 26,807,133 chars (mean 52,563/contract) |

### Why the character offsets are the point

Each annotation looks like this:

```json
{"question": "Highlight the parts ... related to \"Governing Law\" ...",
 "answers": [{"text": "DISTRIBUTOR AGREEMENT", "answer_start": 44}]}
```

`answer_start` is an exact character position in the contract. Tomorrow, Day 3
maps those offsets onto each chunk's `char_start`/`char_end` range and knows,
without a human in the loop, **which chunk is the correct answer to which
question.** That is a golden eval set for free.

Hand-writing an equivalent set would be roughly 40 hours of reading contracts.
This is why the brief calls CUAD the highest-leverage thing in the project.

### And the 68% unanswerable questions are not waste

CUAD asks all 41 category questions of every contract, so most have no answer.
That looks like noise; it is actually the other half of the eval. A retrieval
system that answers everything confidently is broken — **14,208 questions with
a known-correct answer of "not in this contract"** is exactly the test set for
Day 5's refusal accuracy. Keep them, flagged.

### The dataset-access trap

The brief says to use `load_dataset('theatticusproject/cuad-qa')`. That repo
contains three files: `.gitattributes`, `README.md`, and `cuad-qa.py` — a
**loading script**. Dataset loading scripts are disabled in `datasets` v3+ for
security reasons, so that call cannot work.

The durable path is the raw JSON in the sibling repo:

```bash
curl -sL -o data/external/CUAD_v1.json \
  https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json
```

38 MB, SQuAD format, no dependency on `datasets` at all. **Lesson worth
generalising:** when a tutorial names a dataset ID, check what files the repo
actually contains before adding the library it recommends.

---

<a name="part-2"></a>
## Part 2 — The tasks, explained

### Task 2 — Merging two corpora (`build_corpus.py`)

Trivial code, one important idea: every document carries a `source` field
(`edgar` or `cuad`).

**Why that matters.** The two sources are not the same kind of data:

| | EDGAR | CUAD |
|---|---|---|
| Documents | 20 | 510 |
| Mean length | **136,401 chars** | 52,563 chars |
| Origin | raw filing HTML, parsed by us | clean text, prepared by the dataset |

EDGAR documents are 2.6× longer and much messier. Day 4 reports retrieval
metrics **per source**, and if EDGAR scores worse that is a finding about HTML
parsing, not noise to average away. Aggregate metrics hide exactly this.

---

### Task 3 — Structural chunking (`chunk_structural.py`)

The biggest engineering task of the day, and the one that generated the day's
most serious bug.

#### Why structure beats character counting

Day 1 cut every 850 characters. A chunk could begin mid-word and end mid-clause
(Day 1's F7). Contracts are *already* structured — they are numbered — so use it:

```
7.2 Indemnification. Supplier shall indemnify...
```

A chunk that is exactly one clause is a better retrieval unit: it is
self-contained, it has a name (`section_heading`), and a citation to it points
at something a lawyer would recognise.

#### The regex, and why one pattern wasn't enough

The brief's pattern requires the heading to sit **alone on its line**:

```python
r"^\s*(?:Section\s+)?(\d+(?:\.\d+)*)\s*[.\)]?\s+([A-Z][A-Za-z ,\-/]{2,80})?\s*[.:]?\s*$"
#                                                                            ^^^ line must END here
```

That matched only **57%** of documents. Printing the failures showed why — many
contracts run the clause body *inline* after the heading:

```
1. Services and Compensation. Consultant agrees to act as a consultant...
2. Confidentiality. "Confidential Information" means any proprietary...
```

The `$` anchor rejects all of those. Adding a second pattern for the inline form
and taking the **union** of both lifted coverage to **70%**:

| Pattern | Docs with ≥3 matches | Median gap |
|---|---|---|
| Heading alone on line (brief's) | 305 / 530 = 57% | 974 |
| Inline heading, body follows | 197 / 530 = 37% | 842 |
| Any numbered clause start | 317 / 530 = 59% | 592 |
| ALL-CAPS heading line | 236 / 530 = 44% | **193** ← over-matching |
| **Union of the first two** | **373 / 530 = 70%** | **768** |

The ALL-CAPS pattern is instructive: 44% coverage looks fine until you see the
median gap of 193 characters. It was matching party names, `IN WITNESS WHEREOF`,
and exhibit titles — producing many tiny fragments rather than clauses. **Chunk
size distribution is how you detect an over-matching regex.** Coverage alone
would have hidden it.

#### The dedupe subtlety

Two patterns matching the same boundary must count once. Deduping on the raw
match offset does *not* work — the patterns consume leading whitespace
differently, so the same clause is recorded twice at different offsets. Dedupe
on **the start of the line holding the section number**:

```python
key = text.rfind("\n", 0, m.start(1)) + 1
```

Getting this wrong is what made my first attempt score 58% instead of 70%: I had
also "tidied" the first pattern's `\s` to `[ \t]`, which broke contracts that put
the number and heading on *separate* lines. **A whitespace class is a semantic
choice, not a style choice.**

#### Sizes

| | Value |
|---|---|
| Chunks | 18,280 from 530 docs |
| Structural / fallback | 373 / 157 docs (70% / 30%) |
| — by source | EDGAR 85% structural, CUAD 69% |
| Median chunk (all) | 1,000 chars |
| — structural chunks | median **1,554**, mean 1,900 |
| — fallback chunks | median 1,000, mean 986 |
| p90 / max | 3,809 / 4,192 |
| Preamble chunks | 787 |
| Duplicate chunk_ids | 0 |
| **Corpus chars not covered by any chunk** | **0.00%** |

---

### Task 4 — Metadata extraction (`extract_metadata.py`)

Runs **once, offline**. Never in the query path — a user's question must not wait
for a transformer to classify a contract.

#### Why regex before machine learning

For `governing_law` and `effective_date`, regex beats a model: the phrasing is
formulaic, the output must be exact, and a regex is auditable when it is wrong.
Reach for a model only where the target is fuzzy — "what *kind* of contract is
this" has no regex.

The honest counterpoint: regex is brittle in ways you only find by measuring.
This one needed five separate fixes (Part 3, F10–F11).

#### Zero-shot classification, and the taxonomy trap

**Zero-shot classification** means classifying into labels the model was never
trained on. The NLI approach: pose each label as a hypothesis ("This text is
about a *sponsorship agreement*") and ask a natural-language-inference model how
strongly the document entails it. Highest-scoring label wins. No training data
needed — you just supply label names.

Which makes the label names load-bearing. The brief supplies nine:

> NDA · MSA · DPA · SOW · Employment · Software License · Lease · Supply · Distribution

Those describe an EDGAR MSA corpus. Against CUAD they match **35 of 510
documents — 6%.** CUAD's actual composition, taken from its filenames:

| Type | n | | Type | n |
|---|---|---|---|---|
| Distribution | 34 | | Hosting | 18 |
| Maintenance | 34 | | Outsourcing | 18 |
| Strategic Alliance | 32 | | Franchise | 15 |
| Sponsorship | 31 | | Agency | 15 |
| Development | 31 | | Cooperation | 14 |
| License | 26 | | Manufacturing | 14 |
| Endorsement | 23 | | Joint Filing | 13 |
| Services | 22 | | Reseller | 13 |
| IP | 20 | | Transportation | 13 |
| Branding | 20 | | …and 7 more | |
| Supply | 18 | | | |

144 raw title strings collapse into **26 canonical labels covering 95%** of the
corpus. Both label sets are scored, so the improvement is a measured number
rather than an assertion.

> **Interview line:** "The tutorial's nine contract-type labels covered 6% of my
> corpus. I derived a 26-label taxonomy from the dataset's own metadata and
> measured both, so the fix is a before/after number instead of a claim."

#### NER for parties

**NER** — Named Entity Recognition — tags spans as PERSON / ORG / LOC. Keep the
`ORG` entities from the preamble and you have the contracting parties.

`aggregation_strategy="simple"` matters: transformer NER labels *word pieces*,
so "Advanced Lighting Technologies" arrives as several `B-ORG`/`I-ORG` tokens.
The aggregation strategy stitches them into one entity. Without it you get
fragments.

#### What actually ran, and why it isn't NER

**The NLI and NER weights could not be downloaded.** Hugging Face reset the
large-file transfers repeatedly — `snapshot_download` stalled at 0-byte
`.incomplete` stubs, and a `curl -C -` resume loop went 93 MB → 42 MB → 56 MB,
restarting rather than resuming, across ~40 minutes.

So classification and party extraction run through the **local Ollama model**
(`qwen2.5:3b`) instead, selected by `--backend` and recorded per document in
`metadata.json` as `extraction_backend`.

Be precise about what this is:

- It **is** still zero-shot classification — the model has seen none of these
  contracts and gets only a label list.
- It is **not** NLI entailment, and it is **not** NER. It is instruction-following
  generation.
- It yields **no calibrated confidence score**, so the `type_confidence >= 0.45`
  threshold has no equivalent; the model is asked to answer `Unknown` instead.

Do not write "NER" on a CV for this. Write what it is: LLM-based extraction, with
the NLI/NER path implemented and switchable for when the weights are available.

#### What it produced, over all 530 documents

| Field | Coverage | Brief's target | |
|---|---|---|---|
| `parties` (≥1 org) | **95%** | 70–90% | ✅ above |
| `defined_terms` | 89% | — | ✅ |
| `effective_date` | 84% | 50–75% | ✅ above |
| `governing_law` | 72% | 60–85% | ✅ |
| `contract_type` not Unknown | 91% | 65–85% | ✅ |

**And the label-set experiment, scored against CUAD's own title metadata on 488
documents:**

| Label set | Correct | Unknown |
|---|---|---|
| Brief's 9 labels | 60/488 = **12%** | 14% |
| Corpus-derived 26 labels | 340/488 = **69%** | 8% |

**A 57-point difference from changing nothing but the label names.** The failure
mode is visible in the raw predictions: with the brief's labels the model
assigned **MSA to 289 of 530 documents** — when the only available labels are
wrong, a zero-shot classifier does not say "none of these", it picks the least
wrong one and reports it confidently.

The remaining 31% of errors are informative rather than random:

| Count | Truth → Predicted | Why |
|---|---|---|
| 13 | License → Unknown | "Content License" / "Trademark License" collapsed into one `License` label |
| 9 | IP → Unknown | genuinely ambiguous category |
| 8 | Services → Consulting | semantically near-identical |
| 5 | Promotion → Marketing | near-synonyms |
| 5 | Supply → Manufacturing | near-synonyms |

Several of these are taxonomy problems, not model problems — `Promotion` vs
`Marketing` is a distinction the *title* makes and the contract body may not.
That caps achievable accuracy regardless of model quality, and is worth saying
out loud rather than chasing.

---

### Task 6 — Re-indexing with contextual prefixes (`index.py`)

Two changes from Day 1.

**1. Contextual prefix.** Chunking strips the context a clause needs. Retrieved
alone, "Either party may terminate for convenience on 30 days' notice" does not
say *what kind of contract* it terminates. So each chunk is embedded with a short
header:

```
Contract type: Sponsorship | Section 12 TERMINATION | Governing law: New York

12. TERMINATION. Either party may terminate...
```

The exact string used is stored in the payload as `embed_text`, so any vector can
be traced back to the text that produced it. The prefix is kept short
deliberately — it competes with the clause for the model's 512-token budget.

**2. Payload indexes.** Filtered vector search without an index on the filtered
field degrades to a full scan:

```python
client.create_payload_index("clauses", field_name="contract_type",
                            field_schema=models.PayloadSchemaType.KEYWORD)
```

Six fields are indexed: `contract_type`, `governing_law`, `source`, `doc_id`,
`chunking_strategy`, `section_heading`.

This is what makes Day 1's F6 solvable — "the governing law of **this**
agreement" becomes a `doc_id`-filtered search instead of a blend across 530
unrelated contracts.

**Verified after re-indexing:** 18,280 points, 6 payload indexes, and filtered
vector search confirmed returning hits only from the requested document.

---

### Task 7 — Comparing to Day 1 without an eval harness (`compare_day1_day2.py`)

The brief is explicit that today's comparison is **qualitative only** — quality
claims need Day 3's harness. So the comparison measures only things that can be
checked directly, without judgement.

The useful one: does a chunk **begin at a clause boundary** rather than
mid-sentence? That is a regex check over the whole chunk store, not an opinion:

| | Chunks starting at a clause-like boundary |
|---|---|
| Day 1, naive 850-char stride | 419 / 3,219 = **13%** |
| Day 2, structural | 11,451 / 18,280 = **62%** |

That is a real, defensible improvement — and note what it is *not*: it says
nothing about whether answers got better. Day 1's F7 (chunks cut mid-clause) is
now measured rather than anecdotal, which is the most that can be claimed today.

The 38% that still start mid-sentence are mostly the 30% of documents on the
fallback path, plus over-long structural sections that `split_long()` had to
break.

---

<a name="part-3"></a>
## Part 3 — Failures, causes, fixes

Numbering continues from Day 1 (F1–F8). Every one of these was found by
measuring something, not by reading the code.

### F9 — The chunker discarded 16.6% of the answers ✅ FIXED — the day's most important bug

**How it surfaced.** Not from an error. From a text-accounting check: sum the
characters in all chunks and compare to the corpus.

```
source chars          29,535,164
chars kept in chunks  28,644,496
net difference          -890,668  (-4%)
```

4% of the corpus had vanished. `MIN_CHARS` explained only 129,013 of it.

**Cause.** The structural branch builds its first span at
`matches[0].start()` — the first section heading. **Everything before that is
never chunked.** That is the preamble:

```
chars before the first section match: 1,832,665
mean preamble dropped per document:       6,008
```

(1.83M dropped against a 890k net loss, because the fallback path's 17% overlap
inflation was masking it. Two errors partially cancelling is why the net number
looked survivable.)

The preamble is not filler. It holds the parties, the recitals, the effective
date, and — in 25 documents — the governing law.

**Why this was about to ruin Day 3.** CUAD gives exact answer offsets, so the
damage is measurable:

```
answerable CUAD spans:                     13,823
spans falling inside the dropped preamble:  2,292  (16.6%)
=> Day 3 retrieval recall capped at 83.4%
```

Those spans existed in no chunk. **No retriever, no re-ranker, no embedding model
could ever have found them.** And the categories hit hardest:

| Spans lost | Category |
|---|---|
| 1,266 | Parties |
| 237 | Document Name |
| 233 | Agreement Date |
| 182 | Effective Date |
| 38 | Expiration Date |

Tomorrow's eval would have shown ~83% recall with a hard ceiling, and the obvious
reaction is to spend Day 4 tuning retrieval — chasing a ceiling that is not in the
retriever at all.

**Fix.** Emit the preamble as its own chunk, tagged `is_preamble`:

```python
if sections[0][0] > 0:
    spans.append((0, sections[0][0], None, "PREAMBLE"))
```

**Verified after:**

| | Before | After |
|---|---|---|
| Answer spans inside some chunk | 83.4% | **100.0%** |
| Corpus chars uncovered | ~4% | **0.00%** |

> **Interview line:** "I checked whether chunk characters summed to corpus
> characters and found 4% missing. It turned out the structural chunker started
> at the first numbered section, so every contract preamble was discarded — and
> because CUAD has character-level answer offsets I could price the damage
> exactly: 16.6% of annotated answers were in no chunk, capping retrieval recall
> at 83.4% before I'd written a single eval. That's the kind of bug that makes
> you tune the wrong component for a week."

---

### F10 — Governing law was searched in the wrong half of the document ✅ FIXED

The brief extracts metadata from `text[:20000]`:

```python
head = text[:20000]        # "metadata lives in the preamble"
```

True for the effective date. **False for governing law**, which sits in the
closing Miscellaneous section — often 100,000 characters in. A head-only search
structurally cannot find most governing-law clauses.

**Fix.** Effective date searches the head; governing law searches the whole text.

---

### F11 — Governing law kept returning the wrong jurisdiction ✅ FIXED — 84% → 98% dev

After F10 raised coverage to 73%, accuracy against 50 hand labels was 84%. The
error pattern was the interesting part — **four of eight errors predicted
"Delaware"**:

> "...a company organized under **the laws of the State of Delaware**, with
> offices at..."

That is a party's **incorporation jurisdiction**, not the governing law. Almost
every US corporation is Delaware-incorporated, so the preamble is full of this.

**Fix — exploit position.** Incorporation appears in the opening recitals;
governing law appears in the closing section. So stop taking the first match and
score all candidates:

```python
score  = 3 if within 400 chars after a "Governing Law" heading else 0
score += 2 * (match_position / document_length)      # later is better
score += 1 if preceded by govern/construed/interpreted/enforced
reject    if preceded by incorporated/organized/formed/existing
```

Two smaller fixes found by hand-labelling, both invisible from the code:

- **Case.** Two of fifty documents write "by the **L**aws of the State of
  Illinois". The patterns were lowercase-only, so those silently missed.
- **OCR damage.** One CUAD contract reads `the laws ofthe State ofTexas` — the
  spaces are gone. `\s*` rather than `\s+` between keywords absorbs it.

Also still-unfixed forms, deliberately left for Day 3 (see the overfitting note
below): `"construed in accordance with Ohio law"` puts the jurisdiction *before*
the word "law", which the pattern cannot match at all.

---

### F12 — I overfit my own accuracy number ✅ CAUGHT, and this is the story to tell

After F11, governing-law accuracy was **98% (49/50)**.

That number is not trustworthy, **and I knew why: I had used those same 50
documents to design the scoring rules.** They were a *dev set* being reported as
a *test set*. Tuning against a sample and then quoting your score on that sample
is one of the most common ways ML numbers become fiction.

So I labelled a **disjoint 30-document held-out set** (different seed, no overlap),
scored it once, and changed nothing afterwards:

| Set | Strict | Lenient | Status |
|---|---|---|---|
| Dev (50 docs, tuning happened here) | **98%** | 98% | inflated |
| **Held-out (30 docs, never tuned on)** | **86%** | **90%** | ← the honest number |

**A 12-point gap.** That gap *is* the overfitting, quantified.

The held-out errors are real and I am deliberately not fixing them, because
patching errors revealed by a test set silently converts it into another dev set.
They go on the Day 3 list:

| Truth | Predicted | Cause |
|---|---|---|
| Ohio | (missing) | "Ohio law" — jurisdiction before the noun |
| Ontario | Canada | "laws of the Province of Ontario **and the federal laws of Canada**" — my position bonus preferred the later match |
| California | (missing) | phrasing not covered |
| New York | New York State | pure normalisation |

Note the second one: **the position heuristic that fixed F11 caused a new error
here.** That is what a held-out set is for.

> **Interview line:** "I got governing-law extraction to 98%, then realised I'd
> tuned the rules on the same fifty documents I was scoring against. I labelled a
> disjoint held-out set and it came in at 86% — the 12-point gap was my
> overfitting. I quote 86%, and I left the held-out errors unfixed so the set
> stays valid."

---

### F13 — "Reply NONE if unsure" made a small model refuse everything ✅ FIXED

Party extraction returned **empty for 4 of the first 6 documents.** The window
was correct and the party names were plainly inside it — verified by hand.

**Cause: the escape hatch in the prompt.**

```
"...One per line, at most 5, no commentary. If none are identifiable, reply NONE."
```

`qwen2.5:3b` answered `NONE` for documents whose parties were right there. Same
prompt, same window, different phrasing:

| Prompt | contract_02 | contract_04 |
|---|---|---|
| "...If none are identifiable, reply NONE." | `NONE` | `NONE` |
| **"Which companies are entering into this agreement?"** | ✅ Advanced Lighting Technologies, Fiberstars | ✅ EdgeMode, CUDO Ventures |

**This is the third instance of the same pattern in this project.** Day 1 F1:
prohibitions ("do not invent labels") made the model produce the forbidden
output. Day 1 F4: an explicit refusal path was taken too readily. Here: an
explicit "reply NONE" was taken every time.

**Generalisation worth stating in an interview:** on a small instruction-tuned
model, *any* easy exit you name in the prompt raises the odds it is taken. State
the task positively; handle the empty case in code, where it is deterministic.

Two follow-on bugs found while fixing this:

- **The window was anchored at character 0.** EDGAR exhibits open with filing
  boilerplate (`EX-10.5 5 a05-19878_1ex10d5.htm MATERIAL CONTRACTS ...
  CONFIDENTIAL TREATMENT REQUESTED`), so `text[:3000]` can miss the recital.
  Fixed by anchoring the window on "by and between" and friends — **tried
  strongest-pattern-first**, because a bare `between` matches ordinary prose and
  sent one window 20,000 characters past the recital.
- **The model sometimes separates names with `<br>`** instead of newlines, which
  collapsed them into one over-long line that the length filter dropped. One
  document silently lost all three of its parties to this.

Result: 8/8 on the smoke sample, with correct party names.

---

### F14 — My own CUAD loader silently dropped a contract ✅ FIXED

First run reported **509** contracts; the dataset has 510.

**Cause.** My `doc_id` sanitiser truncated titles to 120 characters. Two Playboy
contracts differ only in a trailing `1`/`2` beyond that cutoff, so they collided
— and my "defensive" `if doc_id in seen: continue` silently dropped the
117,118-char one, keeping the 3,558-char one.

**Two lessons, and the second is the real one:**

1. Truncation is not a safe ID strategy. Long titles now get a hash of the full
   title appended.
2. **The `continue` was the actual bug.** A collision here means data loss, so it
   must be loud. It now raises. A defensive skip that hides a broken invariant is
   worse than a crash.

The same class of error bit `chunk_id` too: section numbers legitimately recur
within a contract (tables of contents, exhibits, restated sections), producing
**644 duplicate chunk_ids** until a span index was added. Duplicate IDs in a
vector store mean silently overwritten points.

---

### F15 — `MAX_CHARS` was not enforced ✅ FIXED

`split_long()` breaks text on sentence boundaries. If a single "sentence" exceeds
the limit — a table, a list, anything without terminal punctuation — it passes
through whole. Worst case: a **19,249-character chunk against a 4,000 limit.**

Fixed with a hard fallback split, folding a runt tail back into the previous
piece so a 4,001-char input does not emit a 1-character chunk. Consequence,
deliberate and documented: a merged piece may exceed the cap by up to
`MIN_CHARS` (observed max 4,192). Slightly over the cap beats a runt.

---

### What today did *not* fix

| Day 1 failure | Status after Day 2 |
|---|---|
| F3 — scores can't express relevance | Still open. Cross-encoder is Day 4. |
| F4 — citations sometimes missing | Still detected-not-enforced. Day 5. |
| F5 — near-duplicates waste top-k | Partly helped: clause chunks are less redundant than overlapping windows, and running-header noise persists. Not measured yet. |
| F6 — "this agreement" over many docs | **Now solvable** — `doc_id` payload index exists. Still a product decision. |
| F7 — chunks cut mid-clause | **Largely fixed** for the 70% chunked structurally; the 30% on fallback still cut arbitrarily. |

---

### F16 — The contextual prefix made the similarity floor *worse* ⚠️ measured, kept anyway

Day 1's F3 was that retrieval scores could not express relevance: random chunk
pairs averaged 0.643 and reached **p99 = 0.7787**, while the best relevant chunk
scored **0.7670** — *below the noise ceiling*.

Re-measured on Day 2:

| | random p99 | relevant top-1 | verdict |
|---|---|---|---|
| Day 1 | 0.7787 | 0.7670 | relevant **below** the floor ❌ |
| Day 2 | 0.7517 | **0.8130** | relevant **above** the floor ✅ |

The gap went from **−0.012 to +0.061**. That is a real improvement in
discriminability.

**But which change caused it?** Two things changed at once — clause chunking and
the contextual prefix — so I ablated: same 300 chunks, same model, only the
embedded string differs.

| Embedded string | mean | p90 | p99 |
|---|---|---|---|
| Day 1 naive chunks, no prefix | 0.6430 | 0.7060 | 0.7787 |
| Day 2 clause chunks, **text only** | **0.6180** | 0.6875 | **0.7454** |
| Day 2 clause chunks, **with prefix** | 0.6403 | 0.6981 | 0.7517 |

**The prefix moves the floor in the wrong direction.** It raises mean pairwise
similarity by 0.022 — which makes sense once stated: every chunk now begins with
the same boilerplate shape (`Contract type: … | Section … | Governing law: …`),
so all chunks look *more* alike. The entire improvement came from **clause
chunking**, and the prefix partially undid it.

**Why keep it anyway?** Because the floor is not the only thing that matters. The
prefix should help *relevance* — matching a query to the right clause — even
while raising background similarity. Those two effects trade off, and **nothing
available today can settle which dominates.** That is precisely a Day 3 question:
run the eval harness with and without the prefix and keep whichever wins.

What today establishes is that the brief's recommendation is **not free**, and
that the honest description of the change is "plausible, ablated, unresolved" —
not "improvement".

> **Interview line:** "The tutorial recommended a contextual prefix on each
> chunk. I ablated it and found it raised mean pairwise similarity by 0.022 —
> making chunks harder to tell apart, because they all start with the same
> header. The gain I'd assumed came from the prefix actually came from clause
> chunking. I kept the prefix, but as an open question for the eval harness
> rather than a win."

---

### F17 — Better retrieval, worse answer: F6 became blocking ✅ FIXED

Running `ask.py` unchanged on the new index:

```
[1] 0.8130  cuad_ChinaRealEstateInformationCorp…::s10.11   Governing Law
[2] 0.8114  cuad_KNOWLABS_INC…INTELLECTUAL_PROPERTY…::s12   Governing Law
[3] 0.8104  cuad_TUNIUCORP…COOPERATION_AGREEMENT::s9        Governing Law
[4] 0.8102  contract_15::s22.1                              Governing Law
[5] 0.8049  cuad_VARIABLESEPARATEACCOUNT…::s12              Governing Law

--- Answer ---
Not found in the provided excerpts.
```

**Retrieval is near-perfect — all five hits are genuine Governing Law clauses,
which Day 1 never achieved — and the answer got *worse*.**

The model was right to refuse. The question says "**this** agreement" and it was
handed the governing-law clauses of *five different agreements*, each naming a
different jurisdiction. There is no correct single answer. On Day 1 the same
question produced a confident, spliced, wrong answer (F2); today the same
ill-posed question produces an honest refusal. **The refusal is the better
behaviour, and it looks like a regression on a scoreboard.** Worth remembering
when reading Day 3's numbers.

**Fix:** Day 2's `doc_id` payload index makes scoping a one-line filter, so
`ask.py` gained `--doc` and `--contract-type`:

```
$ uv run python ask.py --doc cuad_ADAMSGOLFINC_…ENDORSEMENT_AGREEMENT \
    "What is the governing law of this agreement?"

Retrieved (scoped: cuad_ADAMSGOLFINC_…ENDORSEMENT_AGREEMENT):
  [1] 0.7322  APPLICABLE LAW  …::c018::s25::p0

--- Answer ---
[…::c018::s25::p0] The governing law of this agreement is the laws of the
State of Kansas.
```

Correct, cited, and **independently confirmed** — Kansas is what I hand-labelled
that document as, before any of this ran.

Note the scoped top-1 score is **0.7322**, well below the unscoped 0.8130. Within
one contract there are far fewer candidates, so absolute scores drop while the
answer gets *better*. Another reason absolute similarity is not a quality
measure (Day 1 F3).

---

<a name="part-4"></a>
## Part 4 — New concepts

Day 1's glossary still applies. These are new today.

| Term | Definition |
|---|---|
| **CUAD** | Contract Understanding Atticus Dataset — 510 contracts, 13,823 attorney-reviewed clause spans with character offsets, 41 categories, CC BY 4.0. |
| **SQuAD format** | `{context, question, answers:[{text, answer_start}]}`. `answer_start` is a character offset, which is what makes automatic eval possible. |
| **Answer span / offset** | The exact character range of a correct answer. Map it onto chunk ranges to know which chunk *should* be retrieved. |
| **`is_impossible`** | SQuAD flag for a question with no answer in the document. CUAD's 14,208 of these are the refusal-accuracy test set. |
| **Structural chunking** | Splitting on document structure (section numbers) rather than character count, so a chunk is one clause. |
| **Fallback strategy** | What to do when structure isn't detectable. Here: Day 1's fixed windows, recorded per chunk as `chunking_strategy` so metrics can be split by it. |
| **Preamble** | Everything before the first numbered section: parties, recitals, dates. Metadata-dense, and easy to drop by accident (F9). |
| **Denormalisation** | Copying document-level metadata onto every chunk so a vector search can filter on it without a join. Costs storage, buys filtering. |
| **Payload index** | A Qdrant index on a payload field. Without it, filtered search degrades to a full scan. |
| **Contextual prefix** | A short header prepended to a chunk *before embedding* (`Contract type: … | Section 12 TERMINATION`) restoring context that chunking stripped. |
| **Zero-shot classification** | Classifying into labels the model was never trained on, supplied at inference time. |
| **NLI / entailment** | Natural Language Inference: does text A imply statement B? Zero-shot classifiers pose each label as a hypothesis and rank by entailment. |
| **NER** | Named Entity Recognition — tagging spans as PERSON/ORG/LOC. |
| **`aggregation_strategy`** | Stitches word-piece entity tokens into whole entities. Without it, "Advanced Lighting Technologies" arrives in fragments. |
| **Taxonomy fit** | Whether your label set matches your data. The brief's 9 labels fit 6% of CUAD — a silent accuracy cap. |
| **Dev set vs test set** | Dev is what you tune against; test is scored once and never tuned on. Reporting a dev score as a test score is how numbers become fiction (F12). |
| **Held-out set** | A sample deliberately untouched during development, to measure generalisation. |
| **Overfitting (to a sample)** | Rules tuned so tightly to specific examples that they don't generalise. Measured here as the 12-point dev↔held-out gap. |
| **Provenance** | Who produced a label and how. "AI-labelled" and "attorney-verified" are different claims and must not be conflated on a CV. |

---

<a name="part-5"></a>
## Part 5 — Interview questions

**"How did you build your evaluation data?"**
> Two sources with deliberately different provenance. CUAD gives 13,823
> attorney-reviewed answer spans with character offsets, so for retrieval I can
> generate golden questions automatically by mapping offsets onto chunk ranges.
> For metadata I hand-labelled 80 documents — 50 as a dev set and 30 held out. I
> keep the provenance recorded in the label files, because AI-labelled ground
> truth and attorney-verified ground truth are different claims.

**"What was your worst bug?"**
> My structural chunker started each document at its first numbered section, so
> every contract preamble was silently discarded — 1.8 million characters, mean
> 6,000 per document. I found it by checking whether chunk characters summed to
> corpus characters; 4% were missing. Because CUAD has character offsets I could
> price it exactly: 16.6% of annotated answers were in no chunk, so retrieval
> recall was capped at 83.4% before I'd written an eval. After the fix, span
> coverage is 100% and text loss is 0%.

**"How do you know your metadata accuracy is real?"**
> Because I checked it wasn't. I got governing law to 98%, then realised I'd
> tuned the scoring rules on the same 50 documents I was scoring. I labelled a
> disjoint 30-document held-out set, scored once, and got 86%. I quote 86%. I
> also left the held-out errors unfixed, because patching them would turn the
> test set into another dev set.

**"Why regex for governing law instead of a model?"**
> The phrasing is formulaic, the output must be exact, and a regex is auditable
> when it's wrong. But it needed five fixes I'd never have predicted: the clause
> is at the end of the document not the start; the dominant error was picking a
> party's Delaware incorporation state, fixed by scoring candidates on position
> rather than taking the first match; two documents capitalise "Laws"; and one
> has OCR damage — "laws ofthe State ofTexas" with the spaces missing.

**"Your prompt says 'reply NONE if unsure' and you removed it. Why?"**
> Because it made a 3B model answer NONE for documents whose parties were
> plainly in the window. It's the third time this project that naming an easy
> exit in the prompt made the model take it — the first was a citation
> instruction where prohibitions produced the forbidden output. My rule now:
> state the task positively and handle the empty case in code, where it's
> deterministic.

**"You said NER but ran an LLM. Explain."**
> Hugging Face kept resetting the large-file transfers, so the NLI and NER
> weights never landed. I implemented both backends, made it a flag, and ran the
> local Ollama path. It's genuinely zero-shot classification, but it is not NLI
> entailment and not NER, and it gives no calibrated confidence — so the
> confidence threshold has no equivalent. I record the backend per document
> rather than letting the distinction get lost.

**"Why keep the naive chunker?"**
> It's the baseline. I can't claim structural chunking is better without a
> number, and the number needs something to beat. Same reason I scored both
> contract-type label sets instead of only the good one.

**"Why is 68% of your annotation data unanswerable?"**
> CUAD asks all 41 category questions of every contract, so most have no answer.
> That's not waste — it's the refusal test set. A system that confidently answers
> everything is broken, and those 14,208 questions have a known-correct answer of
> "not in this contract".

---

<a name="part-6"></a>
## Part 6 — Honest status and CV numbers

### What to put on a CV

Use these. They are all measured, and each is defensible under questioning.

| Claim | Value | Provenance |
|---|---|---|
| Corpus | 530 contracts, 29.5M chars | 510 CUAD + 20 SEC EDGAR |
| Clause chunks indexed | 18,280 | Qdrant, 384-dim BGE-small, 6 payload indexes |
| Structurally chunked | **70%** of documents | measured; EDGAR 85%, CUAD 69% |
| Chunks starting at a clause boundary | **62%** (from 13%) | regex check over all chunks |
| Annotated answer spans reachable | **100%** (from 83.4%) | CUAD offsets vs chunk ranges |
| Corpus text loss through chunking | **0.00%** | span-coverage accounting |
| **Contract-type classification** | **69%** (from 12%) | 488 CUAD docs, title-derived ground truth |
| **Governing-law extraction** | **86% strict / 90% lenient** | 30-doc **held-out** set (98% on dev — never quote that one) |
| Governing-law coverage | 72% of 530 documents | — |
| Effective-date coverage | 84% (from 38%) | — |
| Party extraction | 95% of documents (≥1 org) | LLM-based, not NER |

### The three numbers that carry an interview

1. **16.6% → 0%** of annotated answers unreachable (F9). Shows you audit data,
   not just code.
2. **12% → 69%** contract-type accuracy from fixing the label set, not the model
   (F-Task 4). Shows you diagnose before optimising.
3. **98% → 86%** when you stop scoring on your dev set (F12). Shows you can be
   trusted with your own metrics — which is rarer and more valuable than the
   other two.

### Status of every failure so far

| # | Failure | Status |
|---|---|---|
| F1 | Fabricated citation IDs | ✅ Fixed Day 1 |
| F2 | Cross-contract sentence splice | ✅ Fixed Day 1 (symptom of F1) |
| F3 | Scores can't express relevance | ⚠️ Materially better — relevant top-1 moved from **below** the noise ceiling (0.767 vs p99 0.779) to **above** it (0.813 vs p99 0.752). Still needs the Day 4 cross-encoder. |
| F4 | Citations sometimes missing | ⚠️ Detected, not enforced → Day 5 |
| F5 | Near-duplicates waste top-k | ✅ Much improved — top-5 now spans **5 distinct documents** on all four stock questions |
| F6 | Single-doc question, many-doc corpus | ✅ **Fixed** — `ask.py --doc` / `--contract-type`, backed by payload indexes |
| F7 | Chunks cut mid-clause | ✅ Largely fixed — clause-boundary starts **13% → 62%**, and 19/20 retrieved chunks |
| F8 | Ollama truncated 60% of the prompt | ✅ Fixed Day 1 |
| F9 | **Preamble dropped, 16.6% of answers unreachable** | ✅ **Fixed** — 100% span coverage |
| F10 | Governing law searched in the wrong half | ✅ Fixed |
| F11 | Incorporation state mistaken for governing law | ✅ Fixed — dev 84% → 98% |
| F12 | **Accuracy number overfit to its own dev set** | ✅ Caught; held-out 86% is the quoted figure |
| F13 | "Reply NONE" made the model refuse everything | ✅ Fixed |
| F14 | Loader silently dropped a contract | ✅ Fixed (and made loud) |
| F15 | `MAX_CHARS` not enforced | ✅ Fixed |
| F16 | Contextual prefix worsens the similarity floor | ⚠️ Ablated and kept; Day 3 decides |
| F17 | Good retrieval, refused answer (F6 blocking) | ✅ Fixed via `--doc` scoping |

### The three recurring themes

Worth naming out loud in an interview, because they generalise beyond this project:

1. **Silent failures dominate.** F8, F9, F10, F14 produced no error, no warning,
   and plausible-looking output. Every one was found by an accounting check —
   *do the totals reconcile?* — not by reading code or watching it run.

2. **Naming an easy exit in a prompt makes a small model take it.** Three
   instances now (Day 1 F1, Day 1 F4, Day 2 F13). State the task positively;
   handle the empty case in code.

3. **A number you tuned against is not a measurement.** F12. The 12-point
   dev↔held-out gap is the honest price of iterating on a sample.

4. **An intervention that sounds right can measure wrong.** F16 — the contextual
   prefix the brief recommends makes chunks *more* similar to each other, and the
   improvement I'd have credited to it actually came from clause chunking. The
   ablation took four minutes and changed the conclusion.

### Do not claim

- ❌ "NER-based party extraction" — the NER weights never downloaded; an LLM did it.
- ❌ "98% governing-law accuracy" — that's the dev set you tuned on. It's 86%.
- ❌ "Attorney-verified metadata labels" — CUAD's *clause spans* are
  attorney-reviewed; the 80 metadata labels are AI-generated and the label files
  say so.
- ❌ "Improved retrieval quality" — not measurable until Day 3's harness exists.
  Today's changes are *justified*, not yet *validated*. What you *can* say is
  structural: clause-boundary starts went 13% → 62%, top-5 spans 5 distinct
  documents, and relevant chunks now score above the corpus noise ceiling.
- ❌ "The contextual prefix improved retrieval" — the ablation says it raises the
  similarity floor. It is an open question, not a win.

### Tomorrow

Day 3 builds the golden eval set from CUAD offsets and the metrics runner.
Everything after it is measured against what gets built tomorrow — which is why
today's F9 mattered so much: an eval harness sitting on a corpus that can only
reach 83.4% of its own answers would have produced a plausible, stable, and
completely misleading number.

Carry these three open questions into Day 3:

1. **Ablate the contextual prefix** (F16). It is currently kept on faith.
2. **Fix the held-out governing-law errors** — `"Ohio law"` (jurisdiction before
   the noun), and the position heuristic preferring "federal laws of Canada" over
   "Province of Ontario". Then validate on a *third* sample, because patching
   held-out errors spends that set.
3. **Decide the product question** (F6). `--doc` scoping works, but the golden
   eval set cannot be written until you choose: per-contract Q&A, or cross-corpus
   comparison ("12 of 530 contracts specify Delaware law").

