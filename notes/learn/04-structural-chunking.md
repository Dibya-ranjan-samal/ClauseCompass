# 04 — Structural Chunking

---

## 1. What it is

Cutting a document at **its own structural boundaries** — numbered sections,
headings, articles — so that one chunk is one semantic unit, rather than one
arbitrary window of characters.

## 2. Why it exists

Module 03 measured the damage fixed-width chunking does: **87% of chunks began
mid-thought**. But a contract is not undifferentiated prose. It is explicitly,
deliberately structured:

```
7.  INDEMNIFICATION
7.1 Supplier's General Third Party Indemnity. Except for claims covered by...
7.2 Customer's Indemnity. Customer shall indemnify...
8.  LIMITATION OF LIABILITY
8.1 Neither party shall be liable for...
```

The author already told you where the boundaries are. Fixed-width chunking
ignores that and cuts at character 850 regardless.

**The principle generalises:** if your documents carry structure, use it.
Sections in contracts, headings in papers, functions in code, `<h2>` in HTML.
Structure is free supervision.

## 3. Intuition

```
FIXED-WIDTH (stride 850):
  |------------|------------|------------|------------|
  7.1 Suppli.. |..er's Gene | ..ral Thir | d Party In..
        ^ cuts mid-word, mid-clause, mid-sentence

STRUCTURAL:
  |-- 7.1 --------|-- 7.2 -----------|-- 8.1 --------|
  one clause      one clause         one clause
```

The right-hand version gives every chunk three properties the left one lacks:

1. It **starts** at something a human would call a beginning.
2. It **is** a complete thought — usable as standalone evidence for a citation.
3. It carries **metadata for free** — the section number and heading, which
   module 09 exploits.

## 4. The mechanism

### Step 1 — detect boundaries with a regex

```python
SECTION = re.compile(
    r"^\s*(?:(?:Section|Article|ARTICLE|SECTION)\s+)?"   # optional keyword
    r"(\d+(?:\.\d+)*)\s*[.\)]?\s+"                       # 7  or  7.2  or  7.2.1
    r"([A-Z][A-Za-z ,\-/]{2,80})?\s*[.:]?\s*$",          # optional heading text
    re.MULTILINE,
)
```

Read it in pieces:

| Fragment | Matches | Why |
|---|---|---|
| `^\s*` with `re.MULTILINE` | start of any line | headings sit on their own line |
| `(?:Section\|Article\|...)\s+`, optional | `Section 7.2`, `ARTICLE IV` | drafters vary |
| `(\d+(?:\.\d+)*)` | `7`, `7.2`, `7.2.1` | **capture group 1** — the number |
| `\s*[.\)]?\s+` | `7.` or `7)` | punctuation after the number |
| `([A-Z][A-Za-z ,\-/]{2,80})?` | `Indemnification` | **group 2** — optional heading. Must start uppercase |
| `$` | end of line | a heading is a *whole* line, not a number inside prose |

The `^...$` anchors are the load-bearing part. Without them, `"...as set forth
in 7.2 below"` matches and you split mid-sentence — the regex would *create* the
problem it exists to solve.

### Step 2 — turn boundaries into spans

Each match starts a chunk; the next match ends it.

```python
for i, m in enumerate(matches):
    start = m.start()
    end   = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    spans.append((start, end, m.group(1), (m.group(2) or "").strip()))
```

Note this is **non-overlapping and gapless** — chunk *i* ends exactly where
chunk *i+1* begins. No overlap is needed, because boundaries are meaningful
rather than arbitrary. That's a hidden bonus: you avoid module 03's 17.6%
inflation.

### Step 3 — fall back when there's no structure

```python
if len(matches) < 3:                    # no detectable structure
    strategy = "fallback"
    spans = [(i, min(i + 1000, len(text)), None, None)
             for i in range(0, len(text), 850)]
```

Some documents are unnumbered prose, scanned exhibits, or one-page letters. For
those, fixed-width is the honest answer. **Recording *which* strategy produced
each chunk** (`chunking_strategy` in the payload) is what makes the statistics in
§6 possible.

### Step 4 — bound the size

A section can be enormous — one Sabre exhibit ran 408,405 characters. So oversized
sections are split at sentence boundaries:

```python
MIN_CHARS, MAX_CHARS = 200, 4000
pieces = split_long(body, MAX_CHARS) if len(body) > MAX_CHARS else [body]
```

`MIN_CHARS` drops fragments — a chunk of `"8."` is not evidence.

## 5. In this system, with the numbers

`chunk_structural.py` over 530 documents:

| Metric | Value |
|---|---|
| Chunks | **18,280** |
| Documents chunked structurally | **70%** (EDGAR 85%, CUAD 69%) |
| Chunks starting at a clause boundary | **62%**, up from 13% |
| Corpus text lost | **0.00%** |

### The statistic that nearly lied to you

Reported naively, median chunk size is **1,000 characters** — comfortably inside
the 400–1,200 target. But split by strategy:

| Strategy | n | median | mean |
|---|---|---|---|
| structural | 13,237 | **1,554** | 1,900 |
| fallback | 5,043 | **1,000** | 986 |

The fallback chunks are *exactly* 1,000 because that is the window size, and they
drag the combined median onto the target. **Structural chunks are genuinely
larger than the brief anticipated** — because a real contract clause is larger
than 1,000 characters.

> Reporting the blended median would have hidden that. This is the same error as
> averaging EDGAR and CUAD retrieval scores together, or averaging over questions
> with and without gold chunks (module 11): **a mean over a mixture of
> populations describes neither.**

Always ask: *is this average taken over one population or several?*

## 6. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| Regex over-matches | chunks tiny; median far below target | median chunk size; sample 5 chunks | anchor with `^...$`; require uppercase heading |
| Regex under-matches | fallback rate high; huge chunks | `% structural` per strategy | loosen pattern; print unmatched docs' headings |
| **`MAX_CHARS` not enforced** | a few enormous chunks; encoder truncates | `max(len(c["text"]))` vs `MAX_CHARS` | assert the invariant |
| Preamble discarded | see module 05 — **the big one** | span coverage | emit text before the first match |
| Blended statistics | median looks fine, chunks aren't | group by `chunking_strategy` | report per population |
| Numbered lists inside a clause | one clause split into fragments | inspect chunks with no heading | require a heading, or a minimum length |

**On `MAX_CHARS`:** you shipped this bug (F15) and caught it with a one-line
assertion. The lesson is that a constant named `MAX_CHARS` is a *claim*, and a
claim should be asserted:

```python
assert max(len(c["text"]) for c in chunks) <= MAX_CHARS
```

## 7. The near-duplicate discovery

Structural chunking created a second-order effect worth studying, because the
first hypothesis was wrong.

Retrieval kept spending top-5 slots on the same documents. The natural
explanation: standard MSA boilerplate repeats *across* filers, so many contracts
contain near-identical clauses.

Measured, over sampled chunk pairs above 0.95 similarity:

| | count |
|---|---|
| cross-contract pairs | **0** |
| within-contract pairs | **3** |

The hypothesis was false. The redundancy is *internal* to each document:

| Pair | Sim | What it actually is |
|---|---|---|
| `contract_10::157` vs `::191` | 0.9900 | the same entire-agreement clause, repeated by amendments |
| `contract_19::395` vs `::458` | 0.9430 | SEC confidential-treatment legend on every page |
| `contract_15::445` vs `::0` | 0.9132 | running header `Sabre – MSA Exhibit 10.103 EXECUTION VERSION` |

The last two matter most: **running headers and page legends are embedded inside
otherwise-useful chunks**, inflating similarity between unrelated parts of the
same document. That makes header stripping a *retrieval* fix, not a cosmetic one
— and it changes the remedy from "dedupe across documents" to "cap slots per
`doc_id`, and strip repeated page furniture".

## 8. Compared with

| | Fixed-width (module 03) | Structural (this module) |
|---|---|---|
| Boundary | character count | document's own headings |
| Overlap needed | yes, ~15% | no — boundaries are meaningful |
| Storage inflation | 17.6% | 0% |
| Chunk = semantic unit | no | yes, when structure is detected |
| Metadata produced | none | section number + heading, free |
| Size predictability | exact | varies with the document |
| Works on unstructured text | always | needs a fallback (30% here) |

## 9. Interview angle

**"How do you chunk documents?"** — Name the progression and the measurement:
fixed-width as a baseline, then structural chunking on section numbers, which
took clause-boundary starts from 13% to 62% and reduced storage inflation from
17.6% to zero. Then the honesty: 30% of documents had no detectable structure and
fell back to fixed-width, and I recorded which strategy produced each chunk so I
could report statistics per population.

**"What was subtle about your chunk-size statistics?"** — A strong story. The
blended median was 1,000 characters, inside my target — but that was an artifact
of mixing two strategies. Fallback chunks are exactly 1,000 by construction and
dragged the median onto target; structural chunks were actually median 1,554.
Averaging over a mixture described neither population.

**"You said boilerplate repeats across contracts — did you check?"** — Yes, and I
was wrong. Zero cross-contract pairs above 0.95, three within-contract. The
redundancy was intra-document: amendments repeating clauses, plus running headers
and confidentiality legends embedded inside good chunks. That changed the fix from
cross-document dedupe to per-document slot caps and header stripping.

## 10. Test yourself

**Q1.** Why do the `^` and `$` anchors matter in the section regex? Give a
concrete string that breaks without them.

<details><summary>Answer</summary>

Without anchors the pattern matches a section reference *inside* prose. For
example `"...as more fully set forth in 7.2 Indemnification below,"` contains
`7.2 Indemnification`, so the regex would declare a boundary mid-sentence and
split the clause there.

With `^\s*...$` and `re.MULTILINE`, a match must occupy an entire line — which is
how real headings appear. The anchors are what stop the boundary detector from
manufacturing the mid-sentence cuts it was built to eliminate.

</details>

**Q2.** Structural chunking needs no overlap. Why not, and what do you save?

<details><summary>Answer</summary>

Overlap is insurance against *arbitrary* boundaries — it hedges the risk that a
cut lands mid-thought. Structural boundaries are chosen by the document's author
to separate ideas, so a clause is whole within one chunk by construction and there
is no boundary risk to hedge.

You save `OVERLAP/stride` in storage, embedding compute and index size — 17.6% at
the Day 1 settings — and you avoid creating near-duplicate adjacent chunks that
compete for top-k slots.

</details>

**Q3.** Your chunker reports `median 1000, target 400–1200 ✅`. What single
follow-up question should you ask before believing it?

<details><summary>Answer</summary>

**"Is that median over one population or several?"**

Here it was two: 13,237 structural chunks (median 1,554) and 5,043 fallback
chunks (median exactly 1,000, because that's the window size). The fallback
chunks pulled the blended median onto target, hiding that structural chunks
overshot it substantially. Group by strategy and report separately.

</details>

**Q4.** You find 20% of documents falling back to fixed-width. Give two
diagnostics you'd run before loosening the regex.

<details><summary>Answer</summary>

1. **Print the first 30 lines of several unmatched documents** and look at how
   their headings are *actually* formatted — roman numerals (`ARTICLE IV`),
   lettered (`(a)`), all-caps with no number, or genuinely unstructured prose. You
   cannot fix a pattern without seeing what it failed on.
2. **Check whether fallback documents differ systematically** — are they short
   letters, amendments, scanned exhibits? If they are genuinely unstructured, the
   regex is correct and loosening it would produce false boundaries in the other
   80%.

The failure mode of loosening blindly is over-matching, which turns clauses into
fragments across the *whole* corpus to rescue a minority. Measure the median
chunk size before and after any regex change.

</details>

**Q5.** Harder. You measure two chunks from the *same* contract at 0.99 cosine
similarity. Give three distinct explanations and how you'd distinguish them.

<details><summary>Answer</summary>

1. **Genuine repetition** — an amended contract restating the same clause (the
   entire-agreement clause appearing in the original and in each amendment).
   *Distinguish:* read both; the prose is legitimately near-identical and both are
   real clauses.
2. **Page furniture inside chunks** — a running header, footer or
   confidentiality legend repeated on every page, embedded in otherwise-unrelated
   chunks. *Distinguish:* the chunks' *substantive* content differs while sharing
   a boilerplate prefix or suffix; find the repeated line by counting the most
   frequent lines in the document.
3. **Overlap artifact** — adjacent chunks sharing a large fraction of text
   because overlap is high relative to stride. *Distinguish:* check whether the
   chunk indices are consecutive and whether the shared span equals `OVERLAP`.

They demand different fixes: (1) per-`doc_id` slot caps or MMR at retrieval time,
(2) header stripping in the parser, (3) reduce overlap. Diagnosing which you have
before choosing a fix is the whole point — the project's first hypothesis was
"cross-contract boilerplate", which measurement ruled out entirely.

</details>

---

**Previous:** [03 — Chunking fundamentals](03-chunking-fundamentals.md) ·
**Next:** [05 — Span coverage](05-span-coverage.md)
