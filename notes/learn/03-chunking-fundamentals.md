# 03 — Chunking: Size, Stride, Overlap

---

## 1. What it is

**Chunking** is splitting a document into the units you will embed, index and
retrieve. The chunk — not the document — is the atom of a RAG system.

## 2. Why it exists — two independent reasons

People usually give one. There are two, and they push in different directions.

### Reason 1: retrieval granularity

Suppose you embed each *whole* contract as one vector. Your average CUAD contract
is 52,563 characters. That single 384-dim vector must represent governing law,
indemnification, payment terms, termination, insurance — everything, averaged
together.

Now search for "governing law". Every contract contains a governing-law clause,
so every contract's average vector is mildly similar to the query. You cannot
discriminate. **The vector has been averaged into meaninglessness.**

> A vector represents its input. If the input is a whole contract, the vector
> represents "a contract" — which is exactly the anisotropy problem from module
> 00, self-inflicted.

### Reason 2: context budget

Even if document-level vectors worked, you must *send* the retrieved unit to the
model. A 52,563-character contract is ~13,000 tokens. Five of them exceeds the
context window. The retrieved unit has to be small enough to fit in the prompt.

**Both reasons push toward smaller chunks. What pushes back?** A chunk too small
loses the context needed to interpret it: `"...shall not exceed $1,000,000"` is
useless without knowing what "it" is. So chunk size is a genuine tradeoff, not a
value to minimise.

## 3. Intuition: the sliding window

The simplest scheme takes a fixed number of characters, then slides forward by
slightly less than that, so consecutive chunks share a tail.

```
text:      [==========================================================]
chunk 0:   [------ 1000 chars ------]
chunk 1:              [------ 1000 chars ------]
chunk 2:                         [------ 1000 chars ------]
                       ^^^^^^^^^^
                       150-char overlap
```

Two parameters:

- **`CHUNK_SIZE`** — how much text per chunk (1,000 characters here)
- **`OVERLAP`** — how much consecutive chunks share (150)

And one derived quantity that actually governs behaviour:

- **`stride = CHUNK_SIZE − OVERLAP = 850`** — how far the window advances

Chunk *n* begins at character `n × stride`. The stride, not the size, determines
how many chunks you get.

## 4. Why overlap exists

Without overlap, a clause that straddles a boundary is crippled in *both* pieces:

```
no overlap:
  chunk 4: "...the Supplier shall indemnify the Customer against all"
  chunk 5: "claims arising from breach of Section 7.2."
```

Neither chunk states the complete obligation. Embed either and you get a partial
meaning; retrieve either and the model sees half a sentence.

With 150 characters of overlap, the boundary region appears *whole* in one of
them:

```
with overlap:
  chunk 4: "...the Supplier shall indemnify the Customer against all claims arising"
  chunk 5: "against all claims arising from breach of Section 7.2."
```

**Overlap is insurance against arbitrary boundaries.** It is not free.

## 5. The arithmetic, verified against your corpus

Predicted chunk count is `total_chars / stride`. Your Day 1 EDGAR corpus:

| Quantity | Value |
|---|---|
| Corpus | 2,728,031 chars |
| `stride` | 850 |
| **Predicted** chunks (`2,728,031 / 850`) | ~3,209 |
| **Actual** chunks | **3,219** ✅ |
| Chunks exactly 1,000 chars | 3,197 (99%) — the rest are document tails |
| Stored text vs source text | 3,207,733 vs 2,728,031 |

The last row is the cost of overlap:

```
inflation = (3,207,733 − 2,728,031) / 2,728,031 = 17.6%
```

And the closed form:

```
inflation = OVERLAP / stride = 150 / 850 = 17.6%   ✅ matches
```

**Derive it once so you never guess again.** Each chunk covers `CHUNK_SIZE`
characters but advances only `stride`, so per stride of source you store
`CHUNK_SIZE` characters:

```
stored / source = CHUNK_SIZE / stride = CHUNK_SIZE / (CHUNK_SIZE − OVERLAP)
                = 1000/850 = 1.176  ->  17.6% more than the source
```

You store, embed and pay for 17.6% more text than you have. That is the price of
boundary insurance — and it scales badly: at `OVERLAP = 500`, stride is 500 and
inflation is **100%**. You would double your index for the same corpus.

## 6. Implementation

```python
def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap          # advance by the stride
    return chunks
```

Four things to notice:

1. **`text[start:start+size]`** — Python slicing clamps at the end, so the final
   chunk is short rather than an error. That's why 22 of your 3,219 chunks aren't
   exactly 1,000 characters.
2. **`start += size - overlap`** — the stride. Writing `start += size` here is
   the classic bug: it silently disables overlap while everything still runs.
3. **`overlap >= size` is an infinite loop** — the stride becomes ≤ 0 and `start`
   never advances. Worth an assertion in real code.
4. **Complexity** is O(n) in characters, O(n/stride) chunks. Chunking is never
   your bottleneck; embedding is.

And the identity that must hold:

```python
assert len(chunks) == math.ceil(len(text) / (size - overlap))
```

If that assertion fails, your stride logic is wrong. It is the cheapest possible
test of a chunker.

## 7. Choosing the parameters

There is no universal answer, but there is a reasoning process.

| If chunks are **too small** | If chunks are **too large** |
|---|---|
| lose interpretive context ("shall not exceed $1M" — what shall?) | dilute the vector across several topics |
| more chunks → bigger index, slower encode | fewer, coarser retrieval units |
| pronouns and cross-references dangle | may exceed the encoder's token limit and be silently truncated |
| precise retrieval, poor answers | vague retrieval, verbose answers |

Practical anchors:

- **Encoder limit is a hard ceiling.** BGE-small handles 512 tokens ≈ 1,536
  characters. A 4,000-character chunk is *truncated at embed time*, so a third of
  it never influences its own vector. Nothing warns you.
- **Match the semantic unit.** For contracts that's the clause; for papers, the
  paragraph or section; for code, the function. Which is module 04's argument.
- **Overlap 10–20% of chunk size** is a common default. Yours is 15% (150/1000),
  giving 17.6% inflation.

## 8. In this system

`chunk.py` — the Day 1 baseline, deliberately kept:

```python
CHUNK_SIZE = 1000
OVERLAP = 150
```

Results: **3,219 chunks** from 20 documents; per-document min 16, median 108, max
542 — the spread tracks real contract length variation.

**Why keep a chunker you've superseded?** Because module 04 replaces it and you
cannot claim an improvement without a baseline to measure against. Day 4's
ablation table only means something because `chunk.py` still exists.

## 9. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| `start += size` (overlap disabled) | boundary answers unfindable | assert the chunk-count identity (§6) | use the stride |
| `overlap >= size` | hangs | assert `overlap < size` | — |
| Chunk exceeds encoder limit | tail of long chunks invisible | tokenise and compare to `max_seq_length` | smaller chunks, or window (module 10) |
| **Mid-sentence cuts** | retrieved chunks start mid-word | regex for a clause-like opening | structural chunking (module 04) |
| Answer split despite overlap | span longer than the overlap | span-coverage check (module 05) | larger overlap, or structural chunks |
| Overlap too large | index and cost balloon | compute `OVERLAP/stride` | reduce overlap |

The **mid-sentence cut** is the one that matters most here, because you measured
it. A regex for "does this chunk begin at something that looks like a clause
boundary?" over all chunks:

| | Chunks starting at a clause-like boundary |
|---|---|
| Day 1, fixed 850-char stride | **419 / 3,219 = 13%** |
| Day 2, structural | 11,451 / 18,280 = **62%** |

87% of your naive chunks began mid-thought. That is not a cosmetic complaint:
a chunk beginning `"possession, directly or indirectly, of the power to
direct..."` cannot stand alone as evidence for a citation, even when retrieval
finds it correctly.

## 10. Compared with

| Strategy | Cuts on | Strength | Weakness |
|---|---|---|---|
| **Fixed-width + overlap** | character count | trivial, predictable count, works on any text | ignores meaning; 87% mid-sentence starts |
| **Sentence-window** | sentence boundaries | never cuts mid-sentence | clauses span many sentences |
| **Structural** (module 04) | section numbers/headings | chunk = semantic unit | needs detectable structure; 30% fell back |
| **Semantic** | embedding-similarity drops between sentences | adapts to content | costs an embedding pass over every sentence; opaque |
| **Recursive** (LangChain-style) | tries separators in order (`\n\n`, `\n`, ` `) | good general default | still ignores domain structure |

**Fixed-width is the right first implementation** — not because it's good, but
because it is a baseline you can beat and measure. Reaching for semantic chunking
first means you never learn what it bought you.

## 11. Interview angle

**"How did you choose your chunk size?"** — A weak answer names a number. A
strong one names the *constraints*: the encoder's 512-token limit is a hard
ceiling (beyond it the text is silently truncated), the semantic unit for
contracts is the clause, and overlap costs `OVERLAP/stride` in storage — 17.6% at
my settings. Then: I started fixed-width as a baseline and replaced it with
structural chunking, which took clause-boundary starts from 13% to 62%.

**"What does overlap actually cost?"** — Derive it: each chunk covers
`CHUNK_SIZE` but advances `stride`, so stored/source = `CHUNK_SIZE/stride`. At
1000/150 that's 17.6% more text stored, embedded and paid for. It also creates
near-duplicate adjacent chunks, which can consume top-k slots.

**"Would you use semantic chunking?"** — Only after measuring that structural
chunking is the bottleneck. It costs an embedding pass over every sentence and
produces boundaries you can't explain to a reviewer, and my structural chunker
already handled 70% of documents. I'd want the eval to show clause-boundary
failures are what's limiting recall before paying for it.

## 12. Test yourself

**Q1.** `CHUNK_SIZE = 800`, `OVERLAP = 200`, document is 10,000 characters. How
many chunks, and what is the storage inflation?

<details><summary>Answer</summary>

`stride = 800 − 200 = 600`. Chunks = `ceil(10000/600) = 17`.
Inflation = `OVERLAP/stride = 200/600 = 33%`.

Note how much worse that is than the project's 17.6% — raising overlap from 15%
to 25% of chunk size nearly doubled the inflation, because it's the *stride* in
the denominator, not the size.

</details>

**Q2.** A developer writes `start += size` instead of `start += size - overlap`.
Everything runs. What exactly is broken, and what single assertion would have
caught it?

<details><summary>Answer</summary>

Overlap is silently disabled: chunks are contiguous and non-overlapping, so any
clause straddling a boundary is split with no whole copy anywhere. Retrieval
still works, the count is *smaller* than expected (fewer chunks), and no error
occurs — so it presents as "recall is a bit low".

The assertion: `len(chunks) == ceil(len(text) / (size - overlap))`. With the bug
you get `ceil(len/size)` chunks, which for 1000/150 is 3,209 vs 2,729 — a 15%
discrepancy that fails loudly.

</details>

**Q3.** Why does embedding a whole 52,000-character contract as one vector make
retrieval *worse*, not just coarser? Connect it to module 00.

<details><summary>Answer</summary>

The vector represents its entire input, so a whole-contract vector encodes "a
commercial contract" — averaging governing law, indemnity, payment and
termination into one point. Since every contract contains roughly the same clause
types, all contract vectors converge toward the same region.

That is the anisotropy problem from module 00, deliberately worsened: the corpus
already has a 0.643 random-pair similarity floor because everything is legal
English; document-level pooling pushes that floor higher still. You lose
discriminative power, not just resolution — for "governing law" every document
scores nearly identically, so ranking becomes arbitrary.

</details>

**Q4.** Your chunks are 1,000 characters and your encoder truncates at 512
tokens. Is that a problem? Show the reasoning.

<details><summary>Answer</summary>

No — 1,000 characters is roughly 250 tokens at ~4 chars/token, comfortably inside
512. So the whole chunk influences its vector.

But it becomes a problem the moment chunk size grows. The project's structural
chunks reach 4,192 characters ≈ 1,050 tokens, and 36% of them exceed the
~1,536-character equivalent of 512 tokens. Those are silently truncated at embed
time. The check is arithmetic you should do *before* changing chunk size:
`max_chunk_chars / 4 < model_max_tokens`.

</details>

**Q5.** Harder. Overlap is meant to prevent an answer being split across chunks.
Give a concrete case where 150-character overlap fails to protect an answer, and
say what you'd do about it.

<details><summary>Answer</summary>

Overlap only protects spans **shorter than the overlap**. A 400-character
indemnification obligation crossing a boundary appears as a 250-char prefix in
one chunk and a 150-char suffix in the next; no chunk contains all 400
characters, so no chunk can serve as complete evidence.

Options, in increasing order of merit:
1. **Raise overlap above the typical answer length** — but inflation is
   `OVERLAP/stride`, so protecting 400-char answers at size 1000 means stride 600
   and 67% inflation. Expensive and it still fails on 700-char answers.
2. **Structural chunking** — cut at clause boundaries so answers rarely straddle
   at all, since a clause *is* the answer unit (module 04).
3. **Measure it rather than assume.** This is exactly what span coverage does:
   check every annotated answer against chunk ranges and count how many are
   fully contained. The project found 16.6% of answers unreachable — not from
   overlap being too small, but from a different bug entirely (module 05).

The general lesson: overlap is a blunt statistical hedge. Chunking on the
document's own structure removes the need for the hedge.

</details>

---

**Previous:** [02 — Ingestion & parsing](02-ingestion-and-parsing.md) ·
**Next:** [04 — Structural chunking](04-structural-chunking.md)
