# Day 1 — Failures, root causes, and evidence

Corpus: 20 EX-10.x exhibits, 20 distinct filers, via `fetch.py`. 3,219 chunks
in Qdrant. Generator: local `qwen2.5:3b`, temperature 0.

Diagnostics run after the first pass are recorded below each failure. Where a
cause is marked CONFIRMED, an experiment isolated it; HYPOTHESIS means it is
still inferred.

---

## F1 — Fabricated citation IDs  ·  CAUSE CONFIRMED  ·  fix is cheap

Asked *"What is the governing law of this agreement?"*, got `[Governing Law::27]`
— an ID absent from `chunks.json`.

**Cause: the label format collides with the contracts' own section numbering.**
Retrieved chunk `contract_12::202` contains the literal heading `27.15 Governing
Law`. The prompt asked for `name::number` IDs, the contract text supplies
`number name` headings, and the model pattern-matched onto the wrong one.

**Isolating experiment** — same question, same retrieved chunks, only the labels
changed:

| Labels | Cited | Valid? |
|---|---|---|
| `[contract_09::8]` … (current `ask.py`) | `[Governing Law::27]` | ✗ fabricated |
| `[1]`…`[5]` (opaque) | `[4]` | ✓ and correct |

Opaque labels fixed it outright. The model is capable of citing; the ID scheme
was the defect. Caveat: n=1 question, deterministic decode — Day 3's harness
must confirm across the golden set before this is called solved.

**Fix:** label excerpts `[1]`…`[k]` in the prompt, map back to `chunk_id` after
generation. Never show the model a label that resembles contract text.

---

## F2 — Sentence spliced across two contracts  ·  SYMPTOM OF F1, not its own bug

The bad answer read *"The construction, performance and validity of this
Agreement shall be governed by the laws of the jurisdiction where the Premises
are located"* — a sentence that exists nowhere in the corpus:

- `"construction, performance and validity of this agreement"` → **contract_12** only
- `"where the premises are located"` → **contract_19** only

Healthways welded to Goodman Networks, fluent and entirely fabricated.

**This was initially logged as an independent, higher-severity failure. It is
not.** In the opaque-label run the splice vanished: the model quoted
contract_19 alone and cited it correctly. Once the invented ID stopped being
tied to nothing, the text stayed anchored to one chunk. Same root cause as F1.

**Still true for Day 3:** faithfulness cannot be scored by keyword overlap —
both halves of that sentence overlap the corpus perfectly. It needs span-level
attribution against the *single* cited chunk.

---

## F3 — Retrieval scores cannot express relevance  ·  CAUSE CONFIRMED + QUANTIFIED

Top-5 scores cluster absurdly tightly (termination: 0.7717 → 0.7677 across five
different companies). Measured the corpus's own similarity floor to find out why:

| Measurement | Cosine |
|---|---|
| Random chunk pair, mean (300 chunks, 44,850 pairs) | **0.6430** |
| Random chunk pair, p90 | 0.7060 |
| Random chunk pair, p99 | **0.7787** |
| Real query, top-1 (governing law) | **0.7670** |
| Real query, top-1 (termination) | 0.7717 |
| Nonsense query, top-1 (`banana bicycle photosynthesis`) | 0.5779 |

**The best relevant chunk scores *below* the 99th percentile of two randomly
chosen chunks.** The embedding space is compressed because all 3,219 chunks are
the same register — contract boilerplate. In this corpus 0.77 does not mean
"highly relevant", it means "is legalese".

Two consequences, and they differ:

- Absolute score **can** gate off-topic queries — 0.578 vs 0.767 is a real gap.
- Absolute score **cannot** rank clause relevance, and **must not** drive
  abstention. Any threshold separating 0.767 from 0.779 is noise-fitting.

**Fix:** cross-encoder re-ranking (Day 4) is not an optimisation here, it is the
only thing that can produce a usable relevance signal. Also contributing: 1,000-
char windows put several topics in one vector, diluting the topical signal.

---

## F4 — Citation compliance is inconsistent, not uniformly broken

*"What are the termination provisions?"* emitted **zero** `[...]` tags, while
being genuinely grounded (`contract_15::386`/`::400` do contain the Termination
Assistance language).

**Cause: HYPOTHESIS —** instruction-following variance in a 3B model. Not the
label collision, since here it simply omitted citations rather than inventing
one.

**Fix:** this is a harness problem, not a prompt problem. Validate output
programmatically — if it contains no label from the valid set, retry once, then
refuse. Score citation *presence* and citation *correctness* as separate Day 5
metrics; they have different causes.

---

## F5 — Intra-document redundancy eats top-k slots  ·  earlier guess was WRONG

Governing-law top-5 spent 2 slots on contract_09 and 2 on contract_12 — 5 slots,
3 documents.

I assumed shared MSA boilerplate across filers. **Measured, and that is false:**
of sampled pairs above 0.95 similarity, **0 were cross-contract and 3 were
within-contract.** The redundancy is internal to each document:

| Pair | Sim | What it is |
|---|---|---|
| `contract_10::157` vs `::191` | 0.9900 | same entire-agreement clause, repeated by amendments |
| `contract_19::395` vs `::458` | 0.9430 | SEC confidential-treatment legend repeating on every page |
| `contract_15::445` vs `::0` | 0.9132 | running header `Sabre – MSA Exhibit 10.103 EXECUTION VERSION` |

Note the last two: **running headers and legends are embedded inside otherwise
useful chunks**, inflating similarity between unrelated parts of the same
document. That makes header stripping a retrieval fix, not a cosmetic one.

**Fix:** cap slots per `doc_id` (payload filter already indexed), or MMR for
diversity; strip repeated page headers/legends in the Day 3 parser.

---

## F6 — Every test question presupposes one document; the corpus has 20

All five stock questions say "**this** agreement". Retrieval spans 20 unrelated
MSAs, so a blended answer is the honest output of a badly-posed question.
**Cause: task framing, not a bug.** Either scope by `doc_id` first, or make the
answer format group findings per contract.

---

## F7 — Fixed-width chunking cuts mid-clause

`contract_09::8` opens *"possession, directly or indirectly, of the power to
direct..."* — the definition began in chunk 7 and the retrieved chunk cannot
stand alone as evidence. **Cause: 1,000-char windows ignore structure.**
Confirms the Day 3 plan (chunk on section numbers).

---

## Not a problem — do not spend Day 4 time here

Chunk hygiene is fine. Measured over all 3,219 chunks: **0** TOC/dot-leader
chunks, **0** exact-duplicate texts, 9 chunks under 75% alphabetic, 4 under 300
chars. The parser is not producing garbage; the failures above are retrieval and
generation failures.

---

## Priority for Day 3–5

1. **F1** — opaque citation labels + post-hoc mapping. Verified fix, ~10 lines. Kills F2 too.
2. **F3** — cross-encoder re-rank; never threshold on bi-encoder score for relevance.
3. **F4** — programmatic citation validation with retry-then-refuse.
4. **F5/F7** — structural chunking + header stripping + per-doc slot cap.
5. **F6** — decide the product question: one-contract Q&A or cross-corpus comparison.

## Corpus collection gotchas (from automating Task 1)

- EDGAR full-text search returns the *file containing the phrase*, so `file_type`
  must be filtered: of ~1,150 hits scanned, **953 were not EX-10.x** (EX-99.x
  press releases, EX-4.13 securities exhibits, bare S-1/A covers).
- One filer dominates results — Synacor appeared 6× in the first 19 hits.
  **158 hits dropped as duplicate filers**; without per-CIK dedupe the
  "20-contract corpus" would have been ~6 companies.
- 13 hits were under 50 KB — stubs and incorporation-by-reference, not contracts.
