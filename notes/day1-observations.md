# Day 1 — Observations

Corpus: 20 EX-10.x exhibits, 20 distinct filers, collected via `fetch.py`
(SEC EDGAR full-text search API). 3,219 chunks indexed in Qdrant.

## 1. The generator fabricates citation IDs from section numbers

Asked *"What is the governing law of this agreement?"*. The answer cited
`[Governing Law::27]` — an ID that **does not exist** in `chunks.json`.

Root cause: the retrieved chunk `contract_12::202` contains the literal heading
`27.15 Governing Law`. The model built a citation out of the *contract's own*
section numbering instead of the `chunk_id` I gave it, because both use the
`::`-adjacent `<name>` `<number>` shape. The prompt's example
(`[contract_01::3]`) wasn't enough to pin the format for a 3B model.

Fix candidates for Day 5:
- Validate every emitted citation against the retrieved ID set; retry or strip on miss.
- Relabel excerpts with opaque short tags (`[1]`, `[2]`) so there is nothing in
  the contract text to confuse them with, then map back to chunk_ids after.

## 2. Worse: it spliced one sentence out of two different contracts

The same answer read:

> "The construction, performance and validity of this Agreement shall be
> governed by the laws of the jurisdiction where the Premises are located..."

That sentence exists nowhere in the corpus. Verified by normalised substring
search over `documents.json`:

- `"construction, performance and validity of this agreement"` → **contract_12** only
- `"where the premises are located"` → **contract_19** only

So the first clause came from Healthways (contract_12 §27.15) and the second
from Goodman Networks (contract_19), welded into one fluent, wrong sentence.
This is the most dangerous failure mode found today: it is grammatical,
plausible, legally-shaped, and completely fabricated.

Implication for Day 3's eval harness: faithfulness cannot be scored by
keyword overlap — both halves individually overlap the corpus fine. It needs
span-level attribution against the *single* cited chunk.

## 3. "What are the termination provisions?" produced zero citations

Answer content was genuinely grounded (`contract_15::386`/`::400` really do
contain the Termination Assistance Services language) but it emitted no
`[...]` tags at all. So citation compliance is inconsistent run to run, not
uniformly broken. Two failure classes to score separately on Day 5:
citation *presence* vs. citation *correctness*.

## 4. The question is implicitly single-document; the corpus is not

Every test question says "**this** agreement" but retrieval spans 20 unrelated
MSAs. Top-5 for governing law pulled chunks from contract_09, _12 and _19 —
three different companies. The blended answer is arguably the honest output of
a badly-posed question. Needs either a doc filter (`doc_id` payload filter is
already indexed) or answers that group findings per contract.

## 5. Naive chunking cuts mid-sentence and orphans headings

`contract_09::8` opens with `"possession, directly or indirectly, of the power
to direct..."` — the definition it belongs to started in chunk 7. Retrieved
chunks also carry running-header noise
(`"Broadridge/Penson Proprietary and Confidential"`) inline with clause text,
because `get_text()` cannot tell a page header from a paragraph.
Confirms the Day 3 plan: chunk on section boundaries, strip repeated headers.

## 6. Retrieval scores are suspiciously flat

Top-5 cosine scores cluster in a narrow band (governing law: 0.767 → 0.726;
termination: 0.7717 → 0.7677 — a 0.004 spread across five different
contracts). With scores that undifferentiated there is no usable relevance
threshold and no signal for refusal. Strong argument for the Day 4
cross-encoder re-ranker, and for treating raw bi-encoder score as unusable
for abstention decisions.

## 7. Corpus collection gotchas (from automating Task 1)

- EDGAR full-text search returns the *file that contains the phrase*, so
  `file_type` must be filtered — of ~1,150 hits scanned, **953 were not EX-10.x**
  (EX-99.28 press releases, EX-4.13 securities exhibits, bare S-1/A cover forms).
- The same exhibit is re-attached across multiple filings, and one filer
  dominates results (Synacor appeared 6× in the first 19 hits). Without a
  per-CIK dedupe the "20-contract corpus" would have been ~6 companies.
  **158 hits dropped as duplicate filers.**
- 13 hits were under 50 KB (stubs/incorporation-by-reference, not contracts).
