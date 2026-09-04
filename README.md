# ClauseCompass

Question answering over real commercial contracts, with citations you can verify
against the source text.

**Status: Day 5 of a 7-day build.** Retrieval is measured against a 148-question
golden eval set; answers are validated in code and scored for faithfulness and
refusal accuracy. No quality claim in this README goes beyond what has actually
been measured — including the parts that did not work.

---

## What it does

```
SEC EDGAR + CUAD  ->  parse  ->  clause chunking  ->  metadata extraction
                                                              |
                                        BGE-small embeddings, Qdrant (18,280 pts)
                                                              |
                    question -> filtered vector search -> top-k -> LLM -> cited answer
```

```bash
uv run python ask.py "Is there an indemnification clause?"

# scoped to one contract - the corpus holds 530, so unscoped "this agreement"
# questions are ill-posed and the model correctly refuses them
uv run python ask.py --doc cuad_ADAMSGOLFINC_03_21_2005-EX-10.17-ENDORSEMENT_AGREEMENT \
  "What is the governing law of this agreement?"
```

Every answer is labelled `[1]`…`[k]` internally and rewritten to real chunk ids
before display, and every emitted citation is validated against the retrieved
set. Uncited or fabricated citations are reported, not hidden.

---

## Corpus

| Source | Docs | Chars | Notes |
|---|---|---|---|
| [CUAD](https://www.atticusprojectai.org/cuad) | 510 | 26,807,133 | 13,823 attorney-reviewed clause spans, 41 categories, CC BY 4.0 |
| SEC EDGAR | 20 | 2,728,031 | EX-10.x exhibits, 20 distinct filers, collected by `fetch.py` |
| **Total** | **530** | **29,535,164** | |

CUAD citation: Hendrycks et al., *CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review*, NeurIPS 2021.

---

## Measured results

Every number below is reproducible from the scripts in this repo. Provenance is
stated because it matters.

### Chunking

| Metric | Value |
|---|---|
| Clause chunks indexed | 18,280 |
| Documents chunked structurally | 70% (EDGAR 85%, CUAD 69%) |
| Chunks starting at a clause boundary | **62%**, up from 13% with fixed-width windows |
| CUAD answer spans reachable by retrieval | **100%**, up from 83.4% |
| Corpus text lost in chunking | **0.00%** |

The 83.4% → 100% figure is the most consequential number here. The structural
chunker originally began each document at its first numbered section, silently
discarding every contract preamble — 1,832,665 characters holding the parties,
recitals and dates. Because CUAD supplies character-level answer offsets, the
damage was measurable exactly: 16.6% of annotated answers existed in no chunk, so
retrieval recall was capped at 83.4% before any eval existed.

### Retrieval ablation

The winning configuration **differs by scope**, which is the main Day 4 finding.

**`doc_scoped`** — filter to the named contract, then retrieve:

| Configuration | R@5 | R@50 | MRR | nDCG@5 | p95 ms |
|---|---|---|---|---|---|
| dense only (Day 3 baseline) | 0.575 | 0.979 | 0.442 | 0.425 | 29 |
| BM25 only | 0.411 | 0.980 | 0.319 | 0.285 | 56 |
| hybrid RRF | 0.481 | 0.987 | 0.394 | 0.356 | 87 |
| hybrid + cross-encoder | 0.606 | 0.987 | 0.435 | 0.431 | 12,046 |
| **hybrid + cross-encoder + entity-stripped query** | **0.762** | 0.987 | **0.657** | **0.646** | 11,599 |

**`corpus_wide`** — search all 18,280 chunks:

| Configuration | R@5 | R@50 | MRR | nDCG@5 | p95 ms |
|---|---|---|---|---|---|
| **dense only (still the winner)** | **0.363** | 0.585 | **0.315** | **0.289** | **37** |
| BM25 only | 0.223 | 0.547 | 0.207 | 0.172 | 58 |
| hybrid RRF | 0.338 | **0.604** | 0.276 | 0.252 | 94 |
| hybrid + cross-encoder | 0.331 | 0.604 | 0.263 | 0.243 | 11,080 |

**Corpus-wide did not improve.** BM25 raised the candidate ceiling (recall@50
0.585 → 0.604) but ranked poorly enough that fusion lowered recall@5, and the
cross-encoder cannot rank its way out of a pool that is only 60% correct. That
negative result is kept because it identifies the binding constraint: corpus-wide
recall@50 is 0.604, so 40% of gold chunks are never retrieved at all.

**The mechanism behind the split.** Golden questions name the contract ("Under
the *Adams Golf Endorsement* agreement, …"), but only **21%** of gold clauses
contain that party name. Corpus-wide, the cross-encoder needs the name to
identify the document; document-scoped, the payload filter already did that, so
the name is noise competing with clause semantics. Stripping it from the
re-ranking query is worth **+0.172 recall@5 doc-scoped and −0.173 corpus-wide** —
the same one-line change, opposite signs. The Day 2 contextual prefix shows the
same pattern (+0.019 corpus-wide, −0.009 doc-scoped), which is what makes it a
mechanism rather than noise.

**Per-type recall@5** — BM25 did *not* rescue the near-miss traps, contrary to
expectation; the cross-encoder did:

| Configuration | single_hop | near_miss | multi_hop |
|---|---|---|---|
| dense, doc_scoped | 0.577 | 0.500 | 0.625 |
| BM25, doc_scoped | 0.421 | **0.250** | 0.354 |
| hybrid + rerank + strip, doc_scoped | 0.756 | **0.917** | 0.708 |

A near-miss trap is legally *adjacent* language, so it shares vocabulary with the
question — exactly what lexical matching cannot separate.

**Parameter sweep:** `rrf_k=10` beat the standard 60 (0.356 vs 0.338). Reducing
BM25 candidates measured monotonically **worse** (0.297 at 5 → 0.356 at 50),
contradicting the usual "BM25 noise crowds out dense hits" advice — the problem
was BM25's ordering, not its volume.

### Day 3 dense baseline (for reference)

Scored on 148 golden questions; the 20 with no gold chunks (18 unanswerable,
2 aggregation) are excluded from aggregates because every metric returns 1.0 for
empty gold.

| Config | recall@5 | recall@10 | recall@50 | hit@5 | MRR | p95 |
|---|---|---|---|---|---|---|
| `corpus_wide` — search all 530 contracts | **0.3632** | 0.4526 | 0.5849 | 0.4375 | 0.3148 | 39.9 ms |
| `doc_scoped` — filter to the named contract | **0.5751** | 0.7107 | 0.9789 | 0.6797 | 0.4421 | 36.6 ms |

**The two configs have different bottlenecks, and that is the useful finding.**
Document-scoped, recall@50 is 0.979 against recall@5 of 0.575 — the right chunk
is retrieved and mis-ranked, which is a **ranking** problem that re-ranking
should fix. Corpus-wide, recall@50 is only 0.585, so 41% of gold chunks are never
retrieved at all — a **retrieval** problem that re-ranking cannot touch, since it
only reorders what was already fetched.

### Generation: validation, faithfulness and refusal

Answers are JSON with a verbatim quote per claim, and **every claim is validated
in code** before a user sees it: the cited label must be one we actually
retrieved, and the quote must string-match its chunk at ≥95%. An answer whose
every claim fails validation becomes a refusal.

Measured over all 148 golden questions — generator `qwen2.5:3b`, judge
`qwen2.5:7b`, fully local, $0:

| Metric | Value |
|---|---|
| **Correct refusal rate** | **1.000** (18/18 unanswerable refused) |
| **False answer rate** | **0.000** |
| Over-refusal rate | 0.469 |
| **Faithfulness** | **0.493** (hallucination rate 0.507) |
| **Judge agreement vs 40 hand labels** | **95%** — 1 lenient, 1 strict |
| Citation validation fires | **64** (60 bad citation, 4 bad quote) |
| Claims raw → kept | 191 → 127 (33% stripped) |
| JSON parse failure rate | 0.007 |
| p95 end-to-end latency | 34.1 s |

**The most important result: the deterministic and semantic checks catch disjoint
failures.** All 127 kept claims passed verbatim quote validation, and the judge
still rejected **61 of them (48%)** — the model quotes accurately and then draws a
conclusion the quote does not support. Shipping only the cheap deterministic check
would have let the system report "all citations verified" while hallucinating in
half its claims.

**Over-refusal, attributed rather than assumed.** 61 of 130 answerable questions
were refused: **57% because the validator stripped every claim** (fabricated
citations, often contract section numbers like `7.3.1(v)`), 40% because the model
returned no claims, 1% parse failure. The standard advice — "soften the refusal
instruction" — would have addressed only the minority and raised the false-answer
rate from its current zero.

**Model capacity is the bottleneck, not the pipeline** (60-question sample, same judge):

| Generator | Faithfulness | Hallucination | Validator fires | JSON parse fail | p95 |
|---|---|---|---|---|---|
| qwen2.5:3b | 0.414 | 0.586 | 36 | 0.000 | 38.7 s |
| **qwen2.5:7b** | **0.885** | **0.115** | **12** | 0.183 | 347.5 s |

Bigger is not uniformly better: 7b more than doubles faithfulness but is **worse
at JSON** (18% parse failure vs 0%), and at 347 s p95 on 8 GB RAM it is
unshippable — swap peaked at 9.3 GB of 10.2 GB.

### Golden eval set

| | |
|---|---|
| Size | **148** questions across **127** distinct contracts |
| Composition | 118 single-hop, 18 unanswerable, 6 near-miss traps, 4 multi-hop, 2 aggregation |
| Unanswerable share | 12% |
| Generation-stage rejection | 6.3% |
| Review-stage rejection | 1.7% (2 of 120 sampled) |

Built from CUAD's character-level answer offsets mapped onto chunk ranges. A
chunk is only gold if it **both** overlaps the annotated span **and** contains
the answer text — overlap alone proves the arithmetic lines up, not that the text
survived chunking. Only 3 of 4,256 generated questions failed that check, a
direct dividend of the Day 2 preamble fix.

Questions are **entity-qualified** ("Under the *Adams Golf Endorsement*
agreement, which state's law governs it?"). The generic CUAD phrasing is
byte-identical across all 530 contracts, which makes corpus-wide retrieval
mathematically impossible — measured at hit@5 = 0/30. Naming the contract raises
that to 4/30, giving a baseline that can actually move.

The 18 unanswerable questions are **verified**, not assumed: each ships with the
terms that must be absent, and is only assigned to a contract where none of them
appear anywhere in the full text. Independently re-checked: 0 leaks.

### Metadata extraction

| Field | Coverage (530 docs) | Accuracy |
|---|---|---|
| Contract type | 91% classified | **69%** on 488 CUAD docs |
| Governing law | 72% | **86% strict / 90% lenient** on a 30-doc held-out set |
| Effective date | 84% | not scored |
| Parties (≥1 org) | 95% | not scored |
| Defined terms | 89% | not scored |

**Contract type, 12% → 69%.** The classifier is unchanged; only the label set
moved. A nine-label taxonomy written for an MSA corpus matched 6% of CUAD, and a
zero-shot classifier given only wrong labels does not abstain — it assigned "MSA"
to 289 of 530 documents. Ground truth is CUAD's own filename metadata, which is
independent of the contract body the classifier reads, and covers all 488
canonicalisable documents rather than a sample.

**Governing law: quote 86%, not 98%.** The scoring rules reached 98% on the
50-document set they were tuned against. A disjoint 30-document held-out set
scored 86%. The 12-point gap is the overfitting. The held-out errors are
deliberately left unfixed so the set stays valid.

---

## Honest caveats

Read these before reusing any number.

- **No retrieval quality metric yet.** Day 3 builds the eval harness. Today's
  changes are justified and structurally verified, not validated.
- **Party extraction is not NER.** The `dslim/bert-base-NER` and
  `typeform/distilbert-base-uncased-mnli` weights could not be downloaded —
  Hugging Face repeatedly reset the large-file transfers. Classification and
  party extraction run through a local Ollama model (`qwen2.5:3b`) instead. Both
  backends are implemented and selected with `--backend`; every row of
  `metadata.json` records which one produced it. This is genuinely zero-shot
  classification, but it is not NLI entailment, it is not NER, and it yields no
  calibrated confidence score.
- **The 80 metadata labels are AI-generated,** not attorney-verified. CUAD's
  *clause spans* are attorney-reviewed; the governing-law labels in
  `data/interim/labels_*.json` were produced by an AI reading the documents, and
  each file records that in its `_provenance` field.
- **The contextual prefix is unresolved.** Prepending contract type and section
  heading before embedding was ablated and *raises* mean pairwise chunk
  similarity by 0.022, making chunks harder to distinguish. It is kept as an open
  question for the eval harness, not as an improvement.
- **Retrieval scores are not a quality signal.** Random chunk pairs in this
  corpus average 0.640 cosine similarity, because every chunk is the same
  register. A score of 0.75 means "is legalese", not "is relevant".
- **The golden set is CUAD-shaped.** 118 of 148 questions come from CUAD's 41
  annotation categories (20 templated, 21 not), so it reflects what attorneys
  chose to annotate rather than the full space of contract questions.
- **The eval set is AI-reviewed, not attorney-verified.** CUAD's clause spans are
  attorney-reviewed; the sampling review, the 30 hard cases and every
  keep/delete decision are mine. `evals/golden/review_decisions.json` records
  the method in its `_provenance` field.
- **Entity-qualified questions are an intervention.** Real users would not always
  name the contract, so `corpus_wide` measures retrieval *given a named
  contract* — easier than a cold query, harder than document-scoped.
- **Hard-case samples are small.** near-miss n=6, multi-hop n=4, aggregation n=2.
  Directional only — the 0.917 near-miss result rests on six questions.
- **The doc-scoped winner is not shippable as-is.** 11.6 s p95 against 29 ms for
  dense. It needs a smaller cross-encoder, fewer candidates, or quantisation.
- **BM25 tokenization drops unhyphenated variants.** Keeping compounds intact
  means `most-favored-nation` scores 0.00 while `most favored nation` scores
  22.79. No scored question depends on it, so it is recorded rather than fixed.
- **The `rerank_candidates` sweep was not run** (20/50/100). Three ~20-minute
  cross-encoder runs to tune a component that is a net loss corpus-wide and
  already near its ceiling doc-scoped. Deliberately skipped, not overlooked.
- **The hallucination rate is 0.507 on the shipped 3B model.** That is the honest
  headline. 0.115 with a 7B generator, but on a 60-question sample and at 347 s
  p95. The pipeline is not the limiting factor; model capacity is.
- **The 40 judge labels are AI-generated,** not attorney-verified — recorded in
  `evals/judge_labels.json` under `_provenance`. The 95% agreement figure must
  travel with every faithfulness number.
- **The judge is the same model family as the generator** (qwen2.5), which risks
  shared blind spots. A different family would be a stronger design.
- **The 7b comparison cannot measure refusal:** its 60-question sample contains no
  unanswerable questions, which sit at the end of `golden.jsonl`.

---

## Running it

```bash
docker compose up -d                  # Qdrant on :6333
ollama serve && ollama pull qwen2.5:3b

curl -sL -o data/external/CUAD_v1.json \
  https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json

uv run python fetch.py                # 20 EX-10.x exhibits from SEC EDGAR
uv run python ingest_cuad.py          # 510 contracts + 13,823 annotations
uv run python build_corpus.py         # merge into documents.json
uv run python chunk_structural.py     # 18,280 clause chunks
uv run python extract_metadata.py     # metadata + chunks_enriched.json  (~90 min)
uv run python index.py                # embed + index + payload indexes  (~5 min)
uv run python eval_metadata.py        # metadata accuracy report

uv run python build_golden.py                              # 148-question golden set
uv run pytest tests/ -q                                    # metric unit tests
uv run python -m clausecompass.eval.runner --retriever dense --scope both
uv run python -m clausecompass.eval.runner --retriever hybrid_rerank --scope doc_scoped
uv run python ablation.py sweep-rrf && uv run python ablation.py table

uv run python precompute_hits.py rerank                       # retrieve once (~20 min)
uv run python -m clausecompass.eval.gen_runner --config final # generation eval
uv run python judge_agreement.py score                        # judge validation

uv run python ask.py "your question"
```

`data/raw/` is committed so results stay reproducible (EDGAR relevance ranking
drifts). `data/interim/` is regenerable and ignored, except the two hand-label
files, which are not.

---

## Layout

| File | Role |
|---|---|
| `fetch.py` | SEC EDGAR collection via the full-text search API |
| `ingest_cuad.py` | CUAD_v1.json → documents + annotations |
| `build_corpus.py` | merge EDGAR + CUAD under a `source` field |
| `chunk_structural.py` | clause-boundary chunking |
| `chunk.py` | Day 1 fixed-width chunker, kept as the baseline to beat |
| `extract_metadata.py` | regex + zero-shot classification + party extraction |
| `eval_metadata.py` | accuracy scoring, dev and held-out |
| `index.py` | embedding + Qdrant indexing + payload indexes |
| `ask.py` | retrieval + cited generation, with `--doc` scoping |
| `compare_day1_day2.py` | structural comparison against Day 1 |
| `build_golden.py` | golden eval set: generate, stratify, review, hard cases, merge |
| `src/clausecompass/eval/metrics.py` | recall@k, hit@k, MRR, nDCG |
| `src/clausecompass/eval/runner.py` | eval runner; takes any `Retriever`, both scopes |
| `src/clausecompass/retrieval/` | `base` (protocol), `dense`, `sparse` (BM25), `fusion` (RRF), `rerank` |
| `src/clausecompass/config.py` | winning parameters per scope, with the measured reasons |
| `ablation.py` | parameter sweep and the ablation tables |
| `src/clausecompass/generation/llm.py` | Ollama access + SQLite response cache |
| `src/clausecompass/generation/answer.py` | grounded prompt + deterministic claim validation |
| `src/clausecompass/eval/gen_runner.py` | refusal, validation and faithfulness metrics |
| `precompute_hits.py` | retrieve the golden set once so generation evals are pure LLM work |
| `judge_agreement.py` | sample and score the judge against hand labels |
| `evals/judge_labels.json` | 40 hand-labelled claim/excerpt pairs + provenance |
| `evals/golden/golden.jsonl` | the eval set — the most valuable artifact here |
| `evals/baseline.json` | recorded dense-only baseline |
| `notes/day1-explained.md` | how the Day 1 stack works, from scratch |
| `notes/day2-explained.md` | how the Day 2 stack works, plus every failure and fix |
| `notes/day3-explained.md` | eval sets and metrics from scratch, plus F18-F20 |
| `notes/day4-explained.md` | BM25, RRF and cross-encoders from scratch, plus F21-F24 |
| `notes/day5-explained.md` | grounded generation and judge validation, plus F25-F28 |
| `notes/day1-observations.md` | Day 1 root-cause analysis |

## Roadmap

| Day | Focus | Status |
|---|---|---|
| 1 | Walking skeleton: parse, chunk, index, ask | ✅ |
| 2 | CUAD corpus, structural chunking, metadata | ✅ |
| 3 | Golden eval set + metrics harness | ✅ |
| 4 | Hybrid retrieval (BM25 + RRF) + cross-encoder re-ranking | ✅ |
| 5 | Grounded generation, faithfulness, refusal accuracy | ✅ |
| 6 | Agentic compliance auditor + observability | next |
| 7 | FastAPI, Docker, CI eval gate, demo | |
