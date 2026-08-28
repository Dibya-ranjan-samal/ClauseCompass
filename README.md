# ClauseCompass

Question answering over real commercial contracts, with citations you can verify
against the source text.

**Status: Day 2 of a 7-day build.** The pipeline runs end to end and the data is
in place. Retrieval quality is **not yet measured** — the eval harness is Day 3,
and no quality claim in this README goes beyond what has actually been measured.

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
uv run python eval_metadata.py        # accuracy report
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
| `notes/day1-explained.md` | how the Day 1 stack works, from scratch |
| `notes/day2-explained.md` | how the Day 2 stack works, plus every failure and fix |
| `notes/day1-observations.md` | Day 1 root-cause analysis |

## Roadmap

| Day | Focus | Status |
|---|---|---|
| 1 | Walking skeleton: parse, chunk, index, ask | ✅ |
| 2 | CUAD corpus, structural chunking, metadata | ✅ |
| 3 | Golden eval set + metrics harness | next |
| 4 | Hybrid retrieval (BM25 + RRF) + cross-encoder re-ranking | |
| 5 | Grounded generation, faithfulness, refusal accuracy | |
| 6 | Agentic compliance auditor + observability | |
| 7 | FastAPI, Docker, CI eval gate, demo | |
