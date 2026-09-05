# Evaluation methodology

Every number in the README is reconstructable from a file in this repo. This
document says how each was produced, and where each one is weak.

---

## The golden set

`evals/golden/golden.jsonl` — **148 questions over 127 distinct contracts.**

| Type | n | Source |
|---|---|---|
| single_hop | 118 | templated from CUAD annotations, sampled and reviewed |
| unanswerable | 18 | hand-authored, absence **verified** |
| near_miss | 6 | hand-authored, both clauses confirmed present |
| multi_hop | 4 | hand-authored, gold spans two categories |
| aggregation | 2 | hand-authored, expected to fail single-pass RAG |

### How a gold chunk is decided

CUAD gives each answer as a character offset into the contract. A chunk is gold
only if it **both** overlaps the annotated span **and** contains the answer text:

```python
overlaps = c["char_start"] < end and c["char_end"] > start
contains = norm(answer)[:120] in norm(c["text"])
```

Overlap alone proves the offsets line up, not that the text survived chunking.
Only **3 of 4,256** generated questions failed this check.

### Questions name their contract, and that is an intervention

Generic CUAD phrasing is byte-identical across all 530 contracts, so corpus-wide
retrieval cannot know which contract is meant — measured at **hit@5 = 0/30**.
Questions are therefore entity-qualified ("Under the *Adams Golf Endorsement*
agreement, …"), which reaches hit@5 4/30. Real users would not always name the
contract, so the corpus-wide number measures retrieval **given a named
contract**: easier than a cold query, harder than doc-scoped.

### Unanswerable questions are verified, not assumed

Each carries the terms that must be absent, and is only assigned to a contract
where none appear anywhere in the full text. Re-checked independently: **0 leaks.**

### Rejection rates — both of them

- **Generation stage: 6.3%** (282 fragment answers, 3 with no gold chunk)
- **Review stage: 1.7%** (2 of 120 sampled)

The review number is low because filtering already happened upstream. Quoting
only one of these would mislead.

### Provenance

CUAD's clause spans are **attorney-reviewed**. The sampling review, the 30
hand-authored questions and every keep/delete decision are **AI-generated** —
`evals/golden/review_decisions.json` records the method in `_provenance`. This is
not an attorney-verified eval set.

---

## Retrieval metrics

`recall@k`, `hit@k`, `MRR`, `nDCG@k` — `src/clausecompass/eval/metrics.py`, 8 unit tests.

**Every metric returns 1.0 for empty gold.** The 18 unanswerable and 2
aggregation questions have no gold chunks, so including them would hand out 20
free 1.0s and inflate `corpus_wide` recall@5 from the true 0.3632 to ~0.45. The
runner excludes them and reports `n_excluded_no_gold`. A unit test pins the hazard.

### Two scopes, because they have different bottlenecks

| | recall@5 | recall@50 | Diagnosis |
|---|---|---|---|
| `corpus_wide` | 0.363 | 0.585 | 41% of gold never retrieved → **retrieval** problem |
| `doc_scoped` | 0.575 | 0.979 | retrieved but mis-ranked → **ranking** problem |

Day 4 tested both predictions. The ranking one was right (re-ranking took
doc-scoped to 0.762). The retrieval one was right in direction but small: BM25
lifted the corpus-wide ceiling only 0.585 → 0.604.

---

## Generation metrics

### Deterministic, and free

- **Citation existence** — every cited label must be one actually retrieved.
- **Quote verification** — the quoted string must appear in the cited chunk
  (whitespace-insensitive exact match, else ≥95% longest-contiguous ratio).

Fired **64 times** over 148 questions (60 bad citations, 4 bad quotes), stripping
**64 of 191 claims (33%)**. A validator that never fires is a validator that
isn't wired up.

### Faithfulness, and why the judge is quoted with its agreement

An LLM judge scores whether each surviving claim is supported by its cited
excerpt. **An unvalidated judge is a vibe with a decimal point**, so it was
validated against 40 hand labels, stratified 20 supported / 20 unsupported — an
all-supported sample cannot detect a lenient judge.

**Judge agreement: 38/40 = 95%**, with exactly one error in each direction (one
too lenient on a vacuous `"..."` claim, one too strict about wording). No
systematic bias, so **faithfulness 0.493 is real, not a judge artifact.**

Labels: `evals/judge_labels.json`, AI-generated, `_provenance` recorded.

### The two checks are orthogonal — this is the main finding

Every claim judged unsupported had **already passed** verbatim quote validation.

- Quote validation catches **fabricated citations and invented quotes** (64 fires)
- Semantic judging catches **valid quotes with unsupported conclusions** (61 claims)

Deterministic string matching cannot detect a correct quote followed by a wrong
inference. Reporting only the deterministic number would look like a clean system.

---

## Agent metrics

Ground truth comes from **CUAD's `Cap On Liability` annotations**, not hand
labels: answerable means the clause exists, unanswerable means it does not, which
is exactly the policy's test. Stronger provenance (attorney-reviewed), larger n,
and no labelling time.

**Its limit:** CUAD records whether a clause *exists*, not whether an existing cap
is *adequate*. So this scores **clause detection**, not compliance judgement. A
policy like "notification within 72 hours" would still need hand labels.

Contracts are chosen **balanced 6 present / 6 absent** — an all-present sample can
only measure recall.

---

## Known limitations

1. **CUAD-shaped.** 118 of 148 questions come from CUAD's 41 categories (20
   templated, 21 not). It reflects what attorneys chose to annotate.
2. **AI-reviewed, not attorney-verified**, for everything except CUAD's own spans.
3. **Small hard-case samples.** near_miss n=6, multi_hop n=4, aggregation n=2,
   agent n=10. Directional only.
4. **Aggregation questions score ~0 by design** and are excluded from retrieval
   aggregates. They motivated the agent layer.
5. **The 7B generator comparison cannot measure refusal** — its 60-question
   sample contains no unanswerable questions, which sit at the end of the file.
6. **Traps are mild.** 0 of 6 near-miss traps beat the gold chunk corpus-wide.
7. **One policy** for the agent, n=10 scored.
8. **Local models throughout.** `cost_usd` is 0.0 everywhere; the field exists so
   a hosted provider drops in, but it measures nothing today.

---

## Reproducing

```bash
make setup && make ingest && make index
make eval        # retrieval, ~2 min, no LLM
make eval-gen    # faithfulness + refusal, ~2 h local
make audit       # agent + injection test
```

Run records land in `runs/` with a git SHA. `evals/baseline.json` holds the
best-per-scope configuration.
