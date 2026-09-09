# 01 — Why RAG Exists

---

## 1. What it is

**Retrieval-Augmented Generation**: instead of expecting a language model to
*know* your documents, you retrieve the few passages relevant to the question and
put them in the prompt, then ask the model to answer *only* from those.

```
question ──> retriever ──> top-k passages ──┐
                                            ├──> LLM ──> grounded answer
question ───────────────────────────────────┘
```

## 2. Why it exists — three problems, with your numbers

You have 530 contracts, **29,535,164 characters ≈ 7.4 million tokens**.

### Problem 1: it does not fit

| Model | Advertised context | Your corpus |
|---|---|---|
| `qwen2.5:3b` | 32,768 tokens | 7,400,000 tokens |

You are over by **226×**. And it's worse in practice: Ollama defaults `num_ctx`
to **2,048** regardless of what the model supports, so the real budget was
2k until you set it explicitly. You measured that: a 20,316-character prompt had
only 2,050 of ~5,079 tokens evaluated — **60% silently discarded** (module 14).

### Problem 2: cost and latency scale with what you send

Even where a long context fits, you pay for it *per question*. Transformer
attention is **O(n²)** in sequence length: doubling the context roughly
quadruples the attention work. Sending 7.4M tokens to answer "what's the
governing law?" is paying to re-read the entire library for every question.

### Problem 3: attention dilutes

Long-context models degrade at finding one fact buried in a huge context —
performance sags for material in the middle. Known as **"lost in the middle"**.
So even a model that *accepts* 7.4M tokens would answer worse than one given the
right 5,000.

### The insight

For any single question, almost all of those 7.4M tokens are irrelevant. Find the
~5 relevant passages and an impossible problem becomes a 5,000-token one.

## 3. Why not fine-tune instead?

This is the comparison you will be asked. The honest answer is that they solve
**different problems**.

| | Fine-tuning | RAG |
|---|---|---|
| Teaches | behaviour, format, style, domain register | nothing — supplies facts at query time |
| Facts live in | weights, entangled and unattributable | an index, addressable and inspectable |
| Add a contract | retrain or adapt | insert rows; done in seconds |
| Remove a contract | very hard (weights don't forget cleanly) | delete rows |
| Can it cite? | no — there is no source to point at | **yes, and this is the point** |
| Wrong output | opaque; is it memorised, blended, invented? | traceable to a retrieved chunk |
| Cost | GPU hours per update | embedding cost per document, once |

For "find the clause and prove it", fine-tuning is the wrong tool: it cannot
produce a citation, cannot be updated per document, and will confidently blend
half-remembered clauses from different contracts. Your Day 1 failure F2 is
exactly that blending — the model welded half a sentence from Healthways to half
from Goodman Networks. RAG doesn't magically prevent that, but it gives you
something to *check the answer against*, which is what made the bug findable.

**When fine-tuning is right:** you need a consistent output format the base model
resists, a domain register it handles badly, or you want to distil a large
model's behaviour into a small one. Those are behaviour problems, not knowledge
problems. In production the two are complements, not rivals: fine-tune for
behaviour, retrieve for facts.

## 4. The tradeoff nobody mentions

RAG does not remove the hard problem. **It relocates it from generation to
retrieval.**

> If the retriever hands the model the wrong passages, a perfect LLM still gives
> a wrong answer. Garbage in, *confident* garbage out.

You measured the ceiling this creates. Corpus-wide, your `recall@50` is
**0.5849** (`evals/baseline.json`). Read that as a statement about the whole
system:

- 41% of the gold clauses are **never retrieved at all**, even at depth 50.
- For those questions, no prompt, no bigger model and no re-ranker can help. The
  evidence was never in the context.

**recall@k is therefore an upper bound on end-to-end answer quality.** That single
realisation is why module 11 exists and why the eval harness came before the
optimisation.

## 5. The full pipeline, and where each module lands

```
INDEXING (offline, once)
  documents ─> parse ─> chunk ─> embed ─> store
     [02]      [02]    [03-05]   [06]    [07]

QUERYING (online, per question)
  question ─> embed ─> search ─> fuse ─> re-rank ─> prompt ─> generate ─> validate
              [06]     [07,08]   [09]     [10]      [14]       [14]        [15]

MEASURING (the part that makes the rest meaningful)
  golden set ─> retrieval metrics ─> generation metrics ─> judge validation
     [12]            [11]                 [15,16]              [13]
```

Two things worth noticing about that diagram:

1. **The embedding model appears in both halves and must be identical.** Indexing
   with BGE and querying with a different model returns noise, because the two
   vector spaces are unrelated. There is no error message for this.
2. **Measurement is a third pipeline, not a step.** Days 1–2 of this project
   produced claims; Day 3 produced the instrument; two of the Day-3 predictions
   turned out wrong. That ordering is the lesson.

## 6. Failure modes of RAG as an architecture

Not bugs in one component — ways the *approach* fails.

| Failure | What it looks like | Detect with | Fix |
|---|---|---|---|
| **Retrieval ceiling** | answers wrong however you prompt | `recall@k` — is the gold chunk even there? | better chunking, hybrid retrieval, query rewriting |
| **Ranking failure** | gold chunk retrieved but not in top-5 | large gap between recall@50 and recall@5 | re-ranking (module 10) |
| **Chunk boundary loss** | answer split across two chunks, neither sufficient | span coverage (module 05) | structural chunking, overlap |
| **Context dilution** | more retrieved context, worse answers | sweep top-k and watch quality *fall* | fewer, better passages |
| **Ungrounded generation** | fluent answer, no support in context | citation + quote validation (module 15) | validate in code, don't ask nicely |
| **Ill-posed question** | model refuses or blends sources | inspect whether the question identifies one document | scope the query (module 09's finding) |
| **Aggregation questions** | "how many contracts use NY law?" answered badly | include such questions in the eval on purpose | agents / structured queries (module 17) |

The last two are worth dwelling on because you hit both. Your golden set
deliberately contains 2 aggregation questions that single-pass RAG *cannot*
answer — no single chunk contains a corpus-wide count. Measuring a known
weakness on purpose is stronger than quietly excluding it.

## 7. Interview angle

**"Why RAG and not fine-tuning?"** — Name the distinction (behaviour vs facts),
then the three decisive properties: citations, per-document updates, and
auditability. Add the nuance that they're complements.

**"What's the main weakness of RAG?"** — That it moves the difficulty into
retrieval and most people never measure it. Then give the number: my corpus-wide
recall@50 is 0.585, so 41% of answers are unreachable regardless of the model —
which told me to fix retrieval coverage, not prompting.

**"Long-context models are getting cheap. Is RAG obsolete?"** — A genuinely good
question to have a view on. Cost scales with context and attention is O(n²), so
re-reading 7.4M tokens per query is wasteful even when possible; models degrade
on facts buried mid-context; and long context still cannot *cite* — you lose the
audit trail that makes the system usable in a legal setting. The honest
concession: for a small corpus that fits comfortably, long context is simpler and
you should prefer it.

## 8. Test yourself

**Q1.** Your corpus is 7.4M tokens and your model accepts 32k. Give three
distinct reasons not to simply send all 7.4M even if a model could accept it.

<details><summary>Answer</summary>

1. **Cost** — you pay per token, per query, to re-send material that is 99.9%
   irrelevant to the question.
2. **Latency** — attention is O(n²) in sequence length, so long contexts are
   disproportionately slow.
3. **Accuracy** — retrieval quality of long-context models degrades for facts in
   the middle of the context ("lost in the middle"), so more context can produce
   a *worse* answer.

A fourth, which is the one that matters for this project: **no citations.** You
lose the ability to point at the source clause.

</details>

**Q2.** A colleague says "we'll fine-tune the model on our 530 contracts so it
knows them." Give the two strongest objections.

<details><summary>Answer</summary>

1. **It cannot cite.** Facts stored in weights have no address; you cannot show a
   lawyer the clause the answer came from, and you cannot verify it
   programmatically. For contract review that is disqualifying.
2. **It cannot be updated per document.** Adding or deleting one contract means
   retraining, and "unlearning" a removed contract from weights is not reliably
   possible — which is both an accuracy and a compliance problem.

Bonus objection: fine-tuning encourages *blending* — the model interpolates
between similar clauses it has memorised, producing plausible sentences that
exist in no contract. That is precisely failure F2 in this project.

</details>

**Q3.** Your `recall@5` is 0.36 and your `recall@50` is 0.58. Your teammate
proposes upgrading the generator to a larger model to improve answer quality.
Why is that the wrong next move, and what is the right one?

<details><summary>Answer</summary>

Because 42% of gold chunks are absent from the top 50, so for those questions the
evidence never reaches the generator — a larger model has nothing extra to work
with. The generator is not the binding constraint.

The gap between recall@5 (0.36) and recall@50 (0.58) says ranking is *also* weak,
so re-ranking would help the retrieved-but-mis-ranked cases. But the bigger prize
is the 42% never retrieved, which needs a retrieval-side fix: better chunking,
hybrid lexical+dense retrieval, or query rewriting.

The general principle: **fix the earliest stage that bounds the outcome.**
Improving a downstream stage cannot recover information the pipeline already
discarded.

</details>

**Q4.** Explain why the embedding model used at indexing time and at query time
must be the same, and what the failure looks like if it isn't.

<details><summary>Answer</summary>

An embedding model defines a coordinate system. Two different models produce two
unrelated coordinate systems, so the geometry that makes nearest-neighbour search
meaningful does not hold across them — dimension 57 of model A has nothing to do
with dimension 57 of model B.

The failure is silent and specific: retrieval still *works* mechanically and
returns k results with plausible-looking scores, but the results are effectively
random. Nothing raises an exception, dimensions may even match (both 384), and
you only notice via a quality metric. This is why the collection should record
which model wrote it.

</details>

**Q5.** Harder. Give an example of a question that is well-formed in English but
that single-pass RAG structurally cannot answer, and explain what architecture
does answer it.

<details><summary>Answer</summary>

*"How many of these contracts are governed by New York law?"*

No single chunk contains the answer, because the answer is a **property of the
corpus**, not of any passage. Retrieval returns k governing-law clauses; the
count requires visiting *every* document. Increasing k does not fix it — the
answer is not a passage.

What answers it: either a structured query against extracted metadata (you have
`governing_law` on every chunk, so this is a `GROUP BY`), or an agent that
iterates over documents and aggregates — which is why module 17 exists. Your
golden set includes two such questions deliberately, so the weakness is measured
rather than hidden.

</details>

---

**Previous:** [00 — Vectors & embeddings](00-vectors-and-embeddings.md) ·
**Next:** [02 — Ingestion & parsing](02-ingestion-and-parsing.md)
