# ClauseCompass — Learning Notes

A concept-first course built on this repository. Teaches the theory behind every
part of the system from zero, using **your own measured results** as the worked
examples.

## How this differs from `notes/day*-explained.md`

| | Organised by | Assumes | Answers |
|---|---|---|---|
| `notes/day1-6-explained.md` (29,343 words) | **build order** — what I did Monday, what broke Tuesday | you already know the concepts | "what happened and why" |
| **this course** (`notes/learn/`) | **concept** — embeddings, then retrieval, then evaluation | classical ML only (Ridge, RMSE, precision/recall, overfitting) | "what is this, why does it work, how would I rebuild it" |

Read the day files as a *lab notebook*. Read this as the *textbook*.

## How to use this

Each module follows the same shape:

1. **What it is** — one definition
2. **Why it exists** — the problem that forced it
3. **Intuition** — before any maths
4. **Worked example** — usually with numbers from this repo
5. **The mechanism** — how it actually works internally
6. **Maths** — where it earns its place, derived not dumped
7. **Implementation** — minimal code that exposes the mechanism
8. **In this system** — the file and the measured result
9. **Failure modes** — what goes wrong, how to detect it, how to fix it
10. **Compared with** — the concept it gets confused with
11. **Interview angle** — the question you'll be asked and what a strong answer contains
12. **Test yourself** — questions with answers hidden in collapsible blocks

**Do not read the answers first.** Commit to an answer, then expand. A concept
you can recognise is not a concept you can derive.

---

## Syllabus

### Part A — Foundations

| # | Module | Core question |
|---|---|---|
| [00](notes/learn/00-vectors-and-embeddings.md) | Vectors, embeddings & similarity | How does a neural net turn "termination clause" into 384 numbers, and what does distance mean? |
| [01](notes/learn/01-why-rag-exists.md) | Why RAG exists | Why not fine-tune? Why not paste everything into the prompt? |
| [02](notes/learn/02-ingestion-and-parsing.md) | Ingestion & parsing | Why is getting text out of real documents the step that silently destroys your eval? |

### Part B — Chunking

| # | Module | Core question |
|---|---|---|
| [03](notes/learn/03-chunking-fundamentals.md) | Chunking: size, stride, overlap | Why chunk at all, and what does overlap actually cost? |
| [04](notes/learn/04-structural-chunking.md) | Structural chunking | How do you cut on meaning instead of character count? |
| [05](notes/learn/05-span-coverage.md) | Span coverage as a correctness check | How do you *prove* your chunker didn't lose the answers? |

### Part C — Retrieval

| # | Module | Core question |
|---|---|---|
| [06](notes/learn/06-dense-retrieval.md) | Dense retrieval & query asymmetry | Why do queries need a prefix and documents don't? |
| [07](notes/learn/07-vector-databases.md) | Vector DBs, ANN & payload filters | What does Qdrant do that NumPy doesn't? |
| [08](notes/learn/08-bm25.md) | BM25, from TF-IDF up | Why does a 1994 keyword algorithm still beat neural retrieval sometimes? |
| [09](notes/learn/09-rank-fusion.md) | Reciprocal Rank Fusion | How do you combine two rankers whose scores mean different things? |
| [10](notes/learn/10-cross-encoders.md) | Cross-encoders & re-ranking | Why is re-ranking cheap enough to be worth 300x the latency — sometimes? |

### Part D — Evaluation

| # | Module | Core question |
|---|---|---|
| [11](notes/learn/11-retrieval-metrics.md) | recall@k, hit@k, MRR, nDCG | Which metric answers which question, and how do they lie to you? |
| [12](notes/learn/12-building-a-golden-set.md) | Building a golden set | How do you manufacture ground truth without 40 hours of labelling? |
| [13](notes/learn/13-llm-as-judge.md) | LLM-as-judge & validating it | How do you trust a metric produced by the thing you're measuring? |

### Part E — Generation

| # | Module | Core question |
|---|---|---|
| [14](notes/learn/14-grounded-generation.md) | Grounded generation & citations | How do you make a model cite, and why does it fabricate IDs? |
| [15](notes/learn/15-verification-layers.md) | Deterministic vs semantic verification | Why does a verbatim-correct quote still support a false claim? |
| [16](notes/learn/16-refusal-and-calibration.md) | Refusal, calibration, asymmetric cost | Why is a system that never says "I don't know" a liability? |

### Part F — Systems

| # | Module | Core question |
|---|---|---|
| [17](notes/learn/17-agents.md) | Agents: tools, schemas, guardrails | How do you stop a loop that can spend money and invent evidence? |
| [18](notes/learn/18-observability-and-serving.md) | Observability & serving | How do you turn "it gave a wrong answer" into a five-minute diagnosis? |

---

## The numbers this course explains

Every one is reconstructable from a file in this repo. If a module states a
number, it names the file.

| Result | Where it comes from | Module |
|---|---|---|
| Random chunk pairs average **0.643** cosine similarity | measured over 44,850 pairs | 06 |
| Chunker silently discarded **16.6%** of annotated answers | CUAD offsets vs chunk ranges | 05 |
| Clause-boundary starts **13% → 62%** | regex over 18,280 chunks | 04 |
| Contract-type accuracy **12% → 69%** | label set changed, model unchanged | 09 |
| Golden set: **148 questions**, 12% unanswerable, verified absent | `evals/golden/golden.jsonl` | 12 |
| Doc-scoped recall@5 **0.575 → 0.762**, MRR **0.442 → 0.657** | `evals/baseline.json` | 10 |
| Corpus-wide: **nothing beat plain dense** (0.363) | the negative result | 09, 10 |
| Same change: **+0.17 / −0.17** depending on scope | entity-stripping ablation | 10 |
| Empty-gold questions would have inflated 0.363 → **~0.45** | 20 of 148 questions | 11 |
| Faithfulness **0.493**, judge agreement **95%** | `runs/*gen-final.json` | 13 |
| Validator stripped **33%** of claims (64 fires) | 191 raw → 127 kept | 15 |
| Correct refusal **1.00**, over-refusal **0.469** | the asymmetry | 16 |
| Agent precision **1.00**, recall **0.20** | `evals/agent_eval.json` | 17 |
| LLM is **93%** of agent wall clock | 413 s of 443 s | 18 |
