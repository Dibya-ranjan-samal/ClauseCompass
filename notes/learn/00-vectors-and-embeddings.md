# 00 — Vectors, Embeddings & Similarity

**Prerequisite check.** You need: what a vector is, dot product, and the idea
that a model has learned parameters. You do *not* need transformers yet — we
build up to them in module 06. If dot product feels shaky, read §3 slowly; the
whole course rests on it.

---

## 1. What it is

An **embedding** is a function that maps a piece of text to a fixed-length list
of real numbers, trained so that **texts with similar meaning map to nearby
points**.

In this project: `BAAI/bge-small-en-v1.5` maps any text to **384 floats**.

```
"termination for convenience"  ->  [0.031, -0.117, 0.204, ..., 0.008]   # 384 numbers
"either party may terminate"   ->  [0.028, -0.109, 0.198, ..., 0.011]   # nearby
"liability shall not exceed"   ->  [-0.142, 0.067, -0.031, ..., 0.093]  # far away
```

## 2. Why it exists

Computers compare strings by equality. `"terminate"` and `"termination"` are
different strings; `"end the contract"` shares no characters with either. But a
lawyer searching for termination provisions wants all three.

Classical IR solved part of this with stemming and synonym lists — brittle,
language-specific, hand-maintained. Embeddings replace that with a learned
function: instead of enumerating which words are similar, you learn a geometry
in which similar *meanings* are close.

**The reframe:** search stops being string matching and becomes
**nearest-neighbour lookup in a geometric space.**

## 3. Intuition: meaning as coordinates

Imagine placing every English phrase on a 2-D map, where you get to choose the
axes. Suppose axis 1 is "how much is this about ending things" and axis 2 is "how
much is this about money":

```
        money ^
              |
   "cap on    |  "termination fee"
    liability"|
        (0.1,0.9)      (0.8,0.7)
              |
              |
   ───────────+──────────────────> ending things
              |
              |  "notice of termination"
              |        (0.9,0.1)
```

"termination fee" sits high on both. "notice of termination" is far right, low
up. Now: **the distance between two points tells you how related the phrases
are** — and it works even though the phrases share no words.

An embedding is exactly this, with **384 axes instead of 2**. Nobody chose what
the axes mean; the model learned them. You never interpret an individual axis —
axis 57 does not mean "liability". Only *distances between points* carry
meaning.

This is the single most important idea in the module. Everything else is
mechanics.

## 4. How the numbers actually get produced

Three stages. Take the input `"Governing Law. This Agreement shall be governed by"`.

### Stage 1 — Tokenisation

Text is split into **tokens**: sub-word units from a fixed vocabulary (BGE uses
a ~30k WordPiece vocabulary).

```
"Governing Law. This Agreement shall be governed by"
->  ['govern', '##ing', 'law', '.', 'this', 'agreement', 'shall', 'be', 'governed', 'by']
->  [ 6789,     2453,   2375, 1012, 2023,  3820,       4618,   2022, 8226,      2011 ]
```

Why sub-words? A fixed vocabulary can't hold every English word, and legal text
is full of rare ones (`indemnitor`, `sublicensee`). Splitting into pieces means
an unseen word still becomes *something* the model can process, rather than a
single "unknown" token. `##ing` means "this piece attaches to the previous one".

**Practical consequence you already measured:** tokens are roughly 4 characters
of English on average. Your corpus is 29,535,164 characters ≈ **7.4 million
tokens**. Remember that number — module 01 needs it.

### Stage 2 — The transformer

Each token id becomes a learned vector (a lookup in an embedding matrix), then
passes through several transformer layers. Each layer lets every token look at
every other token and update its own representation based on context.

That last part is what makes it more than a dictionary. The word `"governed"` in
`"governed by the laws of Delaware"` gets a *different* vector than in
`"governed by a board of directors"`, because the surrounding tokens differ.
Static word vectors (word2vec, 2013) could not do this — one vector per word,
context ignored. Contextual embeddings are the advance.

We treat the transformer as a box for now. Module 06 opens it to the extent you
need for cross-encoders.

### Stage 3 — Pooling

The transformer outputs one vector *per token* — for 10 tokens, ten 384-dim
vectors. But you want **one** vector for the whole passage. So you pool:

- **CLS pooling** — take the vector of a special `[CLS]` token prepended to the
  input, which is trained to summarise the sequence. **BGE uses this.**
- **Mean pooling** — average all token vectors. Used by many sentence-transformer
  models.

The pooled 384-dim vector is the embedding.

## 5. Measuring similarity: cosine

Two vectors, **A** and **B**. Cosine similarity is:

```
                A · B          Σ aᵢbᵢ
cos(A,B) = ───────────── = ─────────────────
             |A| · |B|      √(Σaᵢ²)·√(Σbᵢ²)
```

- **A · B** — the dot product: multiply element-wise, sum.
- **|A|** — the length (L2 norm) of A: `√(a₁² + a₂² + ...)`.
- Dividing by the lengths removes magnitude, leaving only **direction**.

Range: −1 (opposite) to +1 (identical direction). For text embeddings you'll
mostly see 0 to 1.

### Work it by hand

Let A = [3, 4] and B = [4, 3].

```
A · B  = 3·4 + 4·3 = 12 + 12 = 24
|A|    = √(9 + 16)  = √25 = 5
|B|    = √(16 + 9)  = √25 = 5
cos    = 24 / (5 · 5) = 24/25 = 0.96
```

0.96 — nearly the same direction. Now C = [-4, 3]:

```
A · C  = 3·(-4) + 4·3 = -12 + 12 = 0
cos    = 0 / (5 · 5) = 0
```

Zero: **orthogonal**, geometrically unrelated.

### Why cosine and not Euclidean distance?

Euclidean distance `√(Σ(aᵢ-bᵢ)²)` is sensitive to **magnitude**. In text
embeddings, magnitude correlates with things you don't care about — roughly, how
much text there is and how confident the model is. Two passages about
indemnification, one 50 words and one 500, may point the same direction with
different lengths. Cosine says "same topic". Euclidean says "far apart".

**You care about direction, so you use cosine.**

### The normalisation trick

If you scale every vector to length 1 (`v / |v|`), then `|A| = |B| = 1`, and:

```
cos(A,B) = A · B / (1 · 1) = A · B
```

Cosine collapses to a plain dot product — one multiply-and-sum, no square roots.
That is why the code does this:

```python
model.encode(texts, normalize_embeddings=True)
```

Two payoffs: it's faster, and it guarantees the model's notion of similarity
matches Qdrant's `Distance.COSINE` exactly. Mismatched distance metrics between
your encoder and your index is a real and silent bug.

## 6. The geometry is not uniform — and you measured this

Here is where most tutorials stop and where your project got interesting.

Naively you'd assume: unrelated texts score ≈ 0, related texts score ≈ 1. You
measured that on your own corpus and it is **false**.

You sampled 300 chunks and computed all 44,850 pairwise similarities:

| Measurement | Cosine |
|---|---|
| Random chunk pair, mean | **0.6430** |
| Random pair, 90th percentile | 0.7060 |
| Random pair, **99th percentile** | **0.7787** |
| Best *relevant* chunk for a real query | **0.7670** |
| Best chunk for nonsense (`banana bicycle photosynthesis`) | 0.5779 |

Read line 3 against line 4. **The correct answer scored lower than the 99th
percentile of two randomly chosen chunks.**

Two random contract clauses are 0.64 similar because they *are* similar — both
are dense legal English, same register, same vocabulary, same sentence shapes.
The embedding faithfully reports that.

The general name for this is **anisotropy**: embeddings don't spread over the
whole space, they occupy a narrow cone. Distances within a cone compress.

**Consequences you must internalise:**

1. **An absolute cosine score is meaningless without a corpus baseline.** "0.77
   similarity" is not "very relevant" — on your corpus it means "is legalese".
2. **You cannot threshold on raw score** to decide relevance or to abstain.
3. **You can** threshold to detect an off-topic query — 0.578 vs 0.767 is a real
   gap.
4. Comparing scores *within one query's result list* is fine. Comparing across
   queries, or to a fixed constant, is not.

This is why module 10's cross-encoder exists.

## 7. Minimal implementation

Embedding from scratch is a transformer, so instead implement the part you must
be able to reason about — similarity — and verify against the library.

```python
import numpy as np

def cosine(a, b):
    """Cosine similarity between two 1-D arrays."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def cosine_normalised(a, b):
    """If both are already unit length, it's just the dot product."""
    return float(np.dot(a, b))

# verify the equivalence that justifies normalize_embeddings=True
a = np.array([3.0, 4.0]); b = np.array([4.0, 3.0])
an, bn = a / np.linalg.norm(a), b / np.linalg.norm(b)
assert abs(cosine(a, b) - cosine_normalised(an, bn)) < 1e-12
assert abs(cosine(a, b) - 0.96) < 1e-12
```

Retrieval over a whole corpus, as one matrix multiply:

```python
# E: (n_chunks, 384) unit-normalised.  q: (384,) unit-normalised.
scores = E @ q                    # (n_chunks,) — every cosine, one operation
top5   = np.argsort(-scores)[:5]  # highest first
```

Complexity: **O(n·d)** time, one pass over the matrix. For your corpus that's
18,280 × 384 ≈ 7 million multiply-adds — a few milliseconds. Module 07 explains
why you'd still use a vector database.

## 8. In this system

`src/clausecompass/retrieval/dense.py`:

```python
qv = self.model.encode(QUERY_PREFIX + query, normalize_embeddings=True)
```

| Choice | Value | Why |
|---|---|---|
| Model | `BAAI/bge-small-en-v1.5` | 384-dim, ~130 MB, runs on a laptop |
| Normalise | `True` | cosine becomes a dot product; matches Qdrant's metric |
| Device | `mps` | Apple Silicon GPU. Encoding 3,219 chunks took **32 s** |
| Pooling | CLS | BGE's trained pooling; don't override it |

## 9. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| Encoder/index metric mismatch | retrieval quietly mediocre | assert your cosine matches the index's top-1 score | normalise, and set `Distance.COSINE` |
| Different model for index vs query | results look random | store the model name in the collection payload | re-index with one model |
| Thresholding on absolute score | abstains wrongly, or never | measure the random-pair baseline (§6) | rank-based decisions, or a cross-encoder |
| Truncation at max sequence length | long clause's tail invisible | compare `len(tokens)` to the model's limit | chunk smaller, or window (module 10) |
| Assuming axes are interpretable | wasted analysis | — | only ever use distances |

**Truncation deserves emphasis.** BGE-small has a 512-token limit. Your
structural chunks reach 4,192 characters ≈ 1,050 tokens. Anything past 512 tokens
is *silently dropped* — no error. Module 10 shows the same bug biting the
re-ranker, where it cost real recall.

## 10. Compared with

| | Static word vectors (word2vec/GloVe) | Contextual embeddings (BGE) |
|---|---|---|
| Unit | one vector per word type | one vector per passage, context-dependent |
| `"governed"` | always the same vector | differs by sentence |
| Polysemy | cannot handle | handled |
| Cost | lookup table | forward pass through a transformer |
| Training | co-occurrence statistics | masked LM + contrastive fine-tuning |

| | Embedding (bi-encoder) | Cross-encoder |
|---|---|---|
| Input | one text at a time | a *pair*, together |
| Output | a vector | a relevance score |
| Precomputable | **yes** — embed the corpus once | no — per query-document pair |
| Cost at query time | one encode + a dot product | one forward pass per candidate |

That table is the seed of modules 06 and 10. Hold onto it.

## 11. Interview angle

**"What is an embedding?"** — Weak answers recite "a vector representation of
text". A strong answer names the *property*: a learned map into a space where
distance approximates semantic similarity, which turns search into
nearest-neighbour lookup, and where individual dimensions are not interpretable.

**"Why cosine rather than Euclidean?"** — Because magnitude in text embeddings
tracks length and confidence, not topic; cosine isolates direction. And if you
normalise, cosine is a dot product, so it's cheaper too.

**"Your top retrieval score is 0.77. Is that good?"** — This is the question
that separates people who have measured from people who have read. The answer:
*unanswerable without the corpus baseline*. On mine, random pairs average 0.643
and reach 0.779 at the 99th percentile, so 0.77 is at the noise ceiling — it
means "is legalese", not "is relevant". Absolute cosine cannot drive abstention;
it can detect an off-topic query, because nonsense scored 0.578.

## 12. Test yourself

**Q1.** A = [1, 0, 1], B = [0, 1, 1]. Compute cosine by hand.

<details><summary>Answer</summary>

`A·B = 1·0 + 0·1 + 1·1 = 1`. `|A| = √2`, `|B| = √2`. So `cos = 1/(√2·√2) = 1/2 = 0.5`.

</details>

**Q2.** Why does `normalize_embeddings=True` make cosine similarity equal to the
dot product? Show it.

<details><summary>Answer</summary>

Normalising sets `|A| = |B| = 1`. Cosine is `(A·B)/(|A||B|) = (A·B)/(1·1) = A·B`.
The denominator vanishes, so no norms or square roots are needed at query time.

</details>

**Q3.** Two passages, both about indemnification: one 40 words, one 400. Would
you expect cosine or Euclidean distance to judge them more similar, and why?

<details><summary>Answer</summary>

Cosine. Magnitude in text embeddings correlates with passage length, so the
400-word vector is likely longer. Euclidean distance is inflated by that length
difference even when direction is nearly identical; cosine divides it out and
reports the shared direction — i.e. the shared topic.

</details>

**Q4.** Your corpus has random-pair mean cosine 0.643. I hand you a new corpus of
mixed news, code and recipes and you measure random-pair mean 0.21. What does
that difference tell you, and how should it change how you use scores?

<details><summary>Answer</summary>

The second corpus is **heterogeneous** — its documents genuinely differ in
register and topic, so the embedding space is less compressed and distances are
more discriminative. On that corpus an absolute score threshold is far more
defensible, because the gap between "related" and "random" is wide. On the legal
corpus the gap is nearly zero, so ranking is meaningful but absolute values are
not. The lesson generalises: **the usable score range is a property of the
corpus, not of the model**, and must be measured per corpus.

</details>

**Q5.** Harder. A colleague proposes: "flag any retrieved chunk scoring below
0.70 as irrelevant and refuse to answer." Using the numbers in §6, give two
distinct reasons this fails.

<details><summary>Answer</summary>

1. **It refuses correct answers.** Genuinely relevant chunks scored 0.767 and
   others below — the *relevant* distribution overlaps 0.70 heavily, so the
   threshold discards true positives.
2. **It accepts garbage.** Random pairs average 0.643 and reach 0.779 at p99, so
   a large mass of *irrelevant* chunks clears 0.70 too. The threshold sits inside
   the noise distribution, meaning it has poor precision *and* poor recall
   simultaneously.

The deeper point: a single global threshold assumes the score is calibrated
against relevance. It is calibrated against *register*. The fix is either
rank-based logic or a model that scores the query-document pair jointly
(module 10).

</details>

**Q6.** Your embedding model has a 512-token limit. You feed it a 1,500-token
clause. What happens, what does the API return, and how would you *detect* this
in production?

<details><summary>Answer</summary>

The input is silently truncated to the first 512 tokens; tokens beyond that never
influence the vector. No exception, no warning — the call returns a normal
384-dim embedding, so nothing downstream looks broken. The embedding simply
represents the first third of the clause.

Detection: tokenise before encoding and compare length to the model's
`max_seq_length`, logging or asserting when you exceed it. This is the same class
of bug as Ollama's silent `num_ctx` truncation, where 60% of a prompt was
discarded with no error (module 14) — **silent capacity limits are the most
dangerous kind, because every log line looks healthy.**

</details>

---

**Next:** [01 — Why RAG exists](01-why-rag-exists.md)
