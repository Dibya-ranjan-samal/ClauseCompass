# 13 — LLM-as-Judge, and Validating It

> **An unvalidated LLM judge is a vibe with a decimal point.**

Almost nobody does the validation step. That is exactly why doing it is worth
talking about.

---

## 1. What it is

Using a language model to score outputs that have no programmatic ground truth.
Here: **faithfulness** — is this claim actually supported by the excerpt it
cites?

```
claim + cited excerpt ──> judge model ──> {"supported": true|false, "reason": "..."}
```

## 2. Why it exists

Module 15 shows you can verify a lot deterministically: does the cited chunk id
exist, does the quoted string appear in it. Both are free and exact.

But consider:

```
Excerpt: "Either Party may terminate this Agreement at any time prior to
          delivery of the Product."

Claim:   "Neither party can terminate for convenience under this agreement."
```

The claim **contradicts** the excerpt. No string match detects that — the claim
quotes nothing falsely, invents no citation, and shares vocabulary with its
source. Detecting it requires *understanding*, and the only scalable source of
understanding is another model.

**The uncomfortable position this puts you in:** you are measuring a model's
output with a model. That circularity is real, and validation is how you bound
it.

## 3. The judge prompt

```python
JUDGE_SYSTEM = """You are evaluating whether a claim is supported by contract text.

Answer with ONLY valid JSON: {"supported": true, "reason": "one short sentence"}

supported true  - the excerpt directly states or unambiguously entails the claim
supported false - the excerpt does not contain this information, contradicts it,
                  or the claim adds specifics not present in the excerpt

Be strict. Plausibility is not support. If the claim adds a number, a duration,
or a party name that is not in the excerpt, it is not supported."""
```

Four deliberate choices:

| Choice | Why |
|---|---|
| **Binary, not 1–5** | a 1–5 scale invites the model to cluster on 3 and 4; you cannot define the boundaries reliably, and inter-rater agreement on ordinals is worse |
| **"Be strict. Plausibility is not support."** | the default failure mode is leniency — models reward plausible-sounding claims |
| **Names the adding-specifics failure** | the most common real hallucination is a correct quote with an invented number attached |
| **Requires a `reason`** | forces the judge to state its basis, which makes *your* validation possible — you can read why it decided |

## 4. Judging one claim at a time — a design decision

```python
for c in a.claims:
    excerpt = "\n\n".join(t.text for t in hits if t.chunk_id in c["chunk_ids"])
    v = judge_claim(c["statement"], excerpt, judge_llm)
```

Not "score this whole answer 1–10". **One claim, its own cited excerpt, a binary
verdict.**

Why decomposition matters:

- An answer with 3 good claims and 1 fabricated one is not "75% good" in any
  useful sense — you want to know *which* claim failed.
- The judge sees only the excerpt that claim cites, so it cannot be rescued by
  context from a *different* chunk. This is the crucial isolation: it makes the
  judge answer "does *this* evidence support *this* claim", not "is this answer
  broadly reasonable".
- Faithfulness becomes a well-defined fraction: `supported / judged`.

## 5. The parse-failure trap

```python
except Exception:
    # A judge that cannot be parsed must not silently count as "unsupported":
    # that would inflate the hallucination rate with judge failures.
    return {"supported": None, "reason": "judge_parse_failed"}
```

**This is a real and easy mistake.** The obvious `except: return False` conflates
two entirely different events:

| Event | Meaning | Should count as |
|---|---|---|
| Judge says `false` | the claim is unsupported | a hallucination |
| Judge output unparseable | **you don't know** | excluded from the denominator |

Returning `False` on a parse failure means every malformed judge reply is
recorded as a hallucination by the *generator*. Your hallucination rate then
partly measures your judge's JSON compliance — a completely different quantity,
and one that moves when you change judge models.

The three-valued return (`True`/`False`/`None`) with `None` excluded from the
denominator keeps the metric measuring what it claims to.

> **Generalise it:** "unknown" and "no" are different values. Collapsing them is
> one of the most common ways a metric quietly changes meaning.

## 6. Validating the judge — the step that makes the number quotable

### Sample selection: stratify by verdict

```python
yes = [p for p in pairs if p["judge_supported"]]
no  = [p for p in pairs if not p["judge_supported"]]
take_no = min(len(no), N // 2)
picked = random.sample(no, take_no) + random.sample(yes, min(len(yes), N - take_no))
```

**Why stratify?** Because you need to detect **both error directions**:

- If you sample only claims the judge called *supported*, you can detect
  **leniency** (judge said yes, human says no) but are blind to **strictness**.
- The reverse for an all-unsupported sample.

A 20/20 split measures both. From a pool of 125 judged pairs (64 supported, 61
unsupported), you take 20 of each.

### Then label them yourself, and score

```
judge agreement: 38/40 = 95%
  judge too lenient (said supported, human said not): 1
  judge too strict  (said not, human said supported): 1
  verdict: trust it
```

**One error in each direction.** That balance matters more than the headline
percentage: it means there is **no systematic bias**. A judge that is 95% accurate
but wrong in only one direction is *biased* and its errors accumulate; a judge
wrong symmetrically has errors that partly cancel in aggregate.

### The two disagreements, in detail

Worth studying because they show what judge errors actually look like.

**#8 — judge too lenient.** The claim text was literally `"..."` — three dots. The
3B generator emitted a vacuous claim. The judge marked it `supported` because the
excerpt (about IP ownership) was coherent. But **a vacuous claim cannot be
supported**: there is no proposition to support. Human verdict: `false`.

**#39 — judge too strict.** The claim: *"The contract prohibits the assignment of
the Intellectual Property Agreement without prior written consent of the other
party."* The excerpt was §8.01, *"No Assignment or Transfer Without Consent…
neither this Agreement nor any of the rights… shall be assigned… without the
prior written consent of the other Party."*

The judge said `false`, reasoning *"the excerpt does not specifically mention
Intellectual Property Agreement"*. But the excerpt says "this Agreement" and the
document **is** the IP Agreement. Over-pedantic. Human verdict: `true`.

Notice the shapes: leniency shows up as **rewarding coherence over content**;
strictness shows up as **demanding literal token overlap where reference suffices.**

### The threshold table

| Agreement | What you may claim |
|---|---|
| ≥ 90% | trust it; report faithfulness confidently |
| 85–90% | usable; state the agreement rate next to **every** faithfulness number |
| 75–85% | tighten the judge prompt, add examples, re-measure |
| < 75% | **do not report faithfulness as a headline.** Report only deterministic validation |

At 95% you're in the top band — so **faithfulness 0.493 is a real number, not a
judge artifact.** That inference is the entire payoff of the validation step.

## 7. The result

| Metric | Value |
|---|---|
| Claims judged | 127 (64 supported, 61 unsupported) |
| **Faithfulness** | **0.493** |
| **Hallucination rate** | **0.507** |
| Judge agreement | **95%** (38/40) |
| Judge model | `qwen2.5:7b` |
| Generator model | `qwen2.5:3b` |

**Half the claims that survived deterministic validation are not supported by the
text they cite.** Module 15 explains how that's possible when every one passed a
verbatim quote check.

### One choice that reduced a known bias

Judge (7B) ≠ generator (3B). Models exhibit **self-preference bias** — they rate
their own outputs more favourably. Using the same model for both would inflate
faithfulness by an unknown amount. Using a different (and stronger) model doesn't
eliminate the concern but does remove the most direct form of it.

## 8. Failure modes

| Failure | What happens | Detect | Fix |
|---|---|---|---|
| **No validation at all** | you're quoting a number of unknown meaning | — | hand-label 40, stratified |
| Parse failure → `False` | hallucination rate measures judge JSON compliance | count parse failures separately | three-valued return, exclude `None` |
| Unstratified validation sample | blind to one error direction | check the verdict balance of your sample | stratify 50/50 |
| **Self-preference bias** | inflated faithfulness | judge with a different model | different, ideally stronger, judge |
| Position bias | in pairwise judging, first option favoured | swap order, re-run | randomise or average both orders |
| Verbosity bias | longer answers rated better | correlate score with length | penalise length, or judge per claim |
| Judging whole answers | can't localise the failure | — | decompose into claims |
| Ordinal scales (1–5) | clustering on middle values | plot the distribution | binary verdicts |
| Judge sees more than the citation | rescued by unrelated context | pass only the cited excerpt | isolate per claim |
| Stale validation | judge or prompt changed | version the judge; re-validate on change | re-measure agreement |

The last one bites in production: **agreement is a property of a specific
(judge model, judge prompt, task) triple.** Change any one and the number is no
longer yours.

## 9. Compared with

| | Deterministic checks | LLM judge | Human labels |
|---|---|---|---|
| Cost | free | ~1 LLM call per claim | expensive |
| Speed | microseconds | seconds | minutes |
| Catches fabricated citations | ✅ | ✅ | ✅ |
| Catches unsupported inference | ❌ | ✅ | ✅ |
| Reproducible | ✅ exactly | ⚠️ at temp 0, per version | ⚠️ inter-rater variance |
| Needs validation | no | **yes** | it *is* the validation |
| Scales | ✅ | ✅ | ❌ |

**The right architecture uses all three:** deterministic checks on every request,
an LLM judge on the eval set, and human labels on a sample of the judge. Each
validates the layer above it.

## 10. Interview angle

**"How do you measure hallucination?"** — Two layers. Deterministically: every
cited chunk id must exist in what was retrieved, and every quoted string must
match its source — that stripped 33% of claims. Then an LLM judge for semantic
support, scoring each claim against only the excerpt it cites. Faithfulness came
out 0.493 — but that number is only quotable because I validated the judge.

**"How do you know your judge is any good?"** — I hand-labelled 40 claim/excerpt
pairs, stratified 20 supported / 20 unsupported so both error directions were
measurable, and got 95% agreement with one error in each direction. The balance
matters as much as the rate: symmetric errors partly cancel in aggregate, a
one-sided judge biases the metric. Below 75% I wouldn't have reported faithfulness
at all.

**"Why stratify the validation sample?"** — An all-supported sample can only
detect leniency; an all-unsupported sample can only detect strictness. A 50/50
split measures both. My two errors were exactly one of each — a vacuous `"..."`
claim the judge affirmed, and a correctly-supported claim it rejected for not
repeating the document's name literally.

**"Isn't using an LLM to grade an LLM circular?"** — Partly, and you bound it
rather than eliminate it. Three mitigations: use a *different, stronger* model
than the generator to avoid self-preference bias; decompose into per-claim binary
verdicts against only the cited excerpt so the judge can't be rescued by
unrelated context; and validate against human labels so you can state the
agreement rate alongside every number. "Faithfulness 0.49, judge validated at 95%
agreement on 40 stratified hand labels" is a defensible sentence. "Faithfulness
0.49" alone is not.

## 11. Test yourself

**Q1.** Your judge returns unparseable output for 5 of 130 claims. Why is
`return {"supported": False}` wrong, and what should you do?

<details><summary>Answer</summary>

Because it conflates *"the claim is unsupported"* with *"I could not read the
judge's answer."* Those are different events with different causes: the first is
a fact about the generator, the second is a fact about the judge's JSON
compliance.

Counting them as `False` means your hallucination rate partly measures judge
formatting reliability — so it would move if you swapped judge models even with
an identical generator. That makes the metric non-comparable across runs.

Correct handling: a three-valued return where parse failures are `None`, excluded
from the denominator, and **counted and reported separately** so you know how
often it happened. If parse failures are frequent, that's a judge problem to fix,
not a number to bury.

</details>

**Q2.** You validate your judge on 40 pairs, all of which the judge marked
`supported`. Agreement is 92%. What can and cannot you conclude?

<details><summary>Answer</summary>

**Can conclude:** when the judge says *supported*, it's right ~92% of the time.
That bounds its **leniency** — roughly 8% false positives.

**Cannot conclude anything about strictness.** You have zero examples of claims
the judge rejected, so you have no idea how often it wrongly rejects supported
claims. If it rejects half of all valid claims, this sample cannot see it.

Consequence for the metric: your faithfulness number could be substantially
*understated* and this validation would look fine. Fix: stratify by verdict, 20
of each.

</details>

**Q3.** Your judge and your generator are the same model. Name the bias and
estimate its direction.

<details><summary>Answer</summary>

**Self-preference bias** — models systematically rate their own generations more
favourably than other models' generations of equal quality. Contributing
mechanisms: the output is in the model's own idiom and token distribution, so it
is more "expected" and easier to process, and it shares the same misconceptions —
if the generator misreads a clause, the judge is likely to misread it identically
and therefore judge the claim consistent.

Direction: **inflates faithfulness / deflates hallucination rate.**

The second mechanism is the nastier one: correlated errors mean the judge is
blind to exactly the mistakes the generator is prone to. Mitigation is a
different and ideally stronger judge — this project used 7B to judge 3B — plus
human validation to bound whatever remains.

</details>

**Q4.** Why binary verdicts rather than a 1–5 faithfulness score?

<details><summary>Answer</summary>

1. **The boundaries are undefinable.** You cannot write a rubric distinguishing 3
   from 4 that two annotators apply consistently, so inter-rater agreement on
   ordinals is materially worse than on binary.
2. **Models cluster on middle values**, avoiding the extremes, which compresses
   the usable range and makes the metric insensitive.
3. **Validation becomes hard.** With binary you count agreements and get a clean
   percentage plus a two-cell confusion breakdown. With ordinals you need
   correlation or weighted kappa, and "agreement" needs defining (exact? within
   1?).
4. **The decision it feeds is binary** — keep the claim or strip it. A metric
   should match the granularity of the action it informs.

</details>

**Q5.** Harder. Your judge agreement is 78%, and the errors are all one-sided:
the judge is too lenient. Faithfulness reads 0.85. What may you report, and what
would you do next?

<details><summary>Answer</summary>

**What you may report:** not 0.85 as a faithfulness figure. At 78% agreement you
are in the "tighten the prompt and re-measure" band, and one-sided errors make it
worse — leniency errors don't cancel, they accumulate in one direction.

You may report the **deterministic** validation numbers (citation existence and
quote match rates), because those need no judge. And you may report the judge
result explicitly framed as an upper bound: *"faithfulness ≤ 0.85, measured with a
judge validated at 78% agreement whose errors are all in the lenient direction."*

**A rough correction.** If the judge is lenient on ~22% of the cases it approves,
the true rate is nearer `0.85 × (1 − 0.22) ≈ 0.66`. That is an estimate, not a
measurement — the sample is small and the error rate itself has wide confidence
intervals — but it tells you the gap is large enough to matter.

**What I'd do next, in order:**
1. **Read the disagreements.** One-sided error usually has one cause. Is the
   judge accepting claims that add unstated numbers? Rewarding fluency? The
   `reason` field it was forced to produce is the evidence.
2. **Tighten the prompt against that specific failure**, with 2–3 few-shot
   examples of the exact error — while being wary of module 14's finding that
   small models parrot prompt examples.
3. **Re-validate on a fresh sample.** Re-using the same 40 pairs after tuning
   against them is module 12's overfitting mistake: the number would no longer be
   held out.
4. **Consider a stronger judge** before more prompt engineering. Agreement at 78%
   with systematic bias often indicates the judge lacks the capability, not the
   instructions — and this project measured exactly that pattern (3B → 7B roughly
   doubled faithfulness on the same task).

</details>

---

**Previous:** [12 — Building a golden set](12-building-a-golden-set.md) ·
**Next:** [14 — Grounded generation](14-grounded-generation.md)
