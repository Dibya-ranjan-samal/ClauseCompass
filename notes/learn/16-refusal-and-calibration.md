# 16 — Refusal, Calibration & Asymmetric Error Cost

---

## 1. What it is

**Refusal** is a system saying "I don't know" or "not found in the provided
excerpts" rather than answering.

Measured as three separate rates, which must all be reported together:

| Metric | Definition | Over |
|---|---|---|
| **Correct refusal** | refused ÷ questions that *should* be refused | unanswerable questions |
| **False answer** | answered ÷ questions that should be refused | unanswerable questions |
| **Over-refusal** | refused ÷ questions that *should* be answered | answerable questions |

## 2. Why it exists

A system that always answers is a liability. A system that never answers is
useless. **Neither rate alone tells you anything**, because each is trivially
optimisable at the total expense of the other:

| Strategy | Correct refusal | Over-refusal | Useful? |
|---|---|---|---|
| Always refuse | **1.00** ✅ | **1.00** ❌ | no |
| Never refuse | **0.00** ❌ | **0.00** ✅ | no |

Reporting both is what demonstrates you understand the tradeoff. Reporting one is
how you mislead — usually yourself.

## 3. Making refusal measurable

Two design decisions turn refusal from a vibe into a number.

### An exact refusal string

```python
REFUSAL_TEXT = "Not found in the provided excerpts."
```

A fixed string is machine-detectable. "I'm not sure", "The excerpts don't
appear to contain…", "It seems this isn't addressed" are all refusals, and all
require fuzzy detection you'll get wrong.

### A structured boolean

```json
{"answer_found": false, "answer": "", "missing_info": "..."}
```

Better still: the model reports refusal as **data**, not prose. No parsing.

```python
"correct_refusal": (not q["answerable"]) and (not a.answer_found),
"false_answer":    (not q["answerable"]) and a.answer_found,
"over_refusal":    q["answerable"] and (not a.answer_found),
```

Three mutually exclusive booleans per question. Aggregate and you have the table.

**This only works because the eval set contains questions that should be
refused** — 18 of 148, with absence *verified* (module 12). You cannot measure
refusal without deliberately unanswerable questions, which is why they were
authored rather than hoped for.

## 4. The result

| Metric | Value | Healthy range (brief) |
|---|---|---|
| **Correct refusal** (18 unanswerable) | **1.00** | 0.75–0.92 |
| **False answer** | **0.00** | 0.08–0.25 |
| **Over-refusal** (130 answerable) | **0.469** | 0.05–0.20 |

Perfect on refusing the unanswerable. **Refusing 47% of the answerable ones.**

Per the brief's own interpretation table:

> correct_refusal ≈ 1.0 **and** over_refusal > 0.25 → *too conservative — refuses
> when it shouldn't* → soften the refusal instruction

That advice is **wrong here**, and finding out why required instrumenting the
cause.

## 5. Diagnosing over-refusal — don't accept the obvious explanation

61 of 130 answerable questions were refused. Why?

```python
for x in over_refusals:
    if x["parse_failed"]:                             why["parse_failed"] += 1
    elif x["n_bad_citation"] or x["n_bad_quote"]:     why["claims_stripped_by_validator"] += 1
    elif x["n_claims_raw"] == 0:                      why["model_returned_no_claims"] += 1
```

| Cause | count | share |
|---|---|---|
| **Validator stripped every claim** | 35 | **57%** |
| Model returned no claims at all | 25 | **40%** |
| Parse failure | 1 | 1% |

**Only 40% is model conservatism.** The majority — 57% — is module 15's validator
correctly rejecting claims with fabricated citations, which flips `answer_found`
to false (module 15 §4).

So "soften the refusal instruction" would address **at most 40%** of the problem
and would make the other 57% *worse* by letting unverifiable claims through. The
real fix for the majority case is a model that cites correctly.

> **Two mechanisms producing one metric.** An aggregate number invites a single
> explanation; instrumenting the *cause* per question is what separates them.
> Same lesson as module 04's blended median and module 11's mixed populations.

### And a genuinely good result hiding in the same data

```python
unanswerable: 18, all refused = True
  of those, how many had claims stripped?     0
  how many returned zero claims outright?     18
```

**All 18 unanswerable questions returned zero claims from the model itself** — the
validator never had to intervene.

That distinction matters enormously for what you may claim. If the validator had
been rescuing them, correct refusal of 1.00 would be an artifact of your
plumbing. Instead it's genuine model behaviour: shown five excerpts that don't
address cryptocurrency custody, the model recognised it had nothing.

**Correct refusal 1.00 is real.** You only know that because you checked the
mechanism, not just the rate.

## 6. Asymmetric error cost — the framing that matters more than any metric

Not all errors cost the same, and in compliance the asymmetry is severe:

| Error | Consequence |
|---|---|
| **False negative** — miss a real violation | undetected legal exposure; potentially a lawsuit |
| **False positive** — flag a compliant clause | ten minutes of a lawyer's review time |

**These differ by orders of magnitude.** So the system should be tuned toward
**recall**, accepting more false positives, and you should say so explicitly.

Now judge your own system against that:

| | Value | Direction |
|---|---|---|
| Correct refusal | 1.00 | conservative ✅ |
| Over-refusal | 0.469 | conservative ❌ |
| Agent precision (module 17) | 1.00 | conservative |
| **Agent recall** (module 17) | **0.20** | conservative ❌❌ |

**Every failure points the same way: the system under-asserts.** For a compliance
tool that is precisely backwards — it never false-alarms and misses four-fifths
of what it should catch.

One sentence of business judgement here is worth more than a metric:

> *In compliance, false negatives are the expensive error. A missed violation is
> a lawsuit; a false positive is ten minutes of review. So this system's
> precision-1.00 / recall-0.20 profile is optimised backwards for its own
> domain.*

## 7. Where the conservatism comes from

Not the prompt. Two pieces of evidence:

**1. It's a consistent model trait across tasks.** Over-refusal 0.469 (module 15's
pipeline), agent recall 0.20 (module 17's agent), faithfulness 0.493. Same
direction in three separate measurements with different prompts.

**2. Model capacity moves it; prompting barely does.** Swapping the generator 3B →
7B roughly doubled faithfulness on the overlapping sample. Meanwhile module 17's
prompt fix — sharpening the definition of "silent" — moved agent detections from
0/6 to 1/6 and left recall at 0.20.

**Conclusion: model capacity, not prompt wording.** Which makes it a budget
decision, and saying that plainly is more useful than more prompt engineering.

## 8. The related failure: naming an easy exit

Module 14's F13, restated because it belongs here.

`extract_metadata.py` initially instructed: *"Reply NONE if unsure."* The model
replied `NONE` **to almost everything**.

The mechanism is the same as module 14's prohibition finding: **naming an easy
exit in a prompt makes a small model take it.** A refusal path is necessary — a
model with no way to say "not found" will invent something — but describing it
too invitingly turns it into the default.

The resolution:

> **State the task positively; handle the empty case in code.**

Give the model one job — extract, or answer — and let your own logic detect the
empty result. Don't offer a comfortable alternative to doing the work.

There's a real tension with module 14's rule that an explicit refusal path
prevents invention. Both are true; the difference is *emphasis*. `"If the
excerpts are insufficient, set answer_found to false"` is a mechanism. `"Reply
NONE if unsure"` is an invitation.

## 9. Calibration — the concept behind all of this

**Calibration** is whether a model's expressed confidence matches its actual
accuracy. Perfectly calibrated: claims made with 80% confidence are right 80% of
the time.

The pipeline collects `confidence: "high" | "medium" | "low"`, which enables a
calibration check you should know how to run:

```python
for level in ("high", "medium", "low"):
    subset = [r for r in rows if r["confidence"] == level]
    print(level, mean(r["faithfulness"] for r in subset if r["faithfulness"]))
```

If high-confidence answers aren't measurably more faithful than low-confidence
ones, **the confidence field is noise** and should not be shown to users or used
for routing. Self-reported LLM confidence is frequently uninformative, and it's
cheap to check.

Better-calibrated signals, roughly in order of reliability:

1. **Cross-encoder relevance score** on the top chunk — a model trained to judge
   relevance, unlike a self-report.
2. **Retrieval score gap** between rank 1 and rank 2 — a large gap means the top
   result is distinctly better; near-ties mean the ranking is arbitrary.
3. **Agreement across retrievers** — if dense and BM25 both rank a chunk first,
   that's evidence (module 09's principle).
4. **Validator outcomes** — claims stripped is a directly observed quality signal,
   not a prediction.

**What you must not use:** the raw cosine score, for module 00's reason — on this
corpus 0.77 means "is legalese", so it cannot express confidence in relevance.

## 10. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **Reporting only correct refusal** | looks perfect, system is useless | report over-refusal too | always both |
| No unanswerable questions in the eval | refusal unmeasurable | count them; target 10–20% | author them, verify absence |
| Fuzzy refusal detection | miscounted refusals | exact string or a boolean field | structured `answer_found` |
| **Attributing over-refusal to one cause** | wrong fix applied | instrument the cause per question | the 57/40/1 breakdown |
| Validator-driven refusals uncounted | can't tell plumbing from model | check whether claims were stripped | record `n_bad_citation` |
| Naming an easy exit | model always takes it | check the output distribution | positive framing; code-side empty handling |
| Trusting self-reported confidence | routing on noise | correlate confidence with faithfulness | use cross-encoder score or the gap |
| Ignoring asymmetric cost | tuned for the cheap error | state which error costs more, then tune | optimise recall in compliance |

## 11. Compared with

| | Over-refusal | Hallucination |
|---|---|---|
| Failure | says nothing when it could answer | says something unsupported |
| User sees | "not found" | a confident wrong answer |
| Detectable by the user? | ✅ obviously | ❌ that's the problem |
| Cost in compliance | missed violation — **expensive** | wasted review time |
| Cost in consumer search | annoyance | reputational / safety risk |
| Direction it biases | conservative | reckless |

**The domain decides which you prefer.** In compliance, prefer hallucination
*with a citation you can check* over silence — because the citation makes the
error catchable, whereas silence yields nothing to check. In consumer-facing
health or legal advice, the ordering reverses.

## 12. Interview angle

**"How do you measure whether your system knows what it doesn't know?"** — With
questions that are deliberately unanswerable, and absence verified rather than
assumed. 18 of my 148 questions, each shipping the terms that must be absent from
the contract, checked against the full text — zero leaks. Then three rates:
correct refusal 1.00, false answer 0.00, over-refusal 0.469.

**"Correct refusal of 1.00 sounds too good. Is it real?"** — Fair suspicion, and I
checked the mechanism rather than the rate. All 18 unanswerable questions returned
**zero claims from the model itself** — the validator never intervened. Had the
validator been rescuing them, 1.00 would be an artifact of my plumbing. It's
genuine model behaviour.

**"Your over-refusal is 47%. The playbook says soften the refusal instruction."**
— That would fix at most 40% of it and make the rest worse. I instrumented the
cause: 57% of over-refusals are my validator correctly stripping claims with
fabricated citations, which flips `answer_found` false. Only 40% is model
conservatism. Softening the instruction would let unverifiable claims through to
address the smaller half.

**"Is this system shippable?"** — Not for compliance, and I'd say why. Every
failure points the same direction — over-refusal 0.469, agent recall 0.20,
precision 1.00. It never false-alarms and misses most of what it should catch,
which is backwards for a domain where a missed violation is a lawsuit and a false
positive is ten minutes of review. And it's model capacity rather than prompting:
3B → 7B roughly doubled faithfulness, while a prompt fix moved agent detections
from 0/6 to 1/6.

## 13. Test yourself

**Q1.** A system reports correct refusal 1.00. What single number do you demand
before believing it's good?

<details><summary>Answer</summary>

**Over-refusal.** Correct refusal is trivially maximised by refusing everything,
which also drives over-refusal to 1.00 and makes the system useless.

Reported here: correct refusal 1.00 *and* over-refusal 0.469 — so the perfect
score is substantially explained by a general reluctance to answer, not by
precise discrimination.

A close second: whether the eval set actually contains unanswerable questions, and
how many. Correct refusal over 2 questions is not a rate.

</details>

**Q2.** Your over-refusal is 47%. Give two mechanisms that could produce it and
the check that distinguishes them.

<details><summary>Answer</summary>

1. **Model conservatism** — it declines to commit, returning no claims at all.
2. **Downstream validation** — the model *does* answer, but every claim is
   stripped for a fabricated citation or unverifiable quote, so `answer_found`
   flips to false.

**The check:** for each over-refusal, look at `n_claims_raw` and the validator
counters. `n_claims_raw == 0` is conservatism; `n_bad_citation > 0` with claims
present is validation.

Measured here: 57% validation, 40% conservatism, 1% parse failure. They demand
opposite fixes — softening the prompt helps the second and worsens the first —
which is why the aggregate alone is actively misleading.

</details>

**Q3.** Why must a refusal be an exact string or a boolean rather than detected
by keyword?

<details><summary>Answer</summary>

Because refusal is expressible in unbounded ways — "I'm not sure", "the excerpts
don't appear to address", "this isn't specified", "unable to determine" — so any
keyword matcher has both false positives and false negatives.

False negatives undercount refusals, inflating apparent answer coverage. False
positives are worse: an answer *containing* "not found" in a different sense —
*"the term 'indemnity' is not found in Section 4, but Section 7 provides…"* —
would be scored as a refusal when it's a substantive answer.

A structured boolean (`answer_found`) removes the parsing problem entirely: the
model states its own status as data. The exact-string convention is the fallback
when you can't get structure.

</details>

**Q4.** Your system self-reports `confidence: high|medium|low`. How do you find
out whether that field is worth anything?

<details><summary>Answer</summary>

Correlate it with a measured quality signal — bucket questions by reported
confidence and compute faithfulness (or recall@5) within each bucket:

```python
for level in ("high", "medium", "low"):
    sub = [r for r in rows if r["confidence"] == level]
    print(level, len(sub), mean(r["faithfulness"] for r in sub if r["faithfulness"]))
```

If faithfulness is flat across the three buckets, the field is **noise** — it
should not be surfaced to users or used for routing, because it conveys no
information about correctness. If it's monotonic (high > medium > low), it's
usable, and you can pick a threshold for when to escalate to a human.

Also check the **distribution**: if 95% of answers are "high", the field has no
discriminative power regardless of correlation — a common failure, since models
tend to report high confidence by default.

</details>

**Q5.** Harder. You must reduce over-refusal from 0.47 to below 0.20 without
increasing false answers above 0.05. Give a concrete plan with the expected
contribution of each step.

<details><summary>Answer</summary>

Start from the causal breakdown: 61 over-refusals = 35 validator-stripped + 25
no-claims + 1 parse failure. Any plan must address the 35 first, since that's the
majority — and note the constraint (false answers ≤ 0.05) forbids simply loosening
validation.

**Step 1 — fix citation compliance, targeting the 35 (57%).**
These are answers where the model *had* something to say but cited it wrongly.
Rather than accept the strip, **retry once** with the invalid labels named
explicitly: *"You cited [7.3.1(v)], which is not one of the excerpt labels. The
available labels are [1]–[5]."* Module 14 warns that prohibitions backfire, but
this is corrective feedback on a concrete error rather than a general
prohibition — closer to the tool-error feedback that worked in module 17, where
the agent self-corrected from a rejection message. Expected recovery: a
meaningful share of 35. **Must be A/B'd**, not assumed.

**Step 2 — a larger generator, targeting both causes.**
The 25 no-claims cases are conservatism, which is capacity-driven; 3B → 7B
roughly doubled faithfulness. A stronger model both cites better (helping step 1's
population) and commits more readily. Likely the largest single effect, and a
budget decision rather than an engineering one.

**Step 3 — raise top-k, targeting the 25.**
Some no-claims cases may genuinely lack the evidence: corpus-wide recall@5 is
0.363, so for many questions the answer *wasn't retrieved*. Refusing is then
**correct behaviour** and would be wrong to eliminate. So first measure how many
of the 25 had a gold chunk in context at all — if few did, this isn't
over-refusal, it's a retrieval problem wearing a refusal costume, and the fix
belongs in modules 09–10.

**Guarding the false-answer constraint.**
False answers are currently 0.00 with margin to 0.05, i.e. you can afford roughly
one false answer on 18 unanswerable questions. So measure false-answer rate after
*every* step and stop if it moves. Critically, **do not loosen the validator** to
reduce over-refusal — that trades directly against the constraint and against the
project's core guarantee.

**The honest possibility to state up front:** if most of the 25 no-claims cases
had no gold chunk in context, then a large part of the 0.47 is not
over-refusal at all, and the target of 0.20 may be unreachable without first
fixing retrieval. Establishing that before optimising is step zero.

</details>

---

**Previous:** [15 — Verification layers](15-verification-layers.md) ·
**Next:** [17 — Agents](17-agents.md)
