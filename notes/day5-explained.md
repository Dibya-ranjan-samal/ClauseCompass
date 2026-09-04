# ClauseCompass Day 5 — Explained From Scratch

Grounded generation, deterministic validation, and the two numbers that make the
project credible. Companion to [day1](day1-explained.md), [day2](day2-explained.md),
[day3](day3-explained.md), [day4](day4-explained.md).

**The core insight:** don't trust the LLM to be accurate — **verify it in code**.
Every citation must resolve to a chunk we actually retrieved, and every quote must
string-match its source. That is deterministic, free, and catches hallucinations
before a user sees them.

**The headline, and it is uncomfortable:** the deterministic validator fired **64
times** and the local 3B model's **hallucination rate is 0.507** — measured with a
judge validated at **95% agreement** against 40 hand labels. Refusal, though, is
perfect: **18/18 unanswerable questions refused, zero false answers.**

---

## Contents

1. [Part 0 — Why verification beats trust](#part-0)
2. [Part 1 — The pipeline](#part-1)
3. [Part 2 — The metrics, explained](#part-2)
4. [Part 3 — Results](#part-3)
5. [Part 4 — Failures and fixes](#part-4)
6. [Part 5 — Glossary](#part-5)
7. [Part 6 — Interview questions](#part-6)
8. [Part 7 — Numbers and honest status](#part-7)

---

<a name="part-0"></a>
## Part 0 — Why verification beats trust

Day 4 proved the system can *find* the right clause. That is only half the
problem. A model handed the correct clause can still:

- cite a chunk that was never retrieved,
- invent a quote that appears nowhere,
- quote correctly and then draw a conclusion the quote does not support,
- assert that a contract *prohibits* something when it is merely *silent*.

The first two are catchable **in code, for free, deterministically**. That is the
day's central design choice: make the model emit structured output with quotes,
then check the quotes with `str.__contains__` instead of asking another model
whether it feels right.

The second two need semantic judgement — and a judge you have not validated is,
as the brief puts it, a vibe with a decimal point.

---

<a name="part-1"></a>
## Part 1 — The pipeline

```
question
   -> retrieval (precomputed: doc-scoped hybrid + cross-encoder, Day 4 winner)
   -> 6 excerpts, labelled [1]..[6]
   -> LLM with a strict JSON schema  ->  {answer_found, answer, claims[], confidence}
   -> DETERMINISTIC VALIDATION
        (1) does every cited label exist in what we retrieved?
        (2) does every quote string-match its cited chunk?
   -> claims split into kept / stripped
   -> answer_found = model said yes AND at least one claim survived
   -> LLM JUDGE on each kept claim -> faithfulness
```

### Two design decisions that came from earlier days

**1. Labels are `[1]`…`[6]`, not chunk ids.** Day 1's F1 was the model fabricating
`[Governing Law::27]` because chunk-id-shaped citations collide with the
contracts' own section numbering — and our ids literally contain `::s25::`. The
generator gets opaque integers and they are mapped back to real chunk ids after
validation.

**That fix reduced the habit but did not cure it.** From the very first smoke
test:

```
[q0001] Under the Transphorm,Inc Joint Venture agreement, what happens on a change of control?
   REASON: citation_not_retrieved  cited=['7.3.1(v)']  unknown=['7.3.1(v)']
   CLAIM : A change of control triggers a termination of the agreement if the
           acquiring/succeeding entity is deemed objectionable to the Japanese Government
```

`7.3.1(v)` is a **contract section number**, not one of the six labels it was
given. Same failure class as Day 1, four days later. The difference is that this
time the validator stripped both claims, `answer_found` flipped to false, and the
user got a refusal instead of an authoritative-looking unverifiable answer.

**That is the single best demonstration of the day's thesis.** The prompt fix
made the failure rarer; the code fix made it *harmless*.

**2. Retrieval is precomputed to a file.** The Day 4 winner costs ~11.6 s p95, so
running it inside every generation eval would mean ~29 minutes of re-ranking per
pass. The golden set is fixed, so `precompute_hits.py` retrieves once and writes
`golden_hits.json`. Cheaper than a cache layer, and nothing to invalidate.

### The prompt, and why each rule is there

| Rule | Failure it prevents |
|---|---|
| Cite every statement | Unverifiable prose that reads as authoritative |
| Verbatim quotes | Gives a **free deterministic** faithfulness check via string match |
| Exact refusal wording | Makes refusal machine-detectable, so it can be scored |
| "Silent ≠ prohibited" | The most dangerous legal-reasoning error there is |
| Untrusted-data framing | Prompt injection via contract text |

### The validation code, which is the point of the day

```python
for c in data.get("claims") or []:
    cits = [str(x) for x in (c.get("citations") or [])]
    unknown = [x for x in cits if x not in labels]
    if unknown or not cits:                       # (1) citation must exist
        stripped.append({**c, "reason": "citation_not_retrieved"})
        continue
    ratio = quote_ratio(c.get("quote", ""), [labels[x] for x in cits])
    if ratio < 0.95:                              # (2) quote must match source
        stripped.append({**c, "reason": "quote_unverified"})
        continue
    kept.append(c)
```

`quote_ratio` tries exact whitespace-insensitive containment first — one line,
free — and only falls back to `difflib`'s longest contiguous match for
punctuation drift.

Then the rule that makes it bite:

```python
found = bool(data.get("answer_found")) and bool(kept)
```

**An answer whose every claim failed validation is not an answer.** It becomes a
refusal. This is what converts a silent quality problem into a visible one.

---

<a name="part-2"></a>
## Part 2 — The metrics, explained

### Refusal: three rates, and you must report all three

| Metric | Definition | Why alone it lies |
|---|---|---|
| **correct refusal** | of unanswerable questions, fraction refused | A system that always refuses scores 1.0 |
| **false answer** | of unanswerable questions, fraction answered | The hallucination you actually fear |
| **over-refusal** | of answerable questions, fraction refused | A system that always answers scores 0.0 |

Reporting `correct_refusal` alone is the classic dishonest metric: *always refuse*
gets a perfect score and is useless. The pair is the tradeoff, and only the pair
is meaningful.

### Faithfulness: the judge, and why validating it is mandatory

An LLM reads each kept claim beside its cited excerpt and answers
`{"supported": true|false}`. Faithfulness is the fraction supported;
hallucination rate is `1 − faithfulness`.

**A judge parse failure must not count as "unsupported"** — that would inflate the
hallucination rate with judge bugs rather than model errors, so it returns `None`
and is excluded:

```python
except Exception:
    return {"supported": None, "reason": "judge_parse_failed"}
```

**Validating the judge (Task 6).** 40 claim/excerpt pairs, hand-labelled, then
scored against the judge. One detail matters: the sample is **stratified 20
supported / 20 unsupported**, because an all-supported sample cannot detect a
lenient judge — you would have no negatives to catch it on.

```
judge agreement: 38/40 = 95%
  judge too lenient (said supported, human said not): 1
  judge too strict  (said not, human said supported): 1
  verdict: trust it
```

**One error in each direction** — no systematic bias. The two disagreements are
instructive:

| # | Judge | Mine | Why |
|---|---|---|---|
| 8 | supported | **not** | The claim text is literally `"..."`. A vacuous claim cannot be supported; the judge affirmed it anyway |
| 39 | not supported | **supported** | The excerpt *is* `Section 8.01 No Assignment or Transfer Without Consent` and says exactly what the claim says. The judge refused it only because the excerpt says "this Agreement" instead of naming the Intellectual Property Agreement |

95% agreement is what licenses the sentence *"hallucination rate 0.507"*. Without
it that number would be unpublishable.

---

<a name="part-3"></a>
## Part 3 — Results

### Full run: 148 questions, qwen2.5:3b generating, qwen2.5:7b judging

```
  n_questions                148
  faithfulness_mean          0.493
  hallucination_rate         0.507
  n_judged_questions          69
  correct_refusal_rate       1.0        (18/18)
  false_answer_rate          0.0
  over_refusal_rate          0.469
  citation_validation_fires   64        (60 bad citation, 4 bad quote)
  claims_raw                 191  ->  claims_kept 127   (33% stripped)
  parse_failure_rate         0.007      (1 of 148)
  truncation_rate            0.0
  latency_p95_ms          34,101
  cost_total_usd             0.0        (fully local)
```

Against the brief's healthy bands: refusal is **excellent**, faithfulness is
**far below** (0.88–0.97 expected), over-refusal is **far above** (0.05–0.20
expected). `citation_validation_fires` of 64 is well inside the "3–20+" range and
emphatically not the suspicious zero.

### Where the over-refusals come from — measured, not assumed

61 of 130 answerable questions were refused. The brief's interpretation table says
`correct_refusal ≈ 1.0` with high over-refusal means "too conservative — soften
the refusal instruction". **The data says that would fix at most 40% of it:**

| Cause | Count | Share |
|---|---|---|
| Validator stripped every claim | 35 | **57%** |
| Model returned zero claims | 25 | 40% |
| JSON parse failure | 1 | 1% |

The majority is a **citation-format** problem, not conservatism. Softening the
refusal instruction would have left 57% of the over-refusals untouched — and made
the false-answer rate worse.

### The refusals are genuine, not a validator artifact

Worth checking, because the validator could have been silently doing the model's
job:

```
unanswerable: 18, all refused = True
  of those, how many had claims stripped?     0
  how many returned zero claims outright?    18
```

**All 18 returned zero claims on their own.** The perfect refusal rate is real
model behaviour. The validator never had to rescue it.

### The most important finding: the two checks are orthogonal

Every one of the 127 kept claims **already passed** verbatim quote validation —
≥95% string match against its cited chunk. The judge still rejected **61 of
them**:

| Claim (quote verified ✅) | Judge |
|---|---|
| "Neither party can terminate for convenience under the Dialog Semiconductor agreement." | The excerpt does not mention termination for convenience |
| "The agreement does not specify a notice period for convenience termination." | The excerpt does not mention any notice period |
| "The agreement imposes a minimum commitment of at least 250 Users per Business Entity." | The excerpt does not mention any minimum commitment in Users per Business Entity |

**The model quotes correctly and then draws a conclusion the quote does not
support.**

| Check | Catches | Count |
|---|---|---|
| Deterministic quote/citation validation | fabricated citations, invented quotes | 64 |
| Semantic judging | valid quote, unsupported conclusion | 61 |

Neither subsumes the other. **A project that ships only the deterministic check
would report "all citations verified" while hallucinating in half its claims.**
That is the single most transferable lesson of the day.

### The "silent ≠ prohibited" error, quantified

The prompt explicitly warns against it. It still happens, and it is measurably
the worse failure mode:

| Claim shape | n | judged unsupported |
|---|---|---|
| Negative assertions ("does not specify", "neither party can…") | 14 | **11 (78%)** |
| Positive claims | 113 | 50 (44%) |

Negative assertions fail **1.8× more often**. But note the honest reading:
positive over-claiming is the larger *absolute* problem (50 vs 11). A prompt tweak
aimed only at negatives would move the headline barely.

### Task 8 — generator comparison (60 questions, same 7b judge)

| Metric | qwen2.5:3b | qwen2.5:7b |
|---|---|---|
| faithfulness | 0.414 | **0.885** |
| hallucination rate | 0.586 | **0.115** |
| validator fires | 36 | **12** |
| claims raw → kept | 97 → 61 | 42 → 30 |
| **JSON parse failure** | **0.000** | **0.183** |
| over-refusal | 0.417 | 0.567 |
| p95 latency | 38.7 s | **347.5 s** |

**Model capacity is the dominant factor in faithfulness** — 0.414 → 0.885 moves
it from unusable into the brief's healthy band, and validator fires drop by
two-thirds because 7b cites correctly far more often.

Two honest counterweights:

1. **The 7b model is *worse* at JSON**, failing to parse 18% of the time against
   3b's 0%. Since a parse failure becomes a refusal, that is most of why its
   over-refusal is higher. Bigger is not uniformly better.
2. **347 s p95** — nearly six minutes per question. On an 8 GB machine a 4.7 GB
   model swaps heavily; swap hit 9.3 GB of 10.2 GB during this run. Unshippable.

**Caveat that must travel with this table:** the 60-question sample contains
**no unanswerable questions** (they sit at the end of `golden.jsonl`), so refusal
accuracy is not comparable here. Only the full 148-question 3b run measures it.

---

<a name="part-4"></a>
## Part 4 — Failures and fixes

### F25 — Section-number citations survived the Day 1 fix ✅ contained

Day 1 replaced chunk-id labels with `[1]`…`[k]` and fabrication dropped to 0/4 on
the stock questions. At 148 questions it is back: **60 bad-citation fires**, many
of them contract section numbers like `7.3.1(v)`.

**Cause.** The prompt cannot prevent it, only discourage it. Contract text is
dense with `7.3.1`-shaped tokens and the model pattern-matches onto them.

**Fix.** Not a better prompt — a **validator**. The claim is stripped, and if all
claims are stripped the answer becomes a refusal. The failure is now loud and
harmless instead of silent and authoritative.

**Lesson:** a prompt makes a failure rarer; only code makes it safe. Four days of
prompt engineering did not eliminate this, and ten lines of validation did.

---

### F26 — Quote validation gives false comfort ✅ measured

Every kept claim passed a ≥95% verbatim quote match, and **48% of them are still
unsupported**.

**Cause.** String matching proves the quote was copied, not that the claim follows
from it. The two are orthogonal properties.

**Why this matters more than it looks.** The deterministic check is the part that
is free, fast, and easy to be proud of. It is also the part that can make you stop
looking. Reporting "100% of citations verified" would have been *true* and
deeply misleading.

> **Interview line:** "I built deterministic citation and quote validation, which
> fired 64 times over 148 questions. Then I measured faithfulness with a
> judge — validated at 95% agreement against 40 hand labels — and found that 48%
> of the claims that *passed* quote validation were still unsupported. The model
> quotes correctly and then over-concludes. The two checks catch disjoint failure
> classes, and shipping only the cheap one would have looked rigorous while
> hallucinating in half the claims."

---

### F27 — The brief's diagnosis of over-refusal was wrong here ✅ re-attributed

`correct_refusal 1.0` plus `over_refusal 0.469` maps, in the brief's table, to
"too conservative → soften the refusal instruction".

**Measured cause:** 57% of over-refusals are the validator correctly stripping
fabricated citations. Only 40% are the model declining to produce claims.

Softening the refusal instruction would have addressed the minority, and pushed
the false-answer rate up from its current 0.0. **The right fix is better
citation compliance, not a laxer prompt** — and the 7b comparison confirms it:
validator fires fell 36 → 12 with a stronger model, no prompt change.

---

### F28 — Interleaved generate-then-judge would have thrashed 8 GB ✅ fixed before running

My first runner generated and judged per question, alternating a 1.9 GB generator
with a 4.7 GB judge. On 8 GB RAM that makes Ollama unload and reload a 4.7 GB
model **once per question**.

**Fix:** two passes. Generate all 148 answers, then judge all 127 claims. One
model resident at a time.

Even so, swap reached **9.3 GB of 10.2 GB** with 1.2M pageouts. Worth stating
plainly: the local-model path is memory-bound on this hardware, and that is the
real reason the latencies are what they are.

---

<a name="part-5"></a>
## Part 5 — Glossary

| Term | Definition |
|---|---|
| **Grounded generation** | Answering only from supplied context, not model memory |
| **Structured output** | Forcing JSON so the answer is machine-checkable |
| **Citation validation** | Checking every cited id was actually retrieved |
| **Quote verification** | String-matching the model's quote against its source |
| **Faithfulness** | Fraction of claims a judge finds supported by the cited text |
| **Hallucination rate** | `1 − faithfulness` |
| **LLM judge** | A model scoring another model's output |
| **Judge agreement** | How often the judge matches human labels; publishing faithfulness without it is not defensible |
| **Correct refusal** | Refusing when the answer genuinely is not there |
| **Over-refusal** | Refusing when the answer *is* there |
| **False answer** | Answering when it is not there — the dangerous one |
| **Stratified label sample** | Sampling both judge verdicts so both error directions are detectable |

---

<a name="part-6"></a>
## Part 6 — Interview questions

**"How do you stop it hallucinating?"**
> Two layers. Deterministically, the model must emit JSON with a verbatim quote
> per claim, and I check in code that every cited label was actually retrieved and
> every quote string-matches its chunk. If all claims fail, the answer becomes a
> refusal. That fired 64 times over 148 questions. Then I measure what code can't
> catch — semantic support — with a validated judge. Faithfulness is 0.49 with a
> 3B model and 0.89 with a 7B one.

**"Faithfulness of 0.49 sounds bad. Why report it?"**
> Because it's true, and because I can tell you exactly why: it's a 3-billion
> parameter model running locally with no API budget. The same pipeline with a 7B
> generator scores 0.885. The pipeline isn't the bottleneck, model capacity is —
> and I measured that rather than assuming it.

**"How do you know your judge is any good?"**
> I hand-labelled 40 claim/excerpt pairs, stratified 20/20 across its verdicts so
> I could catch bias in either direction, and it agreed 95% — one error each way.
> I quote that agreement rate next to every faithfulness number. An unvalidated
> judge is just a second opinion with a decimal point.

**"What was the most surprising result?"**
> That my deterministic quote validation gave false comfort. Every claim that
> reached the user had a verbatim-verified quote, and 48% of them were still
> unsupported — the model quotes accurately and then over-concludes. The cheap
> check and the expensive one catch disjoint failure classes, and I'd have shipped
> the cheap one and claimed rigour.

**"Your refusal rate is 100%. Isn't that just a system that refuses everything?"**
> Fair challenge, and it's why I report over-refusal too — 0.469, which is high.
> But I checked the mechanism: all 18 unanswerable questions returned zero claims
> from the model itself, not from my validator stripping them. And 57% of the
> over-refusals are the validator correctly rejecting fabricated citations, not
> excessive caution.

**"What would you fix first?"**
> Citation compliance, because it's 57% of my over-refusals and the 7B comparison
> shows it's model-driven, not prompt-driven. Then latency — 34 s p95 locally, and
> 347 s with the 7B model on 8 GB of RAM. Realistically this wants a hosted model,
> which is a cost decision rather than an engineering one.

---

<a name="part-7"></a>
## Part 7 — Numbers and honest status

### Record these

| Metric | Value |
|---|---|
| Faithfulness (qwen2.5:3b) | **0.493** |
| **Hallucination rate (3b)** | **0.507** |
| Faithfulness (qwen2.5:7b, n=60) | **0.885** |
| Hallucination rate (7b, n=60) | 0.115 |
| **Judge agreement (n=40)** | **95%** (38/40, 1 lenient / 1 strict) |
| **Correct refusal rate** | **1.000** (18/18) |
| False answer rate | **0.000** |
| Over-refusal rate | 0.469 |
| Citation validation fires | **64** (60 citation, 4 quote) |
| Claims raw → kept | 191 → 127 (33% stripped) |
| JSON parse failure (3b / 7b) | 0.007 / 0.183 |
| Cost per query | **$0.00** (fully local) |
| p95 end-to-end latency (3b / 7b) | 34.1 s / 347.5 s |

### Do not claim

- ❌ "Hallucination rate 4.9%" — that's the brief's illustrative example. Mine is
  **50.7%** with a 3B model, 11.5% with 7B.
- ❌ "Faithfulness 0.885" without saying it is **n=60 and a different generator**.
- ❌ "100% correct refusal" without the over-refusal rate beside it.
- ❌ "All citations verified" as a quality claim — 48% of verified-quote claims
  were still unsupported.
- ❌ "Attorney-validated judge" — the 40 labels are **AI-generated**, recorded in
  `evals/judge_labels.json` under `_provenance`.
- ❌ Any refusal comparison from the 7b run — that sample has no unanswerable
  questions.

### Status of every failure so far

| # | Failure | Status |
|---|---|---|
| F1–F2 | Fabricated citations, cross-contract splice | ✅ Day 1 |
| F3 | Scores can't express relevance | ✅ Day 4 |
| **F4** | **Citations sometimes missing** | ✅ **Day 5 — enforced, not just detected** |
| F5–F7 | Chunking and retrieval hygiene | ✅ Day 2 |
| F8 | Ollama truncated the prompt silently | ✅ Day 1 — and `num_ctx` set here too, 0% truncation |
| F9–F15 | Corpus, metadata and loader bugs | ✅ Day 2 |
| F16 | Contextual prefix unmeasured | ✅ Day 4 |
| F17 | Good retrieval, refused answer | ✅ Day 2 |
| F18–F20 | Eval-set design and harness bugs | ✅ Day 3 |
| F21–F24 | Scope-dependent signals, wrong-sign predictions, no-op sweep | ✅ Day 4 |
| F25 | Section-number citations survived the prompt fix | ✅ **contained by the validator** |
| F26 | Quote validation gives false comfort | ✅ **measured — 48% of verified claims unsupported** |
| F27 | Brief's over-refusal diagnosis wrong here | ✅ **re-attributed: 57% validator, 40% model** |
| F28 | Interleaved model loading would thrash 8 GB | ✅ fixed before running |

### Open going into Day 6

1. **Hallucination rate 0.507 on the shipped 3B model.** The fix is a bigger
   model, which means an API budget. The pipeline is not the limiting factor.
2. **Over-refusal 0.469.** Attack citation compliance first (57% of it).
3. **7b JSON compliance is 0.183 parse failure.** Needs a schema-constrained
   decoder (Ollama supports a `format` parameter) rather than prompt pleading.
4. **Latency.** 34 s p95 locally; 347 s with 7b on 8 GB. Memory-bound, not
   compute-bound — swap hit 91% during these runs.
5. **The judge is the same family as the generator** (qwen2.5). Same-family
   judging risks shared blind spots; a different model family would be a stronger
   design.

### Tomorrow

Day 6 is the agentic compliance auditor and observability. Today made single
answers trustworthy and *measurably* so. Tomorrow scales that trust across the
corpus — and the 0.507 hallucination rate is the number every agentic loop will
compound, so it is the thing to watch.
