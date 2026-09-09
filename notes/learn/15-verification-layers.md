# 15 — Deterministic vs Semantic Verification

The central finding of the project's generation work, and the one that would be
hardest to reconstruct from theory alone.

---

## 1. What it is

Two independent layers that check whether an answer is actually supported:

| Layer | Mechanism | Cost | Catches |
|---|---|---|---|
| **Deterministic** | string and set operations | free, microseconds | fabricated citations, invented quotes |
| **Semantic** | LLM judge (module 13) | one call per claim | valid quotes with unsupported conclusions |

**They are orthogonal.** Neither subsumes the other, and the measurement proving
that is §6.

## 2. Why both exist

The design insight of the whole day:

> **Don't trust the LLM to be accurate — verify it in code.**

Every citation must resolve to a chunk you actually retrieved. Every quote must
string-match its source. That is deterministic, free, and catches hallucinations
*before the user sees them*.

But module 14 already showed prompting is not enforcement. And this module shows
enforcement-by-string-matching is not sufficient either.

## 3. Layer 1: deterministic validation

```python
def validate(data, labels, quote_threshold=0.95):
    kept, stripped = [], []
    n_bad_citation = n_bad_quote = 0
    for c in data.get("claims") or []:
        cits = [str(x) for x in (c.get("citations") or [])]

        # (1) does every cited label exist in what we actually retrieved?
        unknown = [x for x in cits if x not in labels]
        if unknown or not cits:
            n_bad_citation += 1
            stripped.append({**c, "reason": "citation_not_retrieved", "unknown": unknown})
            continue

        # (2) does the quote actually appear in a cited chunk?
        ratio = quote_ratio(c.get("quote", ""), [labels[x] for x in cits])
        if ratio < quote_threshold:
            n_bad_quote += 1
            stripped.append({**c, "reason": "quote_unverified", "match_ratio": round(ratio, 3)})
            continue

        kept.append(c)
    return kept, stripped, n_bad_citation, n_bad_quote
```

### Check 1 — citation existence

A set membership test. Two cases are rejected:

- `unknown` — the model cited a label that wasn't offered (module 14's F1)
- `not cits` — the model made a factual claim with **no** citation at all

That second case is easy to forget and important: an uncited claim is
unverifiable by construction, so it cannot be allowed through.

### Check 2 — quote verification

```python
def quote_ratio(quote: str, texts: list[str]) -> float:
    nq = norm(quote)
    if not nq:
        return 1.0
    best = 0.0
    for t in texts:
        nt = norm(t)
        if nq in nt:
            return 1.0                          # exact, whitespace-insensitive
        m = SequenceMatcher(None, nq, nt).find_longest_match(0, len(nq), 0, len(nt))
        best = max(best, m.size / len(nq))      # longest contiguous run / quote length
    return best
```

Two-tier by design:

1. **Exact substring after whitespace normalisation.** A verbatim quote *should*
   appear literally; `norm()` collapses whitespace so line breaks and
   double-spaces in the source don't cause spurious failures. This is the fast
   path and returns immediately.
2. **Fuzzy fallback** — the longest contiguous matching block as a fraction of the
   quote, thresholded at 0.95. Tolerates a stray character, a smart quote, an
   en-dash. Anything below 0.95 means the model reworded, paraphrased or invented.

Note `find_longest_match` measures a **contiguous** run, not scattered common
characters. That matters: a paraphrase shares many characters but few long runs,
so it scores low. This is a deliberately strict similarity notion.

### Why the threshold is 0.95 rather than 1.0

Perfect matching would reject legitimate quotes over trivia — a curly apostrophe
instead of straight, a soft hyphen, a Unicode dash. 0.95 admits those and rejects
paraphrase. It is a **defensible approximation you should be able to justify**,
not a magic number.

## 4. The consequence: an answer whose claims all fail is not an answer

```python
found = bool(data.get("answer_found")) and bool(kept)
```

If every claim was stripped, `answer_found` becomes `False` and the user gets the
refusal string instead. This is the **safe failure direction**: incomplete rather
than unverifiable.

It also has a measurement consequence explored in module 16 — it makes
validator-driven refusals indistinguishable from model-driven ones unless you
instrument the cause.

## 5. What it caught

Over 148 questions:

| | |
|---|---|
| Claims produced by the model | **191** |
| Claims that survived validation | **127** |
| **Stripped** | **64 (33%)** |
| — bad citations | 60 |
| — bad quotes | 4 |
| Parse failures | 1 (0.007) |

**A third of the model's claims never reached the user.**

And the brief's warning is worth repeating: **`citation_validation_fires` of zero
is suspicious, not good.** It means either the validator isn't wired up or the
threshold is too loose. A validator that never fires is a comment.

### The single best demo case

Running the live API on *"Under the Transphorm, Inc Joint Venture agreement, what
happens on a change of control?"*:

```
REASON: citation_not_retrieved  cited=['7.3.1(v)']  unknown=['7.3.1(v)']
CLAIM : A change of control triggers a termination of the agreement if the
        acquiring/succeeding entity is deemed objectionable to the Japanese Government
```

`7.3.1(v)` is a **contract section number**, not one of the labels `1–6`. This is
module 14's F1 *recurring* even with opaque labels — the habit is reduced, not
eliminated. Both claims were stripped, `answer_found` went false, and the user
never saw an unverifiable answer.

**That is the layer working exactly as intended**, and it's why the fix and the
validator were built in the same change: the validator caught my own regression
within minutes.

## 6. The orthogonality — the finding that matters most

Now the measurement that reframes everything.

Of the 127 claims that **passed** deterministic validation, the judge (validated
at 95% agreement, module 13) marked:

| | |
|---|---|
| supported | 64 |
| **unsupported** | **61** |

**Every single one of those 61 had already passed verbatim quote validation at
≥95% match against its cited chunk.**

How can both be true? Because the model **quotes correctly and concludes
wrongly**. Real examples:

```
CLAIM : Neither party can terminate for convenience under the Dialog
        Semiconductor Strategic Alliance agreement.
JUDGE : The excerpt does not mention termination for convenience.

CLAIM : The agreement imposes a minimum commitment of at least 250 Users
        per Business Entity.
JUDGE : The excerpt does not mention any minimum commitment in terms of
        Users per Business Entity.
```

The quote is genuine. The **inference drawn from it** is not.

### State it precisely

| Layer | Detects | Structurally cannot detect |
|---|---|---|
| Deterministic | citation doesn't exist; quote isn't in the source | a true quote followed by a false conclusion |
| Semantic (judge) | unsupported inference, contradiction, added specifics | nothing about *format* — it isn't checking ids |

```
        ┌──────────────────────────────────────┐
        │  claims the model produced (191)     │
        │  ┌────────────────────────────────┐  │
        │  │ passed deterministic (127)     │  │   64 stripped: fake ids, fake quotes
        │  │  ┌──────────────────────────┐  │  │
        │  │  │ judged supported (64)    │  │  │   61 rejected: valid quote,
        │  │  └──────────────────────────┘  │  │   invalid inference
        │  └────────────────────────────────┘  │
        └──────────────────────────────────────┘
```

**Reporting only the deterministic number would have looked like a clean
system** — "we validate every citation and quote in code" is true and sounds
airtight. Faithfulness 0.493 is what happens when you also check the semantics.

## 7. The negative-assertion pattern

Digging into *which* claims fail reveals structure:

| Claim type | n | judged unsupported |
|---|---|---|
| **Negative assertions** | 14 | **11 (78%)** |
| Positive claims | 113 | 50 (44%) |

Negative assertions — *"the agreement does not specify…"*, *"neither party
can…"*, *"the contract is silent on…"* — fail **1.8× more often**.

**Why, mechanically:** proving absence requires surveying the *whole* document.
The model sees 5 excerpts. It cannot distinguish *"not in these excerpts"* from
*"not in the contract"*, so it over-generalises from a sample to the whole.

This is module 14's "silent ≠ prohibited" rule being violated, now quantified.

**But read the absolute numbers before prioritising.** Only 14 of 127 claims were
negative, so they explain 11 of 61 failures. **Positive over-claiming is the
larger absolute problem** (50 of 61). A conclusion that stopped at "negatives are
worse" would have sent you after 18% of the issue.

> Rate and volume are different questions. Fix the one that moves the total.

## 8. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **Validator never fires** | `fires == 0` | count fires; it should be non-zero | check it's wired; tighten threshold |
| Threshold too loose | paraphrases pass as quotes | sample passed quotes and read them | raise toward 0.95 |
| Threshold too strict | legitimate quotes rejected | read `match_ratio` on rejections | normalise whitespace; allow 0.95 |
| Uncited claims allowed through | unverifiable prose reaches users | require `cits` non-empty | reject empty citations |
| **Trusting only deterministic checks** | looks clean, hallucinations remain | add a validated judge | both layers |
| Judge without validation | unquotable number | hand-label 40, stratified | module 13 |
| Stripping silently | users see a shorter answer with no signal | report `stripped_claims` | surface them (the API does) |
| Comparing char-level similarity naively | scattered matches inflate the score | use longest *contiguous* match | `find_longest_match` |

## 9. Surfacing rejections rather than hiding them

The API returns `stripped_claims` and the UI renders them in a red panel. That is
a deliberate product choice:

- A reviewer sees **what the model asserted and why it was rejected**, which is
  more informative than a shorter answer with no explanation.
- It makes the system's own error rate visible to its user, which for a legal tool
  is the honest posture.
- It is the best demo material in the project — the model being caught, live.

## 10. Compared with

| | Deterministic | Semantic judge |
|---|---|---|
| Cost per claim | ~0 | one LLM call |
| Latency | microseconds | seconds |
| Deterministic across runs | ✅ exactly | ⚠️ at temp 0, per model version |
| Needs its own validation | ❌ | ✅ (95% agreement here) |
| Catches fake ids / fake quotes | ✅ | incidentally |
| Catches false inference | ❌ **structurally** | ✅ |
| Runs in production per request | ✅ | usually too slow/costly |
| Runs on the eval set | ✅ | ✅ |

**The production architecture that follows:** deterministic checks on **every**
request (free, so no reason not to), the judge on the **eval set** (to know your
hallucination rate), and human labels on a **sample of the judge** (to know the
judge is worth quoting). Each layer validates the one above.

## 11. Interview angle

**"How do you prevent hallucinated citations?"** — Two layers, and the second
exists because the first is insufficient. Deterministically, every cited label
must be one I actually retrieved and every quote must string-match its source
after whitespace normalisation, with a 0.95 fuzzy fallback for punctuation drift.
That stripped 33% of claims — 60 bad citations, 4 bad quotes. Then a validated
LLM judge for semantic support.

**"Your citation validation strips 33% of claims. Isn't that alarming?"** — It's
the system working. Those claims would otherwise have reached the user
unverifiable. What's genuinely alarming is the *next* number: of the claims that
passed, the judge rejected 48%. Every one of those had a verbatim-correct quote.

**"So string matching isn't enough?"** — Structurally not, and this is the finding
I'd lead with. Deterministic checks catch fabricated citations and invented quotes
— 64 fires. They cannot catch a *correct* quote followed by a *wrong* inference,
which was 61 claims. The two layers are orthogonal, and reporting only the
deterministic one would make the system look clean.

**"Which claims fail most?"** — Negative assertions, at 78% unsupported versus
44% for positive claims. Proving absence needs the whole document and the model
sees five excerpts, so it can't tell "not in these excerpts" from "not in the
contract". But only 14 of 127 claims were negative, so positive over-claiming is
the bigger absolute problem — the rate and the volume point at different fixes.

## 12. Test yourself

**Q1.** Why check citation existence *and* quote content? Give a claim that
passes one and fails the other.

<details><summary>Answer</summary>

They check different things: existence is a set membership test on the id, content
is a substring test on the text.

**Passes existence, fails quote:** a claim citing label `[2]` — genuinely
retrieved — with `quote: "liability shall not exceed one million dollars"` when
chunk 2 actually says *"This Agreement is governed by the laws of Delaware"*. The
id is real; the quote is invented.

**Passes quote, fails existence:** a claim quoting text that genuinely appears in
retrieved chunk 3, but citing `[7]` when only 1–5 were offered. The evidence
exists; the pointer doesn't.

Both are hallucinations of different kinds, and each check catches one.

</details>

**Q2.** Your `citation_validation_fires` is 0 over 148 questions. Why is that
suspicious rather than reassuring?

<details><summary>Answer</summary>

Because it is implausible for a real system. Even strong models occasionally cite
something not offered or paraphrase a "verbatim" quote; a 3B model does it 33% of
the time. Zero fires almost certainly means the validator isn't doing its job.

Likely causes, in order of probability:
1. **Not wired up** — the validation function runs but its result is never used to
   filter, or it's called on the wrong field.
2. **Threshold too loose** — a quote threshold near 0 passes anything.
3. **Labels always match trivially** — e.g. you validate against the full chunk
   store rather than only the retrieved subset, so any real chunk id passes.

The diagnostic: deliberately inject a bad claim and assert it gets stripped. If a
fabricated citation passes, the validator is decorative.

</details>

**Q3.** All 61 claims the judge rejected had passed verbatim quote validation.
Explain how that's possible, and what it implies about verification design.

<details><summary>Answer</summary>

The model quoted its source **accurately** and then drew a conclusion the source
doesn't license. For example, quoting *"Either Party may terminate this Agreement
at any time prior to delivery"* and asserting *"neither party can terminate for
convenience"* — the quote is real, the inference contradicts it.

String matching verifies **provenance**, not **entailment**. It confirms the text
came from the source; it says nothing about whether the claim follows from the
text.

Design implication: **verification layers must be chosen by failure mode, not by
strength.** You cannot make a string matcher strict enough to catch inference
errors — no threshold does it, because the string *is* correct. You need a layer
that reasons about meaning, which means a model, which means that layer itself
needs validating (module 13). Layered verification, each layer catching what the
one before structurally cannot.

</details>

**Q4.** Why does `quote_ratio` use `find_longest_match` rather than counting
shared characters or using an edit distance?

<details><summary>Answer</summary>

Because a verbatim quote should be a **contiguous** run of the source, and
contiguity is the property that distinguishes quoting from paraphrasing.

- **Shared-character counting** is far too permissive: two texts in the same
  register share most of their character *inventory* while sharing no phrasing.
  "liability shall not exceed" and "shall the party be liable" score high on
  character overlap and are entirely different statements.
- **Edit distance** conflates several small differences with one large one. A
  paraphrase with the same length can have a modest edit distance while being a
  different assertion.

`find_longest_match` asks: what is the longest unbroken stretch of the quote that
appears verbatim in the source? Dividing by quote length gives the fraction
genuinely lifted. A paraphrase shares many characters but few long runs, so it
scores low — which is the behaviour you want.

</details>

**Q5.** Harder. You're asked to reduce the 0.507 hallucination rate. Rank three
interventions by expected impact, using the module's numbers.

<details><summary>Answer</summary>

**1. A larger generator — highest expected impact.**
The failures are inference errors, not format errors, and inference quality is
the thing model capacity buys. This project measured 3B → 7B roughly doubling
faithfulness on the overlapping sample. No prompt change addresses "quotes
correctly, concludes wrongly" — that is a reasoning limitation. It is also a
budget decision rather than an engineering one, which is worth saying plainly.

**2. Target positive over-claiming — highest impact among prompt/pipeline fixes.**
50 of 61 failures are positive claims, so this is where the volume is. Concretely:
require that each claim's `statement` be *derivable from the quote alone*, and add
a self-check step asking the model to re-read its quote and confirm the statement
follows. Riskier than it sounds given module 14's finding that small models parrot
and over-trigger on prompt instructions — so it must be A/B'd, not assumed.

**3. Suppress negative assertions — highest *rate*, lowest *volume*.**
78% unsupported, but only 14 claims, so at best it recovers ~11 of 61 failures
(18%). Cheap and well-targeted: detect negative assertions programmatically and
either strip them, or require they cite a *survey* rather than an excerpt — since
absence cannot be established from 5 chunks. Attractive because it's a code-level
guarantee rather than a prompt hope.

**What I would *not* do:** tighten the quote threshold. Every failing claim already
passed at ≥0.95, so no threshold change touches them — the layer is orthogonal to
the failure. Recognising that a lever doesn't connect to the outcome is the point
of the orthogonality finding.

</details>

---

**Previous:** [14 — Grounded generation](14-grounded-generation.md) ·
**Next:** [16 — Refusal & calibration](16-refusal-and-calibration.md)
