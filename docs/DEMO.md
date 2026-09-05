# 3-minute demo script

Not recorded — this is the shot list, with the real numbers to say out loud.
Show behaviour and numbers; do not narrate code.

**Setup:** `make serve` in one terminal, `make ui` in another, browser tabs open
on `localhost:8501` and `localhost:8000/docs`.

| Time | Screen | Say |
|---|---|---|
| 0:00–0:20 | Title / corpus | "530 real contracts, 18,280 clauses. One question — which of these lack a cap on liability? Today that's a week of reading." |
| 0:20–0:55 | UI: ask a doc-scoped question | "Answer with claims." **Click a citation, expand it, show the clause text.** Slow down — this is the whole point. "That quote was string-matched against the source before you saw it." |
| 0:55–1:20 | UI: the red "Rejected by the validator" panel | "The model also asserted this. It cited a chunk that wasn't retrieved, so it was stripped. Across the eval set that fired 64 times and removed 33% of claims." |
| 1:20–1:45 | UI: ask an unanswerable question | "12% of my eval set is deliberately unanswerable, with absence verified against the full text. It refuses 100% of those — and over-refuses 47% of answerable ones, which is the real weakness." |
| 1:45–2:15 | `POST /audit` in Swagger, then poll | "Audits take minutes, so the API returns a job id immediately. Findings carry citations, and a finding that asserts a violation without one cannot be constructed — Pydantic rejects it." |
| 2:15–2:40 | UI: Traces tab | "Every stage is a span. `agent.step` is 413 of 443 seconds — the LLM is 93% of wall clock, so optimising the vector DB would recover 7%." |
| 2:40–3:00 | README results tables | "recall@5 0.575 to 0.762 doc-scoped. Corpus-wide the same pipeline didn't help, and that's in the README too — recall@50 of 0.585 says 40% is never retrieved, which is a retrieval problem, not a ranking one." |

**Close on the honest line, not a triumphant one:** *"The headline number is
faithfulness 0.49 with a judge I validated at 95% agreement. That's a small local
model, and I can show you exactly which claims it got wrong and why."*

## The 10-second README GIF

Just the click-citation-to-clause-text flow, nothing else.

```bash
# record a window with QuickTime, then:
ffmpeg -i demo.mov -vf "fps=12,scale=900:-1:flags=lanczos" -loop 0 docs/demo.gif
```

Then add to the top of README.md: `![demo](docs/demo.gif)`
