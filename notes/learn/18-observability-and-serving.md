# 18 — Observability & Serving

The last module, and the one that converts "it gave a wrong answer" into a
five-minute diagnosis.

---

## 1. What it is

**Observability**: recording what your system did, in enough structure to answer
questions you didn't anticipate.

**Serving**: exposing it as an API that loads models once, doesn't block, and
tells you when it's unhealthy.

## 2. Why observability comes first

A single-shot RAG call is easy to debug: print the chunks, print the answer. An
agent that loops — search, read, decide, retry — has **no single place to print**.

You will end up adding logging anyway. So build it first and build it
**queryable**, because the questions you'll want to ask are not the ones you'd
guess:

- Which stage consumes the latency?
- Did the re-ranker promote or demote the right chunk?
- Was that tool call rejected, and why?
- How many steps did this contract take?

`print()` cannot answer any of those after the fact. A table can.

## 3. Spans and traces

| Term | Meaning |
|---|---|
| **Span** | one timed, attributed unit of work — a retrieval, an LLM call, a tool |
| **Trace** | a tree of spans sharing a `trace_id` — one whole request |
| **Parent** | the span that contains this one, giving the tree its shape |

```
audit                                   25,273 ms
├── agent.plan                                9 ms   (cached)
├── agent.audit_contract                 25,237 ms
│   ├── agent.step (0)                    7,578 ms
│   │   └── tool.search_clauses           4,500 ms
│   │       └── retrieve.dense            4,477 ms
│   ├── agent.step (1)                    6,982 ms
│   │   └── tool.search_clauses           2,483 ms
│   ├── agent.step (2)                    6,212 ms
│   │   └── tool.record_finding               7 ms   rejected: bad_args
│   └── agent.step (3)                    4,464 ms
│       └── tool.record_finding               7 ms   status: silent
```

That tree *is* the debugging story. Read it and you know exactly what happened.

## 4. One table, not two

```sql
CREATE TABLE spans (
    span_id TEXT PRIMARY KEY, trace_id TEXT, parent_id TEXT, name TEXT,
    started_at REAL, duration_ms REAL, attributes TEXT, status TEXT
);
```

The obvious design adds a second `traces` table holding per-trace duration, cost
and token totals. **Dropped**, because every one of those is derivable:

- a trace's **root** is the span with `parent_id IS NULL`
- its **duration** is the root's duration
- its **cost** is `sum()` over its spans

```sql
SELECT r.trace_id, r.name, r.duration_ms,
       (SELECT count(*) FROM spans s WHERE s.trace_id = r.trace_id) AS n_spans,
       (SELECT coalesce(sum(json_extract(s.attributes,'$.cost_usd')),0)
          FROM spans s WHERE s.trace_id = r.trace_id) AS cost_usd
FROM spans r WHERE r.parent_id IS NULL
ORDER BY r.started_at DESC;
```

**Two tables means two writes per trace that can disagree.** Derived data that
can drift from its source is a bug waiting to happen. Storing it once and
computing the rollup is strictly safer.

## 5. The implementation, and the one line that matters

```python
@contextlib.contextmanager
def span(self, name: str, **attrs):
    if not self._stack:                        # no parent => this starts a trace
        self.trace_id = uuid.uuid4().hex[:16]
    sid = uuid.uuid4().hex[:16]
    parent = self._stack[-1] if self._stack else None
    self._stack.append(sid)
    bag, status, t0 = dict(attrs), "ok", time.perf_counter()
    try:
        yield bag                              # callers mutate bag to add attributes
    except Exception as e:
        status = f"error: {type(e).__name__}"
        raise
    finally:
        self._stack.pop()
        self.conn.execute("INSERT INTO spans VALUES (?,?,?,?,?,?,?,?)", (...))
```

Three things worth naming:

**`finally` is load-bearing.** A span that raises still gets written, with its
error status. **A tracer that only records successes lies exactly when you need
it most.** The `raise` re-raises after recording, so behaviour is unchanged.

**No `start_trace`/`end_trace`.** A span with an empty stack *is* a new trace, so
there is no paired call to forget. Removing a whole class of bug by removing the
API that permits it.

**The `bag` is yielded, not returned.** Callers add attributes as they learn them:

```python
with tracer.span("retrieve.dense", doc_id=doc_id, k=k) as sp:
    ...
    sp["n_hits"] = len(res)
    sp["top_score"] = float(res[0].score) if res else None
```

Attributes discovered *during* the work are the useful ones.

## 6. The attribute that pays for the whole system

```python
before = {h.chunk_id: i for i, h in enumerate(cands, 1)}
sp["moves"] = [{"chunk_id": h.chunk_id, "was": before[h.chunk_id], "now": i}
               for i, (_, h) in enumerate(ranked[:5], 1)]
```

**Before/after ranks on the re-rank span.** This single attribute answers "did
the re-ranker help or hurt *this query*?" — which aggregate metrics can never
tell you, because they average over queries.

The general principle: **record the thing you'd have to re-run the system to
recover.** Timings you can re-measure; a specific query's rank movement you
cannot.

## 7. What the traces revealed

### Where the time actually goes

| Span | n | total | avg |
|---|---|---|---|
| `agent.step` (LLM) | 75 | **413 s** | 5,511 ms |
| `tool.search_clauses` | 32 | 30 s | 924 ms |
| `retrieve.dense` | 46 | 37 s | 807 ms |
| `tool.record_finding` | 38 | 0.1 s | 3 ms |
| `tool.get_clause` | 6 | 0.0 s | 0 ms |

**The LLM is 93% of wall clock** (413 s of 443 s).

So "how would you make this faster?" now has a data-backed answer:
**optimising the vector database recovers at most 7%.** Cut steps per contract
(currently 2.9) or output tokens. Without the traces, "optimise retrieval" would
have been a plausible and wasted week.

### A guardrail self-correcting, visible

Look at steps 2 and 3 in §3. `tool.record_finding` fires **twice**:

```
tool.record_finding    7 ms   {"doc_id": null, "rejected": "bad_args"}
tool.record_finding    7 ms   {"doc_id": "contract_01", "status": "silent"}
```

The first call omitted `doc_id` and was **rejected by the Pydantic argument
model**. The agent read the error message and retried correctly.

That is module 17's validation working — and **it is only visible because the
rejection was recorded as an attribute.** Aggregate counts told me 31% of
`record_finding` calls were rejected; the trace told me *why* and that the agent
recovered.

### A hypothesis the traces let me test — and I was wrong

The traces showed `retrieve.dense` at 4,477 ms and 2,483 ms inside the agent,
against Day 4's measured **29 ms** p95. My hypothesis: GPU contention, since
SentenceTransformer (MPS) and Ollama (Metal) want the same device.

Measured it:

```
dense, ollama idle      :  130 ms
dense, ollama generating:  131 ms   (1x slower)
```

**No contention.** The hypothesis was wrong. The 4.5 s first call is MPS kernel
warm-up; the 2.5 s second call I could not account for, and with swap at 93%
paging is the likely contributor — but I didn't isolate it, so it's recorded as
**unexplained rather than asserted**.

> Third time in this project a confident causal story failed a five-minute
> measurement. The cheap experiment keeps being worth it — and "I don't know"
> recorded honestly is worth more than a plausible fabrication.

## 8. Serving — the API

### Load models once

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    dense = DenseRetriever(collection=CFG.collection)
    STATE["scoped"] = RerankingRetriever(hybrid, ...)
    STATE["llm"] = OllamaProvider()
    STATE["chunks"] = json.loads(Path("data/interim/chunks_enriched.json").read_text())
    yield
    STATE.clear()
```

**The classic FastAPI + ML mistake is constructing the model per request**, which
turns a 40 ms call into a 10 s one. `lifespan` runs once at startup and once at
shutdown.

The check: **the second request should be much faster than the first.** If it
isn't, you're loading per request.

### Don't block the event loop

```python
@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    # Retrieval and local inference are blocking CPU/GPU work; off the event loop.
    return await asyncio.to_thread(_ask, req.question, req.doc_id, req.top_k)
```

`async def` does **not** make blocking code non-blocking. A synchronous 13-second
call inside `async def` stalls *every* other request, including health checks.
`asyncio.to_thread` moves it to a worker thread and yields control.

### Long work gets a job id

```python
@app.post("/audit", status_code=202)
async def start_audit(req: AuditRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "report": None}
    bg.add_task(_run_audit_job, job_id, req.policy, req.max_contracts)
    return {"job_id": job_id, "status": "running",
            "poll": f"/audit/{job_id}", "eta_s": req.max_contracts * 22}
```

Audits take ~22 s per contract. **202 Accepted with a job id**, poll for the
result. Returning an `eta_s` derived from a *measured* per-contract cost is a
small touch that makes the API honest.

### Encode measurements, not defaults

```python
retriever = STATE["scoped"] if doc_id else STATE["dense"]
```

**The cross-encoder runs only when a `doc_id` is supplied.** Because module 10
measured it at recall@5 0.762 doc-scoped but **0.331 corpus-wide** — worse than
plain dense (0.363) for 300× the latency. Applying it unconditionally would be
slower *and* less accurate on the common path.

> **A measured finding belongs in the code, not just the README.** The conditional
> *is* the finding, expressed executably.

### Health and metrics

```python
@app.get("/healthz")
async def healthz():
    loaded = {"dense", "scoped", "llm"} <= STATE.keys()
    try:    STATE["dense"].client.get_collections(); qdrant = True
    except Exception: qdrant = False
    ollama = await asyncio.to_thread(_ping)
    return {"status": "ok" if (loaded and qdrant and ollama) else "degraded", ...}
```

Checks **dependencies**, not just process liveness — a process that's up with a
dead Qdrant is not healthy.

Note `_ping` goes through `asyncio.to_thread`: a blocking `httpx.get` inside
`async def` stalls the event loop, which is an easy bug to ship in a health check
of all places. Caught here by `ruff`'s `ASYNC210`.

```python
@app.get("/metrics")
async def metrics():
    out = {"retrieval": json.loads(Path("evals/baseline.json").read_text())}
    gen = sorted(Path("runs").glob("*gen-final.json"), key=lambda p: p.stat().st_mtime)
```

Serving your recorded eval numbers makes quality claims **inspectable at
runtime**, not just in a README.

And note `key=lambda p: p.stat().st_mtime`. The obvious `sorted(glob(...))[-1]`
is **alphabetical**, which silently served a one-contract smoke run instead of the
real one earlier in this project. Sorting file paths by name when you mean
"newest" is a bug that looks like working code.

## 9. The CI eval gate

```yaml
- name: Gate on regression
  run: |
    for metric, tol in (("recall@5", 0.02), ("mrr", 0.02)):
        delta = cur[metric] - base[metric]
        if delta < -tol:
            print(f"::error::{metric} regressed by more than {tol}")
            sys.exit(1)
```

**Retrieval-only, so no LLM calls** — free and fast. Generation eval needs a local
model and runs manually.

This is the piece most portfolio projects lack, and the reasoning is worth
stating: unit tests catch *code* regressions; an eval gate catches **quality**
regressions, which are invisible to unit tests. A refactor that silently drops
the query prefix (module 06) passes every unit test and costs 5–10% recall.

**Honest status:** this workflow has never executed — the repository had no remote
when it was written. It's wired to the real entrypoints and the real nested
`baseline.json`, and labelled as untested in both the workflow header and the
README. Claiming a green CI badge you haven't earned is exactly the kind of thing
an interviewer checks.

## 10. Lint caught a five-day-old regression

Worth including because it argues for a tool people skip.

```
F821 Undefined name `sys`  --> ask.py:148
```

`ask.py` has called `sys.stderr` since Day 1. On Day 2 `import sys` was replaced
with `import argparse`. So **every warning path had been dying with `NameError`
for five days** — including the citation-validation warning that Day 1 exists to
produce.

```
NameError: name 'sys' is not defined
```

It never surfaced because the warning path only triggers on unusual answers, and
the happy path never touches it. **Static analysis found in one second what five
days of running the program did not** — because static analysis doesn't need the
buggy branch to execute.

Also caught: `ASYNC210` (the blocking health check above) and `B023` (a closure
capturing a loop variable).

## 11. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| **Tracer records only successes** | errors invisible | force an exception, check the row | write in `finally` |
| Models loaded per request | every request slow | compare 1st vs 2nd request | `lifespan` |
| **Blocking call in `async def`** | one slow request stalls all | health check during a slow call | `asyncio.to_thread` |
| Derived data in a second table | totals disagree with spans | recompute and compare | one table, roll up in SQL |
| `sorted(glob(...))[-1]` for "newest" | serves the wrong file | sort by `st_mtime` | sort by mtime |
| Health check ignores dependencies | "healthy" with a dead database | kill Qdrant, call `/healthz` | check dependencies |
| Long work in the request path | timeouts | measure p95 | 202 + job id |
| No eval gate | quality regressions merge | deliberately break something | CI gate on the metric |
| Unbounded trace growth | disk fills | check DB size over time | retention policy |
| Tracing in the hot path | latency overhead | measure with tracing off | sample, or async writes |

That last row matters at scale: this project commits after every span, which is
fine at 232 spans and wrong at 232,000. Batching or sampling is the production
answer.

## 12. Compared with

| | `print()` | Structured spans | OpenTelemetry |
|---|---|---|---|
| Queryable after the fact | ❌ | ✅ SQL | ✅ |
| Parent/child structure | ❌ | ✅ | ✅ |
| Setup cost | none | ~40 lines | a collector, a backend |
| Cross-service | ❌ | ❌ | ✅ |
| Vendor-neutral standard | — | ❌ | ✅ |
| Right for | a script | one service | a distributed system |

**Be honest about the choice:** a 40-line SQLite tracer is right for a
single-process project and would be wrong for a fleet. OTel's value is
propagating trace context *across services*, which this system doesn't have. The
concepts — spans, traces, attributes, parent ids — are deliberately the same, so
migrating is mechanical.

## 13. Interview angle

**"How do you debug a wrong answer?"** — Open its trace and walk the chain. Was
the gold chunk in the top 50? If yes but not top 5, it's a ranking problem. If
no, it's retrieval. If the right chunk *was* in the top 5 and the answer is still
wrong, it's generation. My best example: all 12 contracts came back "silent", and
four SQLite queries showed retrieval returned genuine cap language while the
agent's own rationale said *"includes a limitation of liability but does not
explicitly mention a liability cap"* — right chunk, wrong reasoning. A
generation problem, fixed with one line of prompt.

**"How would you make this faster?"** — From the traces rather than intuition:
`agent.step` is 413 of 443 seconds, so the LLM is 93% of wall clock and retrieval
is 7%. Optimising the vector database recovers almost nothing. I'd cut steps per
contract — currently 2.9 — and cap output tokens. That's the answer I'd have got
wrong without instrumentation.

**"Why SQLite rather than OpenTelemetry?"** — Because this is one process. OTel's
main value is propagating trace context across service boundaries, which I don't
have, and it costs a collector and a backend. I used the same concepts — spans,
traces, parent ids, attributes — so migrating is mechanical if it becomes
distributed. I'd also mention the design choice: one table, with trace-level
totals rolled up in SQL rather than stored, so there's nothing to drift.

**"What's in your CI?"** — Unit tests plus a **retrieval-only eval gate** that
fails the build if recall@5 or MRR regress by more than 0.02. Retrieval-only
means no LLM calls, so it's free and fast. It matters because unit tests catch
code regressions and are blind to quality ones — a refactor that drops the BGE
query prefix passes every test and costs 5–10% recall. Honest caveat: mine has
never actually run, because the repo had no remote, and it's labelled as untested.

## 14. Test yourself

**Q1.** Why must the span be written in a `finally` block rather than after the
`yield`?

<details><summary>Answer</summary>

Because code after `yield` in a context manager is skipped when the body raises.
Without `finally`, any span whose work throws is never recorded — so the tracer
goes silent precisely when something breaks, which is when you need it.

`finally` guarantees the write on both paths, and recording the exception type in
`status` turns the failure itself into queryable data. The `raise` after recording
preserves normal exception propagation, so tracing changes observability without
changing behaviour.

</details>

**Q2.** Your endpoint is `async def` and calls a synchronous 13-second retrieval.
What happens to a concurrent `/healthz` request?

<details><summary>Answer</summary>

It waits — potentially the full 13 seconds. `async def` does not make blocking
code asynchronous; it only marks the function as a coroutine. A synchronous call
inside it holds the single event-loop thread, so **no other request is served
until it returns**, including health checks.

The practical consequences are nasty: your load balancer's health probe times
out and it marks a working instance unhealthy, and latency for every concurrent
request becomes a function of the slowest one.

Fix: `await asyncio.to_thread(blocking_fn, ...)`, which moves the work to a
thread-pool worker and yields control back to the loop. Detection: hit
`/healthz` while a slow request is in flight and time it.

</details>

**Q3.** Why derive trace-level cost with a SQL `sum()` rather than storing it in a
`traces` table?

<details><summary>Answer</summary>

Because a stored total is **derived data that can drift from its source**. Two
writes per trace — the spans and the rollup — means two chances to disagree: a
crash between them, a span written after the rollup, a retry that double-counts.
Then you have two numbers and no way to know which is right.

Computing `sum(json_extract(attributes,'$.cost_usd'))` over the spans makes the
spans the single source of truth. The cost is a slightly more expensive read,
which is irrelevant for a dashboard querying 50 traces.

The general principle: **store facts, derive summaries.** Store the summary too
only when the read cost is genuinely prohibitive, and then treat it as a cache
with an invalidation story.

</details>

**Q4.** Your traces show `agent.step` at 413 s and all retrieval at 30 s. A
colleague proposes switching to a faster vector database. Respond with numbers.

<details><summary>Answer</summary>

Retrieval is 30 s of 443 s total — **under 7%**. Even an infinitely fast vector
database, taking retrieval to zero, cuts total wall clock by at most 7%, from
443 s to 413 s. Per contract that's 21.9 s down to about 20.4 s.

Meanwhile `agent.step` — the LLM calls — is 93%. The levers that matter are
fewer steps per contract (2.9 currently, so ~1 fewer step is a 30%+ saving), a
smaller or faster model, or fewer output tokens.

This is Amdahl's law applied to a pipeline: the achievable speedup from
optimising a component is bounded by the fraction of time it consumes. The
observability is what turns that from a slogan into an argument — without the
spans, "optimise retrieval" is a plausible and expensive mistake.

</details>

**Q5.** Harder. Design what you'd add to this observability setup to answer:
"has retrieval quality degraded in production over the last month?"

<details><summary>Answer</summary>

The hard part is that **production has no ground truth** — you don't know the
gold chunk for a live user query, so `recall@k` is unavailable. So you need
proxies, plus a way to get real labels.

**Proxy signals to record per request** (all cheap, all already nearly in place):
1. **Top-1 and top-5 retrieval scores**, and the **gap between rank 1 and 2.** A
   drifting distribution suggests the corpus or the query mix has changed. Must be
   tracked as a distribution, not a mean — and remember module 00: absolute cosine
   is corpus-relative, so what matters is *change*, not level.
2. **Validator fire rate** — the fraction of claims stripped for bad citations or
   quotes. This is a genuine quality signal requiring no labels, and a rise means
   the generator is degrading or retrieval is feeding worse context.
3. **Refusal rate** — `answer_found: false` frequency. A jump means either
   retrieval is missing evidence or the model got more conservative.
4. **Empty-result and zero-score rates** — module 08's all-zero BM25 case, and
   filtered searches returning fewer than `k`.

**Getting actual labels:**
5. **User feedback in the trace** — a thumbs-down attached to a `trace_id` makes
   every span for that request available for diagnosis. This is the highest-value
   addition, because it converts complaints into debuggable cases.
6. **Periodic golden-set replay.** Run the 148-question eval against the
   *production* index on a schedule and compare to `evals/baseline.json`. This is
   the only measurement that gives you real recall@k, and it catches index drift
   — a re-index with a different model, a partial upsert, a chunking change — that
   no proxy would.

**Making it actionable:**
7. **Version everything in the trace** — embedding model name, index build id,
   prompt version, git SHA. Without these, "quality dropped last Tuesday"
   cannot be tied to a change. This is the single most common gap.
8. **Alert on the proxies, diagnose with the golden set.** Proxies are sensitive
   but not specific; the replay is specific but periodic. Use the first to know
   *when* and the second to know *what*.

**The honest caveat:** proxy drift tells you something changed, not that quality
fell. A shift in what users ask about moves every proxy without any regression.
That ambiguity is exactly why the scheduled golden-set replay is the load-bearing
piece — a fixed question set is the only thing that holds the input constant so a
change in output means a change in the system.

</details>

---

## You've finished the course

Where each part landed:

| Part | What you should now be able to do |
|---|---|
| **A** Foundations | Explain embeddings and cosine from first principles; argue RAG vs fine-tuning; parse real documents without destroying offsets |
| **B** Chunking | Derive overlap cost; chunk on structure; **prove** your chunker didn't lose answers |
| **C** Retrieval | Derive BM25; explain why RRF fuses ranks; state the re-ranking ceiling and why it bounds everything |
| **D** Evaluation | Compute all four metrics by hand; build a golden set from annotations; validate a judge before quoting it |
| **E** Generation | Design a citation prompt that doesn't collide with content; explain why string matching cannot catch false inference |
| **F** Systems | Bound an agent with schemas and brakes; diagnose a wrong answer from a trace in four queries |

### The three themes that outlast the specifics

**1. Silent failures dominate.** Of the ~34 bugs found across this project, the
expensive ones raised no exception: the discarded preambles (16.6% of answers
unreachable), the truncated prompts (60% discarded), the inert validator, the
clamped sweep parameter, the missing `import sys`. Every one produced
plausible-looking output.

**2. Accounting checks find them.** Not code review, not watching it run —
**asking whether the totals reconcile.** Do stored characters equal source
characters? Are annotated answers reachable? Is `recall@50` unchanged by
re-ranking? Does 120 minus 2 equal 118? A log line that can contradict itself is
a free assertion.

**3. A number you tuned against is not a measurement.** 98% became 86% on
held-out data. Two of three Day-3 predictions were wrong. Three confident causal
stories failed five-minute experiments. The discipline of measuring rather than
reasoning is the whole difference — and the honest reporting of the results that
disappointed you is what makes the rest credible.

**Previous:** [17 — Agents](17-agents.md) · **Index:** [notes.md](../../notes.md)
