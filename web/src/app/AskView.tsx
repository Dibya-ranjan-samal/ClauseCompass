"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ClauseText } from "@/components/ClauseText";
import { ContractPicker } from "@/components/ContractPicker";
import {
  Badge,
  buttonClass,
  Card,
  cx,
  Empty,
  ErrorBox,
  fmtMs,
  inputClass,
  PageHeader,
  prettyDocId,
  Spinner,
} from "@/components/ui";
import { ask, describeLlm, type AskResponse, type Claim, type StrippedClaim } from "@/lib/api";
import { useStored } from "@/lib/useStored";

type Entry = { id: string; question: string; docId: string | null; at: number; res: AskResponse };

const HISTORY_KEY = "cc.ask.history";
const EXAMPLES = [
  "What notice period is required to terminate this agreement?",
  "Is there a cap on liability, and what is it?",
  "Which state's law governs this agreement?",
  "Can either party assign the agreement without consent?",
  "Is there a non-compete or exclusivity obligation?",
];

const REASONS: Record<string, string> = {
  citation_not_retrieved: "Cited an excerpt that was never retrieved",
  quote_unverified: "Quote does not appear in the cited excerpt",
};

export default function AskView() {
  const params = useSearchParams();
  const [question, setQuestion] = useState("");
  const docParam = params.get("doc");
  const [docId, setDocId] = useState<string | null>(docParam);
  // "Ask about this contract" links set ?doc=; the page may already be mounted
  // (kept alive between navigations), so follow the param when it changes.
  const [prevDocParam, setPrevDocParam] = useState(docParam);
  if (docParam !== prevDocParam) {
    setPrevDocParam(docParam);
    if (docParam) setDocId(docParam);
  }
  const [pending, setPending] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<unknown>(null);
  const [current, setCurrent] = useState<Entry | null>(null);
  const [history, setHistory] = useStored<Entry[]>(HISTORY_KEY, []);
  const [focus, setFocus] = useState<{ label: string; quote: string | null } | null>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!pending) return;
    const t0 = Date.now();
    const id = setInterval(() => setElapsed(Date.now() - t0), 200);
    return () => clearInterval(id);
  }, [pending]);

  async function submit(e?: React.FormEvent) {
    e?.preventDefault();
    const q = question.trim();
    if (q.length < 3 || pending) return;
    setPending(true);
    setElapsed(0);
    setError(null);
    try {
      const res = await ask(q, docId);
      const entry = { id: crypto.randomUUID(), question: q, docId, at: Date.now(), res };
      setCurrent(entry);
      setFocus(res.claims[0] ? { label: res.claims[0].citations[0], quote: res.claims[0].quote } : null);
      setHistory([entry, ...history].slice(0, 20));
    } catch (err) {
      setError(err);
    } finally {
      setPending(false);
    }
  }

  function reopen(entry: Entry) {
    setCurrent(entry);
    setQuestion(entry.question);
    setDocId(entry.docId);
    const c = entry.res.claims[0];
    setFocus(c ? { label: c.citations[0], quote: c.quote } : null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  return (
    <>
      <PageHeader
        title="Ask the contracts"
        sub="An LLM drafts the answer, then every claim is checked in code: its citation must point at a retrieved excerpt and its quote must appear there verbatim. Claims that fail are removed and shown separately."
      />

      <Card className="p-4 md:p-5">
        <form onSubmit={submit} className="space-y-3">
          <textarea
            ref={inputRef}
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
            }}
            rows={2}
            maxLength={500}
            placeholder="e.g. What notice period is required to terminate this agreement?"
            className={cx(inputClass, "resize-y text-base")}
          />
          <div className="flex flex-col gap-3 md:flex-row md:items-start">
            <div className="flex-1">
              <ContractPicker value={docId} onChange={setDocId} />
              <p className="mt-1.5 text-xs text-ink-3">
                {docId
                  ? "Scoped to one contract: hybrid retrieval + cross-encoder rerank (recall@5 0.76). Slower — around 10 s of retrieval before generation."
                  : "Whole corpus: dense retrieval (recall@5 0.36). Pick a contract for much better recall."}
              </p>
            </div>
            <button type="submit" disabled={pending || question.trim().length < 3} className={buttonClass.primary}>
              {pending ? <Spinner /> : null}
              {pending ? `Working… ${(elapsed / 1000).toFixed(0)} s` : "Ask"}
              {!pending && <kbd className="hidden text-xs opacity-70 md:inline">⌘↵</kbd>}
            </button>
          </div>
          {!current && !pending && (
            <div className="flex flex-wrap gap-2 pt-1">
              {EXAMPLES.map((ex) => (
                <button
                  key={ex}
                  type="button"
                  onClick={() => {
                    setQuestion(ex);
                    inputRef.current?.focus();
                  }}
                  className="rounded-full border border-line px-3 py-1 text-left text-xs text-ink-2 hover:border-accent hover:text-accent"
                >
                  {ex}
                </button>
              ))}
            </div>
          )}
        </form>
      </Card>

      <div className="mt-6 space-y-6">
        {error != null && <ErrorBox error={error} />}
        {pending && <PendingSteps docScoped={!!docId} elapsed={elapsed} />}
        {current && !pending && (
          <Result entry={current} focus={focus} setFocus={setFocus} />
        )}
        {history.length > 0 && (
          <History
            items={history}
            activeId={current?.id}
            onOpen={reopen}
            onClear={() => setHistory([])}
          />
        )}
      </div>
    </>
  );
}

function PendingSteps({ docScoped, elapsed }: { docScoped: boolean; elapsed: number }) {
  const steps = [
    docScoped ? "Retrieve: dense + BM25 → RRF → cross-encoder" : "Retrieve: dense search over 18k chunks",
    "Generate: the LLM writes a cited JSON answer",
    "Verify: check every citation and quote against the excerpts",
  ];
  return (
    <Card className="p-5">
      <div className="mb-3 flex items-center gap-2 text-sm font-medium">
        <Spinner className="text-accent" /> Running the pipeline · {(elapsed / 1000).toFixed(1)} s
      </div>
      <ol className="space-y-1.5 text-sm text-ink-2">
        {steps.map((s, i) => (
          <li key={s} className="flex gap-2">
            <span className="text-ink-3">{i + 1}.</span>
            {s}
          </li>
        ))}
      </ol>
      <p className="mt-3 text-xs text-ink-3">
        Groq usually answers in a few seconds; a local model takes 10–40 s on a laptop.
      </p>
    </Card>
  );
}

function Result({
  entry,
  focus,
  setFocus,
}: {
  entry: Entry;
  focus: { label: string; quote: string | null } | null;
  setFocus: (f: { label: string; quote: string | null } | null) => void;
}) {
  const r = entry.res;
  const v = r.validation;
  const confTone = r.confidence === "high" ? "ok" : r.confidence === "medium" ? "warn" : "neutral";

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="min-w-0 space-y-6">
        <Card className="p-5">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            {r.answer_found ? <Badge tone="ok">Answered</Badge> : <Badge tone="warn">Not found</Badge>}
            <Badge tone={confTone}>confidence: {r.confidence}</Badge>
            <span className="text-xs text-ink-3">
              {r.llm && <span title={r.llm}>{describeLlm(r.llm).provider} · </span>}
              {fmtMs(r.latency_ms)} generation
              {r.cost_usd > 0 && ` · $${r.cost_usd.toFixed(4)}`} · {r.retriever}
            </span>
          </div>
          <p className="text-sm text-ink-3">{entry.question}</p>
          <p className="mt-2 text-[1.02rem] leading-relaxed">{r.answer}</p>
          {!r.answer_found && (
            <p className="mt-3 text-xs text-ink-3">
              The system refuses rather than guessing when no claim survives verification.
              {!entry.docId && " Try scoping to a specific contract."}
            </p>
          )}

          <div className="mt-4 grid grid-cols-2 gap-3 border-t border-line pt-4 text-xs sm:grid-cols-4">
            <Mini label="Claims proposed" value={v.n_claims_raw} />
            <Mini label="Verified" value={v.n_claims_kept} tone="ok" />
            <Mini label="Bad citation" value={v.n_bad_citation} tone={v.n_bad_citation ? "bad" : undefined} />
            <Mini label="Bad quote" value={v.n_bad_quote} tone={v.n_bad_quote ? "bad" : undefined} />
          </div>
          {(v.parse_failed || v.truncated) && (
            <p className="mt-3 text-xs text-bad">
              {v.parse_failed ? "The model's output could not be parsed as JSON." : "The model's output was truncated."}
            </p>
          )}
        </Card>

        {r.claims.length > 0 && (
          <section>
            <h2 className="mb-2 text-sm font-semibold">Verified claims</h2>
            <div className="space-y-2">
              {r.claims.map((c, i) => (
                <ClaimCard key={i} claim={c} focus={focus} onCite={setFocus} />
              ))}
            </div>
          </section>
        )}

        {r.stripped_claims.length > 0 && (
          <section>
            <h2 className="mb-1 text-sm font-semibold">Removed by the validator ({r.stripped_claims.length})</h2>
            <p className="mb-2 text-xs text-ink-3">
              The model asserted these, but they could not be verified against the source text, so
              they are not part of the answer.
            </p>
            <div className="space-y-2">
              {r.stripped_claims.map((s, i) => (
                <StrippedCard key={i} claim={s} onCite={setFocus} />
              ))}
            </div>
          </section>
        )}
      </div>

      <Sources entry={entry} focus={focus} setFocus={setFocus} />
    </div>
  );
}

function Mini({ label, value, tone }: { label: string; value: number; tone?: "ok" | "bad" }) {
  return (
    <div>
      <div className="text-ink-3">{label}</div>
      <div className={cx("text-base font-semibold tabular-nums", tone === "ok" && "text-ok", tone === "bad" && "text-bad")}>
        {value}
      </div>
    </div>
  );
}

function CiteChip({
  label,
  active,
  onClick,
  bad,
}: {
  label: string;
  active?: boolean;
  onClick?: () => void;
  bad?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cx(
        "rounded px-1.5 py-0.5 font-mono text-xs transition-colors",
        bad
          ? "bg-bad-soft text-bad line-through"
          : active
            ? "bg-accent text-white dark:text-bg"
            : "bg-accent-soft text-accent hover:bg-accent hover:text-white dark:hover:text-bg",
      )}
    >
      [{label}]
    </button>
  );
}

function ClaimCard({
  claim,
  focus,
  onCite,
}: {
  claim: Claim;
  focus: { label: string; quote: string | null } | null;
  onCite: (f: { label: string; quote: string | null }) => void;
}) {
  const isActive = focus?.quote === claim.quote && claim.citations.includes(focus?.label ?? "");
  return (
    <div
      className={cx(
        "rounded-lg border bg-surface p-3 transition-colors",
        isActive ? "border-accent" : "border-line",
      )}
    >
      <div className="flex gap-2">
        <span className="mt-0.5 text-ok" aria-label="verified">✓</span>
        <div className="min-w-0 flex-1">
          <p className="text-sm">
            {claim.statement}{" "}
            {claim.citations.map((l) => (
              <CiteChip
                key={l}
                label={l}
                active={isActive && focus?.label === l}
                onClick={() => onCite({ label: l, quote: claim.quote })}
              />
            ))}
          </p>
          {claim.quote && (
            <blockquote className="mt-2 border-l-2 border-line pl-3 font-serif text-sm text-ink-2 italic">
              “{claim.quote}”
            </blockquote>
          )}
        </div>
      </div>
    </div>
  );
}

function StrippedCard({
  claim,
  onCite,
}: {
  claim: StrippedClaim;
  onCite: (f: { label: string; quote: string | null }) => void;
}) {
  const unknown = new Set(claim.unknown ?? []);
  return (
    <div className="rounded-lg border border-bad/25 bg-bad-soft/40 p-3">
      <div className="mb-1 flex flex-wrap items-center gap-2">
        <Badge tone="bad">{REASONS[claim.reason] ?? claim.reason}</Badge>
        {claim.match_ratio != null && (
          <span className="text-xs text-ink-3">best match {Math.round(claim.match_ratio * 100)}% (needs 95%)</span>
        )}
      </div>
      <p className="text-sm text-ink-2">
        {claim.statement}{" "}
        {(claim.citations ?? []).map((l) => (
          <CiteChip
            key={l}
            label={l}
            bad={unknown.has(String(l))}
            onClick={unknown.has(String(l)) ? undefined : () => onCite({ label: String(l), quote: null })}
          />
        ))}
      </p>
      {claim.quote && (
        <blockquote className="mt-2 border-l-2 border-bad/30 pl-3 font-serif text-sm text-ink-3 italic">
          “{claim.quote}”
        </blockquote>
      )}
    </div>
  );
}

function Sources({
  entry,
  focus,
  setFocus,
}: {
  entry: Entry;
  focus: { label: string; quote: string | null } | null;
  setFocus: (f: { label: string; quote: string | null } | null) => void;
}) {
  const sources = entry.res.sources ?? [];
  const cited = new Set(entry.res.claims.flatMap((c) => c.citations));
  const active = sources.find((s) => s.label === focus?.label) ?? null;

  if (!sources.length) {
    return <Empty>No excerpts were returned for this question.</Empty>;
  }

  return (
    <aside className="min-w-0 lg:sticky lg:top-6 lg:self-start">
      <h2 className="mb-2 text-sm font-semibold">Retrieved excerpts</h2>
      <div className="mb-3 flex flex-wrap gap-1.5">
        {sources.map((s) => (
          <button
            key={s.label}
            type="button"
            onClick={() => setFocus({ label: s.label, quote: focus?.label === s.label ? focus.quote : null })}
            className={cx(
              "rounded-md border px-2 py-1 text-xs transition-colors",
              active?.label === s.label
                ? "border-accent bg-accent-soft text-accent"
                : "border-line text-ink-2 hover:bg-surface-2",
            )}
          >
            <span className="font-mono">[{s.label}]</span>
            {cited.has(s.label) && <span className="ml-1 text-ok">●</span>}
          </button>
        ))}
      </div>
      {active ? (
        <Card className="overflow-hidden">
          <div className="border-b border-line bg-surface-2 px-4 py-2.5">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate text-sm font-medium">
                [{active.label}] {active.section_heading || "Clause"}
              </span>
              <span className="shrink-0 font-mono text-xs text-ink-3">score {active.score.toFixed(3)}</span>
            </div>
            <Link
              href={`/contracts/${encodeURIComponent(active.doc_id)}#${encodeURIComponent(active.chunk_id)}`}
              className="block truncate text-xs text-accent hover:underline"
            >
              {prettyDocId(active.doc_id)} →
            </Link>
          </div>
          <div className="max-h-[60vh] overflow-y-auto px-4 py-3">
            <ClauseText text={active.text} quote={focus?.quote} />
          </div>
        </Card>
      ) : (
        <Empty>Click a citation or an excerpt label to read the source text.</Empty>
      )}
      <p className="mt-2 text-xs text-ink-3">
        <span className="text-ok">●</span> cited by a verified claim. Highlighted text is the exact quote the
        validator matched.
      </p>
    </aside>
  );
}

function History({
  items,
  activeId,
  onOpen,
  onClear,
}: {
  items: Entry[];
  activeId?: string;
  onOpen: (e: Entry) => void;
  onClear: () => void;
}) {
  return (
    <section>
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-sm font-semibold">Recent questions</h2>
        <button onClick={onClear} className="text-xs text-ink-3 hover:text-bad">
          Clear
        </button>
      </div>
      <Card className="divide-y divide-line">
        {items.map((e) => (
          <button
            key={e.id}
            onClick={() => onOpen(e)}
            className={cx(
              "flex w-full items-center gap-3 px-4 py-2.5 text-left text-sm hover:bg-surface-2",
              e.id === activeId && "bg-surface-2",
            )}
          >
            {e.res.answer_found ? (
              <span className="text-ok" aria-label="answered">✓</span>
            ) : (
              <span className="text-warn" aria-label="not found">–</span>
            )}
            <span className="min-w-0 flex-1 truncate">{e.question}</span>
            <span className="hidden max-w-[40%] truncate text-xs text-ink-3 sm:block">
              {e.docId ? prettyDocId(e.docId) : "whole corpus"}
            </span>
          </button>
        ))}
      </Card>
    </section>
  );
}
