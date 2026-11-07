"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ClauseText } from "@/components/ClauseText";
import {
  Badge,
  buttonClass,
  Card,
  cx,
  Empty,
  ErrorBox,
  inputClass,
  PageHeader,
  prettyDocId,
  Spinner,
  Stat,
  type Tone,
} from "@/components/ui";
import {
  ApiError,
  getAudit,
  getChunk,
  startAudit,
  type AuditJob,
  type AuditReport,
  type Chunk,
  type Finding,
  type FindingStatus,
} from "@/lib/api";
import { useStored } from "@/lib/useStored";

// The API keeps jobs in memory; remembering ids lets a reload pick a running
// audit back up instead of orphaning it.
const JOBS_KEY = "cc.audit.jobs";
type JobRef = { id: string; policy: string; n: number; startedAt: number; etaS: number };

const PRESETS = [
  "Every agreement must include a cap on liability.",
  "Every agreement must specify which jurisdiction's law governs it.",
  "Agreements must not renew automatically without a notice period.",
  "Either party must be able to terminate for convenience with written notice.",
];

const STATUS: Record<FindingStatus, { tone: Tone; label: string; help: string }> = {
  violation: { tone: "bad", label: "Violation", help: "Addresses the topic but fails a requirement" },
  unclear: { tone: "warn", label: "Unclear", help: "Relevant language exists but is ambiguous" },
  silent: { tone: "neutral", label: "Silent", help: "The contract does not address the topic" },
  compliant: { tone: "ok", label: "Compliant", help: "Clearly satisfies every required element" },
};
const ORDER: FindingStatus[] = ["violation", "unclear", "silent", "compliant"];

export default function AuditView() {
  const [policy, setPolicy] = useState(PRESETS[0]);
  const [n, setN] = useState(12);
  const [jobs, setJobs] = useStored<JobRef[]>(JOBS_KEY, []);
  const [picked, setActiveId] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<unknown>(null);

  async function start(e: React.FormEvent) {
    e.preventDefault();
    setStarting(true);
    setError(null);
    try {
      const r = await startAudit(policy.trim(), n);
      const ref = { id: r.job_id, policy: policy.trim(), n, startedAt: Date.now(), etaS: r.eta_s };
      setJobs([ref, ...jobs].slice(0, 10));
      setActiveId(r.job_id);
    } catch (err) {
      setError(err);
    } finally {
      setStarting(false);
    }
  }

  const active = jobs.find((j) => j.id === picked) ?? jobs[0] ?? null;
  const activeId = active?.id;

  return (
    <>
      <PageHeader
        title="Compliance audit"
        sub="An agent plans what to look for, then audits each contract in a bounded loop. A finding of compliant or violation cannot be recorded without a real citation — the schema rejects it."
      />

      <Card className="p-4 md:p-5">
        <form onSubmit={start} className="space-y-4">
          <div>
            <label htmlFor="policy" className="mb-1.5 block text-sm font-medium">
              Policy
            </label>
            <textarea
              id="policy"
              value={policy}
              onChange={(e) => setPolicy(e.target.value)}
              rows={2}
              minLength={10}
              maxLength={500}
              className={cx(inputClass, "resize-y")}
            />
            <div className="mt-2 flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button
                  key={p}
                  type="button"
                  onClick={() => setPolicy(p)}
                  className={cx(
                    "rounded-full border px-3 py-1 text-xs",
                    p === policy ? "border-accent text-accent" : "border-line text-ink-2 hover:border-accent hover:text-accent",
                  )}
                >
                  {p}
                </button>
              ))}
            </div>
          </div>
          <div className="flex flex-col gap-4 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label htmlFor="n" className="mb-1.5 flex justify-between text-sm font-medium">
                Contracts to audit <span className="tabular-nums text-ink-2">{n}</span>
              </label>
              <input
                id="n"
                type="range"
                min={1}
                max={100}
                value={n}
                onChange={(e) => setN(Number(e.target.value))}
                className="w-full accent-[var(--accent)]"
              />
              <p className="mt-1 text-xs text-ink-3">
                Up to {fmtDuration(n * 22)} at ~22 s per contract with a local model; Groq is much faster. Runs
                in the background.
              </p>
            </div>
            <button type="submit" disabled={starting || policy.trim().length < 10} className={buttonClass.primary}>
              {starting && <Spinner />} Start audit
            </button>
          </div>
        </form>
      </Card>

      {error != null && (
        <div className="mt-4">
          <ErrorBox error={error} />
        </div>
      )}

      {jobs.length > 1 && (
        <div className="mt-6 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-ink-3">Audits this session:</span>
          {jobs.map((j) => (
            <button
              key={j.id}
              onClick={() => setActiveId(j.id)}
              className={cx(
                "max-w-64 truncate rounded-md border px-2 py-1",
                j.id === activeId ? "border-accent bg-accent-soft text-accent" : "border-line text-ink-2 hover:bg-surface-2",
              )}
              title={j.policy}
            >
              {j.policy}
            </button>
          ))}
        </div>
      )}

      <div className="mt-6">
        {active ? (
          <JobPanel key={active.id} job={active} onForget={() => {
            setJobs(jobs.filter((j) => j.id !== active.id));
            setActiveId(null);
          }} />
        ) : (
          <Empty>Start an audit to see the agent&apos;s plan and per-contract findings here.</Empty>
        )}
      </div>
    </>
  );
}

function fmtDuration(s: number) {
  if (s < 90) return `${Math.round(s)} s`;
  return `${Math.round(s / 60)} min`;
}

function JobPanel({ job, onForget }: { job: JobRef; onForget: () => void }) {
  const [state, setState] = useState<AuditJob | null>(null);
  const [lost, setLost] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const s = await getAudit(job.id);
        if (!alive) return;
        setState(s);
        setError(null);
        if (s.status === "running") timer = setTimeout(poll, 3000);
      } catch (e) {
        if (!alive) return;
        if (e instanceof ApiError && e.status === 404) setLost(true);
        else {
          setError(e);
          timer = setTimeout(poll, 5000);
        }
      }
    };
    poll();
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      alive = false;
      clearTimeout(timer);
      clearInterval(tick);
    };
  }, [job.id]);

  if (lost) {
    return (
      <Card className="p-5 text-sm">
        <p className="font-medium">This audit is no longer on the server.</p>
        <p className="mt-1 text-ink-2">Audit jobs live in API memory, so restarting the API clears them.</p>
        <button onClick={onForget} className={cx(buttonClass.secondary, "mt-3")}>
          Remove from list
        </button>
      </Card>
    );
  }

  if (!state || state.status === "running") {
    const elapsed = (now - job.startedAt) / 1000;
    const pct = Math.min(95, (elapsed / job.etaS) * 100);
    return (
      <Card className="p-5">
        <div className="flex items-center gap-2 text-sm font-medium">
          <Spinner className="text-accent" /> Auditing {job.n} contracts
        </div>
        <p className="mt-1 text-sm text-ink-2">{job.policy}</p>
        <div className="mt-4 h-1.5 overflow-hidden rounded-full bg-surface-2">
          <div className="h-full rounded-full bg-accent transition-all duration-1000" style={{ width: `${pct}%` }} />
        </div>
        <p className="mt-2 text-xs text-ink-3">
          {fmtDuration(elapsed)} elapsed · estimate {fmtDuration(job.etaS)} · progress is time-based, not reported
          by the agent. You can leave this page and come back.
        </p>
        {error != null && (
          <div className="mt-3">
            <ErrorBox error={error} />
          </div>
        )}
      </Card>
    );
  }

  if (state.status === "error") {
    return (
      <div className="space-y-3">
        <ErrorBox error={`Audit failed: ${state.error}`} />
        <button onClick={onForget} className={buttonClass.secondary}>
          Remove from list
        </button>
      </div>
    );
  }

  return <Report report={state.report} />;
}

function Report({ report }: { report: AuditReport }) {
  const [filter, setFilter] = useState<FindingStatus | "all">("all");
  const counts = useMemo(() => {
    const c = { violation: 0, unclear: 0, silent: 0, compliant: 0 } as Record<FindingStatus, number>;
    report.findings.forEach((f) => c[f.status]++);
    return c;
  }, [report]);
  const shown = report.findings
    .filter((f) => filter === "all" || f.status === filter)
    .sort((a, b) => ORDER.indexOf(a.status) - ORDER.indexOf(b.status));
  const aborted = Object.entries(report.aborted);
  const p = report.plan;

  return (
    <div className="space-y-6">
      <Card className="p-5">
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          <Stat label="Contracts examined" value={report.n_contracts_examined} />
          <Stat label="Findings recorded" value={report.findings.length} />
          <Stat label="Did not finish" value={aborted.length} hint="step / time cap" />
          <Stat label="Duration" value={fmtDuration(report.duration_s)} hint={report.total_cost_usd > 0 ? `$${report.total_cost_usd.toFixed(4)}` : undefined} />
        </div>
      </Card>

      <details className="group rounded-xl border border-line bg-surface">
        <summary className="cursor-pointer list-none px-5 py-3 text-sm font-semibold">
          <span className="mr-1 inline-block transition-transform group-open:rotate-90">›</span> Agent&apos;s plan
          <span className="ml-2 font-normal text-ink-3">{p.policy_restated}</span>
        </summary>
        <div className="grid gap-5 border-t border-line px-5 py-4 text-sm md:grid-cols-2">
          <PlanList title="Required elements" items={p.required_elements} />
          <PlanList title="Search queries" items={p.search_queries} />
          <PlanList title="Decoys — near-misses not to mistake for compliance" items={p.decoy_queries} />
          <div>
            <h3 className="mb-1 text-xs font-medium text-ink-3 uppercase">Compliance criteria</h3>
            <p className="text-ink-2">{p.compliance_criteria}</p>
          </div>
        </div>
      </details>

      <section>
        <div className="mb-3 flex flex-wrap gap-2">
          <FilterChip active={filter === "all"} onClick={() => setFilter("all")}>
            All {report.findings.length}
          </FilterChip>
          {ORDER.map((s) => (
            <FilterChip key={s} active={filter === s} onClick={() => setFilter(s)} title={STATUS[s].help}>
              <Badge tone={STATUS[s].tone}>{STATUS[s].label}</Badge> {counts[s]}
            </FilterChip>
          ))}
        </div>
        {shown.length ? (
          <div className="space-y-2">
            {shown.map((f, i) => (
              <FindingRow key={`${f.doc_id}-${i}`} f={f} />
            ))}
          </div>
        ) : (
          <Empty>No findings with this status.</Empty>
        )}
      </section>

      {aborted.length > 0 && (
        <section>
          <h2 className="mb-2 text-sm font-semibold">Contracts without a finding</h2>
          <Card className="divide-y divide-line text-sm">
            {aborted.map(([doc, why]) => (
              <div key={doc} className="flex items-center justify-between gap-3 px-4 py-2">
                <Link href={`/contracts/${encodeURIComponent(doc)}`} className="min-w-0 truncate hover:underline">
                  {prettyDocId(doc)}
                </Link>
                <Badge>{why.replace("_", " ")}</Badge>
              </div>
            ))}
          </Card>
        </section>
      )}
    </div>
  );
}

function PlanList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <h3 className="mb-1 text-xs font-medium text-ink-3 uppercase">{title}</h3>
      {items.length ? (
        <ul className="list-disc space-y-0.5 pl-4 text-ink-2">
          {items.map((x) => (
            <li key={x}>{x}</li>
          ))}
        </ul>
      ) : (
        <p className="text-ink-3">—</p>
      )}
    </div>
  );
}

function FilterChip({
  active,
  onClick,
  title,
  children,
}: {
  active: boolean;
  onClick: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      title={title}
      className={cx(
        "flex items-center gap-1.5 rounded-lg border px-2.5 py-1 text-xs tabular-nums",
        active ? "border-accent bg-accent-soft" : "border-line hover:bg-surface-2",
      )}
    >
      {children}
    </button>
  );
}

function FindingRow({ f }: { f: Finding }) {
  const [open, setOpen] = useState(false);
  const s = STATUS[f.status];
  return (
    <Card className="overflow-hidden">
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-start gap-3 px-4 py-3 text-left hover:bg-surface-2/50">
        <Badge tone={s.tone} className="mt-0.5">
          {s.label}
        </Badge>
        <div className="min-w-0 flex-1">
          <div className="truncate text-sm font-medium">{prettyDocId(f.doc_id)}</div>
          <p className={cx("mt-0.5 text-sm text-ink-2", !open && "line-clamp-2")}>{f.rationale}</p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1">
          {f.severity !== "none" && <span className="text-xs text-ink-3">{f.severity} severity</span>}
          <span className="text-xs text-ink-3">
            {f.citations.length} citation{f.citations.length === 1 ? "" : "s"} {open ? "▴" : "▾"}
          </span>
        </div>
      </button>
      {open && (
        <div className="space-y-3 border-t border-line bg-surface-2/40 px-4 py-3">
          {f.quote && (
            <blockquote className="border-l-2 border-line pl-3 font-serif text-sm text-ink-2 italic">“{f.quote}”</blockquote>
          )}
          {f.citations.map((cid) => (
            <CitedChunk key={cid} chunkId={cid} quote={f.quote} />
          ))}
          {!f.citations.length && <p className="text-xs text-ink-3">No citation — allowed only for silent / unclear.</p>}
          <Link href={`/contracts/${encodeURIComponent(f.doc_id)}`} className="inline-block text-xs text-accent hover:underline">
            Open full contract →
          </Link>
        </div>
      )}
    </Card>
  );
}

function CitedChunk({ chunkId, quote }: { chunkId: string; quote: string | null }) {
  const [c, setC] = useState<Chunk | null>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => {
    getChunk(chunkId).then(setC, setErr);
  }, [chunkId]);
  if (err) return <ErrorBox error={err} />;
  if (!c) return <div className="text-xs text-ink-3"><Spinner className="size-3" /> Loading {chunkId}</div>;
  return (
    <div className="rounded-lg border border-line bg-surface">
      <div className="border-b border-line px-3 py-1.5 text-xs">
        <span className="font-medium">{c.section_heading || "Clause"}</span>
        <span className="ml-2 font-mono text-ink-3">{chunkId.split("::").slice(1).join("::")}</span>
      </div>
      <div className="max-h-72 overflow-y-auto px-3 py-2">
        <ClauseText text={c.text} quote={quote} />
      </div>
    </div>
  );
}
