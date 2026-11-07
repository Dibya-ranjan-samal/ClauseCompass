import type { ReactNode } from "react";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div className={cx("rounded-xl border border-line bg-surface", className)}>{children}</div>
  );
}

export function PageHeader({ title, sub, right }: { title: string; sub?: ReactNode; right?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">{title}</h1>
        {sub && <p className="mt-1 max-w-2xl text-sm text-ink-2">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

const tones = {
  neutral: "bg-surface-2 text-ink-2",
  accent: "bg-accent-soft text-accent",
  ok: "bg-ok-soft text-ok",
  bad: "bg-bad-soft text-bad",
  warn: "bg-warn-soft text-warn",
};
export type Tone = keyof typeof tones;

export function Badge({ tone = "neutral", children, className }: { tone?: Tone; children: ReactNode; className?: string }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-xs font-medium whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

export function Spinner({ className }: { className?: string }) {
  return (
    <span
      aria-hidden
      className={cx(
        "inline-block size-4 animate-spin rounded-full border-2 border-current border-r-transparent",
        className,
      )}
    />
  );
}

export function ErrorBox({ error }: { error: unknown }) {
  const msg = error instanceof Error ? error.message : String(error);
  return (
    <div role="alert" className="rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad">
      {msg}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-line px-6 py-10 text-center text-sm text-ink-3">
      {children}
    </div>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="min-w-0">
      <div className="text-xs text-ink-3">{label}</div>
      <div className="mt-0.5 text-lg font-semibold tabular-nums">{value}</div>
      {hint && <div className="text-xs text-ink-3">{hint}</div>}
    </div>
  );
}

export const buttonClass = {
  primary:
    "inline-flex items-center justify-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-50 dark:text-bg",
  secondary:
    "inline-flex items-center justify-center gap-2 rounded-lg border border-line bg-surface px-3 py-1.5 text-sm text-ink-2 hover:bg-surface-2 hover:text-ink disabled:opacity-50",
};

export const inputClass =
  "w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/20";

export function fmtMs(ms: number) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`;
}

/** "cuad_ACME_..._Supply_Agreement" -> "ACME ... Supply Agreement" for display. */
export function prettyDocId(docId: string) {
  return docId.replace(/^cuad_/, "").replace(/[_]+/g, " ");
}

export function contractTitle(c: { doc_id: string; parties?: string[] | null }) {
  const p = (c.parties ?? []).filter(Boolean);
  if (p.length) return p.slice(0, 2).join(" · ") + (p.length > 2 ? ` +${p.length - 2}` : "");
  return prettyDocId(c.doc_id);
}
