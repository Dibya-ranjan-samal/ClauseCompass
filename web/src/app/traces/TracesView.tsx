"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { Badge, buttonClass, Card, cx, Empty, ErrorBox, fmtMs, PageHeader, Spinner } from "@/components/ui";
import { getTrace, getTraces, type Span, type TraceSummary } from "@/lib/api";

export default function TracesView() {
  const router = useRouter();
  const params = useSearchParams();
  const selected = params.get("id");
  const [list, setList] = useState<TraceSummary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [reloading, setReloading] = useState(true);

  const fetchList = () =>
    getTraces(100)
      .then((r) => {
        setList(r);
        setError(null);
      }, setError)
      .finally(() => setReloading(false));
  useEffect(() => {
    fetchList();
  }, []);
  const load = () => {
    setReloading(true);
    fetchList();
  };

  return (
    <>
      <PageHeader
        title="Traces"
        sub="Every audit and eval run is recorded as a tree of spans in SQLite. Use this to see where time went and what each agent step decided."
        right={
          <button onClick={load} disabled={reloading} className={buttonClass.secondary}>
            {reloading && <Spinner className="size-3" />} Refresh
          </button>
        }
      />
      {error != null && <ErrorBox error={error} />}
      <div className="grid gap-6 lg:grid-cols-[22rem_minmax(0,1fr)]">
        <div className="min-w-0">
          {!list && !error && (
            <div className="flex items-center gap-2 text-sm text-ink-3">
              <Spinner /> Loading…
            </div>
          )}
          {list && !list.length && <Empty>No traces yet. Run an audit to create one.</Empty>}
          {list && list.length > 0 && (
            <Card className="max-h-[75vh] divide-y divide-line overflow-y-auto">
              {list.map((t) => (
                <button
                  key={t.trace_id}
                  onClick={() => router.replace(`/traces?id=${t.trace_id}`, { scroll: false })}
                  className={cx(
                    "block w-full px-4 py-2.5 text-left hover:bg-surface-2/60",
                    selected === t.trace_id && "bg-surface-2",
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-mono text-xs font-medium">{t.name}</span>
                    <span className="text-xs text-ink-3 tabular-nums">{fmtMs(t.duration_ms)}</span>
                  </div>
                  {t.question && <div className="mt-0.5 truncate text-sm text-ink-2">{t.question}</div>}
                  <div className="mt-0.5 flex items-center gap-2 text-xs text-ink-3">
                    {new Date(t.started_at * 1000 - t.duration_ms).toLocaleString()}
                    <span>· {t.n_spans} spans</span>
                    {t.status !== "ok" && <Badge tone="bad">{t.status}</Badge>}
                  </div>
                </button>
              ))}
            </Card>
          )}
        </div>
        <div className="min-w-0">
          {selected ? <TraceDetail key={selected} id={selected} /> : list?.length ? <Empty>Select a trace.</Empty> : null}
        </div>
      </div>
    </>
  );
}

type Node = Span & { depth: number };

function TraceDetail({ id }: { id: string }) {
  const [spans, setSpans] = useState<Span[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [open, setOpen] = useState<string | null>(null);

  useEffect(() => {
    getTrace(id).then(setSpans, setError);
  }, [id]);

  // Depth-first order so children sit under their parent in the waterfall.
  const { rows, t0, total, byName } = useMemo(() => {
    if (!spans) return { rows: [] as Node[], t0: 0, total: 1, byName: [] as [string, number][] };
    const kids = new Map<string | null, Span[]>();
    spans.forEach((s) => kids.set(s.parent_id, [...(kids.get(s.parent_id) ?? []), s]));
    kids.forEach((v) => v.sort((a, b) => a.start_ms - b.start_ms));
    const ids = new Set(spans.map((s) => s.span_id));
    const roots = spans.filter((s) => !s.parent_id || !ids.has(s.parent_id));
    const out: Node[] = [];
    const walk = (s: Span, depth: number) => {
      out.push({ ...s, depth });
      (kids.get(s.span_id) ?? []).forEach((k) => walk(k, depth + 1));
    };
    roots.sort((a, b) => a.start_ms - b.start_ms).forEach((r) => walk(r, 0));
    const t0 = Math.min(...spans.map((s) => s.start_ms));
    const end = Math.max(...spans.map((s) => s.start_ms + s.duration_ms));
    // Self time (span minus its children), so nested spans are not double
    // counted but an agent.step's own LLM call is still visible.
    const agg = new Map<string, number>();
    spans.forEach((s) => {
      const own = s.duration_ms - (kids.get(s.span_id) ?? []).reduce((a, k) => a + k.duration_ms, 0);
      if (own > 0) agg.set(s.name, (agg.get(s.name) ?? 0) + own);
    });
    return { rows: out, t0, total: Math.max(end - t0, 1), byName: [...agg.entries()].sort((a, b) => b[1] - a[1]) };
  }, [spans]);

  if (error) return <ErrorBox error={error} />;
  if (!spans)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> Loading trace…
      </div>
    );

  const selfTotal = byName.reduce((a, [, v]) => a + v, 0) || 1;

  return (
    <div className="space-y-5">
      <Card className="p-4">
        <h2 className="mb-3 text-sm font-semibold">Time by stage <span className="font-normal text-ink-3">· self time, excluding child spans</span></h2>
        <div className="space-y-1.5">
          {byName.map(([name, ms]) => (
            <div key={name} className="grid grid-cols-[10rem_minmax(0,1fr)_4.5rem] items-center gap-3 text-xs">
              <span className="truncate font-mono">{name}</span>
              <div className="h-2 rounded-full bg-surface-2">
                <div className="h-full rounded-full bg-accent" style={{ width: `${(ms / selfTotal) * 100}%` }} />
              </div>
              <span className="text-right text-ink-3 tabular-nums">{fmtMs(ms)}</span>
            </div>
          ))}
        </div>
      </Card>

      <Card className="overflow-hidden">
        <div className="border-b border-line px-4 py-2.5 text-sm font-semibold">
          Waterfall <span className="font-normal text-ink-3">· {rows.length} spans · {fmtMs(total)}</span>
        </div>
        <div className="max-h-[60vh] overflow-y-auto">
          {rows.map((s) => {
            const left = ((s.start_ms - t0) / total) * 100;
            const width = Math.max((s.duration_ms / total) * 100, 0.4);
            const isOpen = open === s.span_id;
            return (
              <div key={s.span_id} className="border-b border-line/60 last:border-0">
                <button
                  onClick={() => setOpen(isOpen ? null : s.span_id)}
                  className={cx("grid w-full grid-cols-[minmax(0,14rem)_minmax(0,1fr)_4.5rem] items-center gap-3 px-4 py-1.5 text-left text-xs hover:bg-surface-2/60", isOpen && "bg-surface-2/60")}
                >
                  <span className="truncate font-mono" style={{ paddingLeft: s.depth * 12 }}>
                    {s.status !== "ok" && <span className="text-bad">● </span>}
                    {s.name}
                    {typeof s.attributes.tool === "string" && <span className="text-ink-3"> → {s.attributes.tool}</span>}
                  </span>
                  <div className="relative h-3">
                    <div
                      className={cx("absolute top-0 h-full rounded-sm", s.status === "ok" ? "bg-accent/70" : "bg-bad")}
                      style={{ left: `${left}%`, width: `${width}%` }}
                    />
                  </div>
                  <span className="text-right text-ink-3 tabular-nums">{fmtMs(s.duration_ms)}</span>
                </button>
                {isOpen && (
                  <pre className="overflow-x-auto bg-surface-2/40 px-4 py-2 font-mono text-xs text-ink-2">
                    {JSON.stringify(s.attributes, null, 2)}
                  </pre>
                )}
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
