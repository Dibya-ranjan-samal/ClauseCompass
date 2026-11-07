"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { Badge, Card, cx, Empty, ErrorBox, inputClass, PageHeader, prettyDocId, Spinner } from "@/components/ui";
import { getContracts, type ContractSummary } from "@/lib/api";

const PAGE = 40;

export default function ContractsView() {
  const [all, setAll] = useState<ContractSummary[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [q, setQ] = useState("");
  const [type, setType] = useState("");
  const [source, setSource] = useState("");
  const [limit, setLimit] = useState(PAGE);

  useEffect(() => {
    getContracts().then(setAll, setError);
  }, []);

  const types = useMemo(() => {
    const c = new Map<string, number>();
    all?.forEach((x) => x.contract_type && c.set(x.contract_type, (c.get(x.contract_type) ?? 0) + 1));
    return [...c.entries()].sort((a, b) => b[1] - a[1]);
  }, [all]);

  const rows = useMemo(() => {
    const t = q.trim().toLowerCase();
    return (all ?? []).filter(
      (c) =>
        (!type || c.contract_type === type) &&
        (!source || c.source === source) &&
        (!t ||
          c.doc_id.toLowerCase().includes(t) ||
          (c.parties ?? []).some((p) => p.toLowerCase().includes(t)) ||
          (c.governing_law ?? "").toLowerCase().includes(t)),
    );
  }, [all, q, type, source]);

  return (
    <>
      <PageHeader
        title="Contracts"
        sub="530 agreements from SEC EDGAR and the CUAD dataset, split into clause-level chunks. Metadata was extracted by a local model and may be incomplete."
      />

      <div className="mb-4 flex flex-col gap-2 sm:flex-row">
        <input
          value={q}
          onChange={(e) => (setQ(e.target.value), setLimit(PAGE))}
          placeholder="Search party, filename or governing law…"
          className={cx(inputClass, "sm:flex-1")}
        />
        <select value={type} onChange={(e) => (setType(e.target.value), setLimit(PAGE))} className={cx(inputClass, "sm:w-52")}>
          <option value="">All types</option>
          {types.map(([t, n]) => (
            <option key={t} value={t}>
              {t} ({n})
            </option>
          ))}
        </select>
        <select value={source} onChange={(e) => (setSource(e.target.value), setLimit(PAGE))} className={cx(inputClass, "sm:w-36")}>
          <option value="">All sources</option>
          <option value="cuad">CUAD</option>
          <option value="edgar">EDGAR</option>
        </select>
      </div>

      {error != null && <ErrorBox error={error} />}
      {!all && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-3">
          <Spinner /> Loading contracts…
        </div>
      )}

      {all && (
        <>
          <p className="mb-2 text-xs text-ink-3">
            {rows.length} of {all.length} contracts
          </p>
          {rows.length ? (
            <Card className="divide-y divide-line">
              {rows.slice(0, limit).map((c) => (
                <Link
                  key={c.doc_id}
                  href={`/contracts/${encodeURIComponent(c.doc_id)}`}
                  className="grid gap-1 px-4 py-3 hover:bg-surface-2/60 md:grid-cols-[minmax(0,1fr)_auto] md:items-center md:gap-4"
                >
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium">
                      {(c.parties ?? []).length ? c.parties!.join(" · ") : prettyDocId(c.doc_id)}
                    </div>
                    <div className="truncate font-mono text-xs text-ink-3">{c.doc_id}</div>
                  </div>
                  <div className="flex flex-wrap items-center gap-1.5 md:justify-end">
                    {c.contract_type && c.contract_type !== "Unknown" && <Badge tone="accent">{c.contract_type}</Badge>}
                    {c.governing_law && <Badge>{c.governing_law} law</Badge>}
                    {c.effective_date && <Badge>{c.effective_date}</Badge>}
                    <span className="text-xs text-ink-3 tabular-nums">{c.n_chunks} chunks</span>
                  </div>
                </Link>
              ))}
            </Card>
          ) : (
            <Empty>No contracts match.</Empty>
          )}
          {rows.length > limit && (
            <button
              onClick={() => setLimit((l) => l + PAGE)}
              className="mt-3 w-full rounded-lg border border-line py-2 text-sm text-ink-2 hover:bg-surface-2"
            >
              Show more ({rows.length - limit} remaining)
            </button>
          )}
        </>
      )}
    </>
  );
}
