"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { ClauseText } from "@/components/ClauseText";
import { Badge, buttonClass, cx, ErrorBox, inputClass, prettyDocId, Spinner } from "@/components/ui";
import { getContract, type ContractDetail } from "@/lib/api";

export default function ContractView() {
  const { docId: raw } = useParams<{ docId: string }>();
  const docId = decodeURIComponent(raw);
  const [c, setC] = useState<ContractDetail | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [q, setQ] = useState("");
  const [target, setTarget] = useState<string | null>(null);

  useEffect(() => {
    getContract(docId).then(setC, setError);
  }, [docId]);

  // Links from Ask / Audit carry the chunk_id as the hash: jump to it.
  useEffect(() => {
    if (!c) return;
    // The browser's own jump-to-hash ran before the text loaded, and CSS
    // :target never fires for client-side (pushState) navigation; do both here.
    const h = decodeURIComponent(window.location.hash.slice(1));
    if (!h) return;
    const raf = requestAnimationFrame(() => {
      setTarget(h);
      document.getElementById(h)?.scrollIntoView({ block: "start" });
    });
    return () => cancelAnimationFrame(raf);
  }, [c]);

  const chunks = useMemo(() => {
    const t = q.trim().toLowerCase();
    if (!c) return [];
    return t ? c.chunks.filter((x) => x.text.toLowerCase().includes(t) || (x.section_heading ?? "").toLowerCase().includes(t)) : c.chunks;
  }, [c, q]);

  if (error) return <ErrorBox error={error} />;
  if (!c)
    return (
      <div className="flex items-center gap-2 text-sm text-ink-3">
        <Spinner /> Loading contract…
      </div>
    );

  const outline = c.chunks.filter((x) => x.section_heading && (x.part ?? 0) === 0);

  return (
    <>
      <Link href="/contracts" className="text-xs text-ink-3 hover:text-ink">
        ← All contracts
      </Link>
      <div className="mt-2 mb-6 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold tracking-tight">
            {(c.parties ?? []).length ? c.parties!.join(" · ") : prettyDocId(c.doc_id)}
          </h1>
          <p className="mt-1 font-mono text-xs break-all text-ink-3">{c.doc_id}</p>
          <div className="mt-3 flex flex-wrap gap-1.5">
            {c.contract_type && c.contract_type !== "Unknown" && <Badge tone="accent">{c.contract_type}</Badge>}
            {c.governing_law && <Badge>{c.governing_law} law</Badge>}
            {c.effective_date && <Badge>Effective {c.effective_date}</Badge>}
            {c.source && <Badge>{c.source.toUpperCase()}</Badge>}
            <Badge>{c.n_chunks} chunks</Badge>
          </div>
        </div>
        <Link href={`/?doc=${encodeURIComponent(c.doc_id)}`} className={buttonClass.primary}>
          Ask about this contract
        </Link>
      </div>

      <div className="grid gap-6 lg:grid-cols-[16rem_minmax(0,1fr)]">
        <aside className="space-y-5 lg:sticky lg:top-6 lg:max-h-[calc(100vh-3rem)] lg:self-start lg:overflow-y-auto">
          {outline.length > 0 && (
            <div>
              <h2 className="mb-2 text-xs font-medium text-ink-3 uppercase">Sections</h2>
              <ul className="space-y-0.5 text-sm">
                {outline.map((x) => (
                  <li key={x.chunk_id}>
                    <a href={`#${encodeURIComponent(x.chunk_id)}`} onClick={() => setTarget(x.chunk_id)} className="block truncate rounded px-2 py-1 text-ink-2 hover:bg-surface-2 hover:text-ink">
                      {x.section_number && <span className="mr-1.5 text-ink-3 tabular-nums">{x.section_number}</span>}
                      {x.section_heading}
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {c.defined_terms.length > 0 && (
            <div>
              <h2 className="mb-2 text-xs font-medium text-ink-3 uppercase">Defined terms</h2>
              <div className="flex flex-wrap gap-1">
                {c.defined_terms.slice(0, 40).map((t) => (
                  <button key={t} onClick={() => setQ(t)} className="rounded bg-surface-2 px-1.5 py-0.5 text-xs text-ink-2 hover:text-accent">
                    {t}
                  </button>
                ))}
              </div>
            </div>
          )}
        </aside>

        <div className="min-w-0">
          <div className="sticky top-0 z-10 -mx-1 mb-3 bg-bg px-1 py-2">
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Find in this contract…" className={inputClass} />
            {q && (
              <p className="mt-1 text-xs text-ink-3">
                {chunks.length} matching chunk{chunks.length === 1 ? "" : "s"}
              </p>
            )}
          </div>
          <div className="space-y-3">
            {chunks.map((x) => (
              <div
                key={x.chunk_id}
                id={x.chunk_id}
                className={cx(
                  "scroll-mt-24 rounded-xl border bg-surface p-4",
                  target === x.chunk_id ? "border-accent ring-2 ring-accent/20" : "border-line",
                )}
              >
                <div className="mb-2 flex items-baseline justify-between gap-3">
                  <h3 className="text-sm font-semibold">
                    {x.section_number && <span className="mr-1.5 text-ink-3">{x.section_number}</span>}
                    {x.section_heading || "—"}
                    {(x.n_parts ?? 1) > 1 && (
                      <span className="ml-2 text-xs font-normal text-ink-3">
                        part {(x.part ?? 0) + 1} of {x.n_parts}
                      </span>
                    )}
                  </h3>
                  <span className="shrink-0 font-mono text-[0.7rem] text-ink-3">{x.chunk_id.split("::").slice(1).join("::")}</span>
                </div>
                <ClauseText text={x.text} quote={q.length >= 3 ? q : null} scroll={false} minRatio={1} />
              </div>
            ))}
          </div>
        </div>
      </div>
    </>
  );
}
