"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { getContracts, type ContractSummary } from "@/lib/api";
import { contractTitle, cx, inputClass } from "./ui";

/**
 * Scope selector: "whole corpus" or one contract. Searching matches doc_id and
 * party names, since nobody remembers a CUAD filename.
 */
export function ContractPicker({
  value,
  onChange,
}: {
  value: string | null;
  onChange: (docId: string | null) => void;
}) {
  const [all, setAll] = useState<ContractSummary[]>([]);
  const [failed, setFailed] = useState(false);
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const [hi, setHi] = useState(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getContracts().then(setAll, () => setFailed(true));
  }, []);

  useEffect(() => {
    const close = (e: MouseEvent) => {
      if (!box.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const matches = useMemo(() => {
    const t = q.trim().toLowerCase();
    const rows = t
      ? all.filter(
          (c) =>
            c.doc_id.toLowerCase().includes(t) ||
            (c.parties ?? []).some((p) => p.toLowerCase().includes(t)) ||
            (c.contract_type ?? "").toLowerCase().includes(t),
        )
      : all;
    return rows.slice(0, 50);
  }, [all, q]);

  const selected = value ? all.find((c) => c.doc_id === value) : null;

  function pick(docId: string | null) {
    onChange(docId);
    setOpen(false);
    setQ("");
  }

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={cx(inputClass, "flex items-center justify-between gap-2 text-left")}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="min-w-0 truncate">
          {value ? (
            <>
              <span className="font-medium">{selected ? contractTitle(selected) : value}</span>
              {selected?.contract_type && (
                <span className="ml-2 text-ink-3">{selected.contract_type}</span>
              )}
            </>
          ) : (
            <>
              <span className="font-medium">Whole corpus</span>
              <span className="ml-2 text-ink-3">{all.length || 530} contracts</span>
            </>
          )}
        </span>
        <span className="text-ink-3">▾</span>
      </button>

      {open && (
        <div className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-line bg-surface shadow-lg">
          <input
            autoFocus
            value={q}
            onChange={(e) => {
              setQ(e.target.value);
              setHi(0);
            }}
            onKeyDown={(e) => {
              if (e.key === "ArrowDown") setHi((h) => Math.min(h + 1, matches.length));
              else if (e.key === "ArrowUp") setHi((h) => Math.max(h - 1, 0));
              else if (e.key === "Enter") pick(hi === 0 ? null : matches[hi - 1].doc_id);
              else if (e.key === "Escape") setOpen(false);
              else return;
              e.preventDefault();
            }}
            placeholder="Search by party, type or filename…"
            className="w-full border-b border-line bg-surface px-3 py-2 text-sm focus:outline-none"
          />
          <ul role="listbox" className="max-h-80 overflow-y-auto py-1 text-sm">
            <li
              role="option"
              aria-selected={!value}
              onMouseEnter={() => setHi(0)}
              onClick={() => pick(null)}
              className={cx("cursor-pointer px-3 py-2", hi === 0 && "bg-surface-2")}
            >
              <div className="font-medium">Whole corpus</div>
              <div className="text-xs text-ink-3">Dense retrieval across all chunks · fast</div>
            </li>
            {failed && <li className="px-3 py-2 text-xs text-bad">Could not load contracts.</li>}
            {matches.map((c, i) => (
              <li
                key={c.doc_id}
                role="option"
                aria-selected={value === c.doc_id}
                onMouseEnter={() => setHi(i + 1)}
                onClick={() => pick(c.doc_id)}
                className={cx("cursor-pointer px-3 py-2", hi === i + 1 && "bg-surface-2")}
              >
                <div className="truncate font-medium">{contractTitle(c)}</div>
                <div className="truncate font-mono text-xs text-ink-3">
                  {c.contract_type && c.contract_type !== "Unknown" ? `${c.contract_type} · ` : ""}
                  {c.doc_id}
                </div>
              </li>
            ))}
            {all.length > 0 && matches.length === 50 && (
              <li className="px-3 py-2 text-xs text-ink-3">Showing first 50 — keep typing to narrow.</li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}
