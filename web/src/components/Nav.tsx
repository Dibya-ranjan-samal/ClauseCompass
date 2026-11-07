"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState } from "react";
import { describeLlm, getHealth, type Health } from "@/lib/api";
import { cx } from "./ui";

const LINKS = [
  { href: "/", label: "Ask", desc: "Question answering" },
  { href: "/audit", label: "Audit", desc: "Policy compliance" },
  { href: "/contracts", label: "Contracts", desc: "Browse the corpus" },
  { href: "/traces", label: "Traces", desc: "What ran, how long" },
  { href: "/evaluation", label: "Evaluation", desc: "Measured quality" },
];

export function Nav() {
  const path = usePathname();
  const active = (href: string) => (href === "/" ? path === "/" : path.startsWith(href));
  return (
    <nav className="flex gap-1 overflow-x-auto md:flex-col md:overflow-visible">
      {LINKS.map((l) => (
        <Link
          key={l.href}
          href={l.href}
          className={cx(
            "rounded-lg px-3 py-2 text-sm whitespace-nowrap transition-colors",
            active(l.href)
              ? "bg-surface-2 font-medium text-ink"
              : "text-ink-2 hover:bg-surface-2/60 hover:text-ink",
          )}
        >
          {l.label}
          <span className="hidden text-xs font-normal text-ink-3 md:block">{l.desc}</span>
        </Link>
      ))}
    </nav>
  );
}

export function HealthPill() {
  const [h, setH] = useState<Health | null>(null);
  const [down, setDown] = useState(false);

  useEffect(() => {
    let alive = true;
    const tick = () =>
      getHealth()
        .then((r) => alive && (setH(r), setDown(false)))
        .catch(() => alive && setDown(true));
    tick();
    const id = setInterval(tick, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const ok = h?.status === "ok" && !down;
  const label = down ? "API offline" : !h ? "Checking…" : ok ? "All systems ready" : "Degraded";
  const parts = h && !down
    ? [
        ["Models", h.models_loaded],
        ["Qdrant", h.qdrant],
        [describeLlm(h.llm_provider).provider, h.llm],
      ] as const
    : [];

  return (
    <div className="group relative">
      <div className="flex items-center gap-2 rounded-lg px-3 py-2 text-xs text-ink-2">
        <span
          className={cx(
            "size-2 rounded-full",
            !h && !down ? "bg-ink-3" : ok ? "bg-ok" : down ? "bg-bad" : "bg-warn",
          )}
        />
        {label}
      </div>
      {h?.llm_provider && !down && (
        <div className="truncate px-3 pb-1 font-mono text-[0.68rem] text-ink-3" title={h.llm_provider}>
          {h.llm_provider}
        </div>
      )}
      {parts.length > 0 && (
        <div className="flex gap-3 px-3 pb-2 text-xs text-ink-3">
          {parts.map(([k, v]) => (
            <span key={k} className={v ? "" : "text-bad"}>
              {v ? "✓" : "✗"} {k}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
