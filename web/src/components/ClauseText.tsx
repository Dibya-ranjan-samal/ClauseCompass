"use client";

import { useEffect, useRef } from "react";
import { findQuote } from "@/lib/highlight";
import { cx } from "./ui";

/** Contract text with the cited quote highlighted and scrolled into view. */
export function ClauseText({
  text,
  quote,
  className,
  scroll = true,
  minRatio = 0.5,
}: {
  text: string;
  quote?: string | null;
  className?: string;
  scroll?: boolean;
  /** Below 1, a near-verbatim quote is still highlighted and labelled partial. */
  minRatio?: number;
}) {
  const m = findQuote(text, quote, minRatio);
  const markRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (scroll) markRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [quote, scroll]);

  return (
    <div className={cx("clause-text font-serif text-[0.92rem] leading-relaxed text-ink", className)}>
      {m && m.ratio < 1 && (
        <p className="mb-2 font-sans text-xs text-warn">
          Partial match: {Math.round(m.ratio * 100)}% of the quote appears verbatim in this text.
        </p>
      )}
      {m ? (
        <>
          {text.slice(0, m.start)}
          <mark ref={markRef}>{text.slice(m.start, m.end)}</mark>
          {text.slice(m.end)}
        </>
      ) : (
        text
      )}
    </div>
  );
}
