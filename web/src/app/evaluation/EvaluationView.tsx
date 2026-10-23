"use client";

import { useEffect, useState } from "react";
import { Card, ErrorBox, fmtMs, PageHeader, Spinner, Stat } from "@/components/ui";
import { getMetrics, type Metrics } from "@/lib/api";

const RET_COLS: [string, string][] = [
  ["recall@5", "recall@5"],
  ["recall@50", "recall@50"],
  ["mrr", "MRR"],
  ["ndcg@5", "nDCG@5"],
  ["latency_p95_ms", "p95"],
];

const pct = (x?: number) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);
const num = (k: string, x?: number) => (x == null ? "—" : k.includes("latency") ? fmtMs(x) : x.toFixed(3));

export default function EvaluationView() {
  const [m, setM] = useState<Metrics | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    getMetrics().then(setM, setError);
  }, []);

  return (
    <>
      <PageHeader
        title="Evaluation"
        sub="Recorded numbers from the eval runs, served by the API so the quality claims are inspectable at runtime. 148-question golden set; see docs/EVAL.md for methodology and limitations."
      />
      {error != null && <ErrorBox error={error} />}
      {!m && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-3">
          <Spinner /> Loading…
        </div>
      )}
      {m && (
        <div className="space-y-6">
          <section>
            <h2 className="mb-2 text-sm font-semibold">Retrieval</h2>
            <Card className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-line text-left text-xs text-ink-3">
                    <th className="px-4 py-2 font-medium">Scope</th>
                    <th className="px-4 py-2 font-medium">Configuration</th>
                    {RET_COLS.map(([, l]) => (
                      <th key={l} className="px-4 py-2 text-right font-medium">
                        {l}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-line tabular-nums">
                  {(["corpus_wide", "doc_scoped"] as const).flatMap((scope) => {
                    const base = m.retrieval.day3_dense_baseline[scope];
                    const best = m.retrieval[scope];
                    return [
                      <tr key={`${scope}-b`} className="text-ink-2">
                        <td className="px-4 py-2" rowSpan={2}>
                          <span className="font-medium text-ink">{scope === "corpus_wide" ? "Whole corpus" : "One contract"}</span>
                        </td>
                        <td className="px-4 py-2">Dense baseline</td>
                        {RET_COLS.map(([k]) => (
                          <td key={k} className="px-4 py-2 text-right">
                            {num(k, base[k])}
                          </td>
                        ))}
                      </tr>,
                      <tr key={`${scope}-w`} className="font-medium">
                        <td className="px-4 py-2">
                          Served: <span className="font-mono text-xs">{best.config}</span>
                        </td>
                        {RET_COLS.map(([k]) => (
                          <td key={k} className="px-4 py-2 text-right">
                            {num(k, best.aggregate[k])}
                          </td>
                        ))}
                      </tr>,
                    ];
                  })}
                </tbody>
              </table>
            </Card>
            <p className="mt-2 text-xs leading-relaxed text-ink-3">{m.retrieval._note}</p>
          </section>

          {m.generation && (
            <section>
              <h2 className="mb-2 text-sm font-semibold">Generated answers</h2>
              <Card className="grid grid-cols-2 gap-5 p-5 sm:grid-cols-4">
                <Stat label="Faithfulness (judge)" value={pct(m.generation.faithfulness_mean)} />
                <Stat label="Hallucination rate" value={pct(m.generation.hallucination_rate)} />
                <Stat label="Correct refusal" value={pct(m.generation.correct_refusal_rate)} hint={`${m.generation.n_unanswerable} unanswerable`} />
                <Stat label="False answers" value={pct(m.generation.false_answer_rate)} />
                <Stat label="Over-refusal" value={pct(m.generation.over_refusal_rate)} hint={`${m.generation.n_answerable} answerable`} />
                <Stat
                  label="Claims stripped"
                  value={`${m.generation.claims_raw - m.generation.claims_kept} / ${m.generation.claims_raw}`}
                  hint={`${m.generation.n_bad_citation} citation · ${m.generation.n_bad_quote} quote`}
                />
                <Stat label="JSON parse failures" value={pct(m.generation.parse_failure_rate)} />
                <Stat label="Latency p95" value={fmtMs(m.generation.latency_p95_ms)} />
              </Card>
            </section>
          )}

          {m.agent && (
            <section>
              <h2 className="mb-2 text-sm font-semibold">Audit agent</h2>
              <Card className="p-5">
                <p className="mb-4 text-sm text-ink-2">
                  Policy: <span className="italic">“{m.agent.policy}”</span> · {m.agent.n} contracts · ground truth: {m.agent.ground_truth}
                </p>
                <div className="grid grid-cols-2 gap-5 sm:grid-cols-4">
                  <Stat label="Precision" value={Number(m.agent.precision).toFixed(2)} hint={`fp = ${m.agent.fp}`} />
                  <Stat label="Recall" value={Number(m.agent.recall).toFixed(2)} hint={`fn = ${m.agent.fn}`} />
                  <Stat label="Citation validity" value={String(m.agent.citation_validity)} />
                  <Stat label="TP / TN" value={`${m.agent.tp} / ${m.agent.tn}`} />
                </div>
              </Card>
            </section>
          )}
        </div>
      )}
    </>
  );
}
