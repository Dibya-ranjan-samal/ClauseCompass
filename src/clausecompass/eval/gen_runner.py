"""Day 5 - generation eval: refusal accuracy, deterministic validation, faithfulness.

Refusal and validation metrics are deterministic and free. Faithfulness needs a
judge, and a judge is only worth reporting alongside its measured agreement with
hand labels - see scripts note in notes/day5-explained.md.

Retrieval comes from the precomputed golden_hits.json (see precompute_hits.py),
so this runner is pure LLM work and re-runs hit the SQLite response cache.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from clausecompass.eval.runner import load_golden
from clausecompass.generation.answer import answer_from_hits

HITS = Path("data/interim/golden_hits.json")

JUDGE_SYSTEM = """You are evaluating whether a claim is supported by contract text.

Answer with ONLY valid JSON: {"supported": true, "reason": "one short sentence"}

supported true  - the excerpt directly states or unambiguously entails the claim
supported false - the excerpt does not contain this information, contradicts it,
                  or the claim adds specifics not present in the excerpt

Be strict. Plausibility is not support. If the claim adds a number, a duration,
or a party name that is not in the excerpt, it is not supported."""


class Hit:
    """Minimal stand-in for retrieval.base.Hit when replaying cached hits."""

    def __init__(self, chunk_id, text):
        self.chunk_id, self.text = chunk_id, text


def load_hits(which="rerank"):
    raw = json.loads(HITS.read_text())
    return {k.split(":", 1)[1]: [Hit(h["chunk_id"], h["text"]) for h in v]
            for k, v in raw.items() if k.startswith(f"{which}:")}


def judge_claim(statement: str, excerpt: str, llm) -> dict:
    user = (f"<excerpt>\n{excerpt}\n</excerpt>\n\nClaim: {statement}\n\n"
            "Is this claim supported by the excerpt? Return only JSON.")
    r = llm.complete(JUDGE_SYSTEM, user, max_tokens=200)
    try:
        t = r.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
        d = json.loads(t)
        return {"supported": bool(d.get("supported")), "reason": d.get("reason", "")}
    except Exception:
        # A judge that cannot be parsed must not silently count as "unsupported":
        # that would inflate the hallucination rate with judge failures.
        return {"supported": None, "reason": "judge_parse_failed"}


def run(gen_llm, judge_llm, config_name, which="rerank", sample_n=None,
        judge=True) -> dict:
    golden = load_golden()
    hits_by_q = load_hits(which)
    if sample_n:
        golden = golden[:sample_n]

    # Two passes, not interleaved. The generator (1.9 GB) and judge (4.7 GB) do
    # not both fit comfortably in 8 GB, and alternating per claim would make
    # Ollama unload and reload a 4.7 GB model on every question.
    answers, judge_pairs = {}, []
    for q in golden:
        hits = hits_by_q.get(q["id"], [])
        answers[q["id"]] = answer_from_hits(q["question"], hits, gen_llm)

    if judge:
        for q in golden:
            a = answers[q["id"]]
            if not (a.answer_found and a.claims):
                continue
            hits = hits_by_q.get(q["id"], [])
            for c in a.claims:
                excerpt = "\n\n".join(t.text for t in hits
                                       if t.chunk_id in (c.get("chunk_ids") or []))
                v = judge_claim(c["statement"], excerpt, judge_llm)
                judge_pairs.append({"qid": q["id"], "statement": c["statement"],
                                    "excerpt": excerpt, "judge_supported": v["supported"],
                                    "judge_reason": v["reason"], "human_supported": None})

    verdicts = {}
    for jp in judge_pairs:
        verdicts.setdefault(jp["qid"], []).append(jp["judge_supported"])

    rows, false_answers = [], []
    for q in golden:
        a = answers[q["id"]]
        vs = [v for v in verdicts.get(q["id"], []) if v is not None]
        f_total, f_supported = len(vs), sum(vs)

        row = {
            "id": q["id"], "type": q["type"], "answerable": q["answerable"],
            "answer_found": a.answer_found,
            "correct_refusal": (not q["answerable"]) and (not a.answer_found),
            "false_answer": (not q["answerable"]) and a.answer_found,
            "over_refusal": q["answerable"] and (not a.answer_found),
            "faithfulness": round(f_supported / f_total, 3) if f_total else None,
            "n_claims_raw": a.validation["n_claims_raw"],
            "n_claims_kept": a.validation["n_claims_kept"],
            "n_bad_citation": a.validation["n_bad_citation"],
            "n_bad_quote": a.validation["n_bad_quote"],
            "parse_failed": a.validation["parse_failed"],
            "truncated": a.validation.get("truncated", False),
            "latency_ms": a.latency_ms, "cost_usd": a.cost_usd,
        }
        rows.append(row)
        if row["false_answer"]:
            false_answers.append({"id": q["id"], "question": q["question"],
                                  "answer": a.answer, "claims": a.claims})

    faiths = [r["faithfulness"] for r in rows if r["faithfulness"] is not None]
    unans = [r for r in rows if not r["answerable"]]
    ans = [r for r in rows if r["answerable"]]
    n = len(rows)
    agg = {
        "n_questions": n,
        "faithfulness_mean": round(statistics.mean(faiths), 3) if faiths else None,
        "hallucination_rate": round(1 - statistics.mean(faiths), 3) if faiths else None,
        "n_judged_questions": len(faiths),
        "n_unanswerable": len(unans),
        "correct_refusal_rate": round(sum(r["correct_refusal"] for r in unans) / max(len(unans), 1), 3),
        "false_answer_rate": round(sum(r["false_answer"] for r in unans) / max(len(unans), 1), 3),
        "n_answerable": len(ans),
        "over_refusal_rate": round(sum(r["over_refusal"] for r in ans) / max(len(ans), 1), 3),
        "citation_validation_fires": sum(r["n_bad_citation"] + r["n_bad_quote"] for r in rows),
        "n_bad_citation": sum(r["n_bad_citation"] for r in rows),
        "n_bad_quote": sum(r["n_bad_quote"] for r in rows),
        "claims_raw": sum(r["n_claims_raw"] for r in rows),
        "claims_kept": sum(r["n_claims_kept"] for r in rows),
        "parse_failure_rate": round(sum(r["parse_failed"] for r in rows) / n, 3),
        "truncation_rate": round(sum(r["truncated"] for r in rows) / n, 3),
        "latency_p95_ms": round(sorted(r["latency_ms"] for r in rows)[int(0.95 * n) - 1], 1),
        "cost_total_usd": round(sum(r["cost_usd"] for r in rows), 4),
    }

    Path("runs").mkdir(exist_ok=True)
    out = Path("runs") / f"{time.strftime('%Y%m%d-%H%M%S')}-gen-{config_name}.json"
    out.write_text(json.dumps({"config": config_name, "retrieval": which,
                               "gen_model": gen_llm.name,
                               "judge_model": judge_llm.name if judge else None,
                               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "aggregate": agg, "per_question": rows,
                               "false_answers": false_answers}, indent=2))
    if judge_pairs:
        # Config-scoped filename. A fixed one let the 7b comparison clobber the
        # 127-pair pool the hand labels were drawn from - it survived only
        # because it had been copied by hand.
        Path("evals").mkdir(exist_ok=True)
        Path(f"evals/judge_pairs-{config_name}.json").write_text(
            json.dumps(judge_pairs, indent=2))

    for k, v in agg.items():
        print(f"  {k:26} {v}")
    print(f"saved -> {out}")
    if false_answers:
        print(f"\n  {len(false_answers)} FALSE ANSWERS (answered an unanswerable question):")
        for fa in false_answers[:3]:
            print(f"    [{fa['id']}] {fa['question'][:90]}")
            print(f"           -> {fa['answer'][:140]}")
    return agg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="final")
    ap.add_argument("--gen-model", default="qwen2.5:3b")
    ap.add_argument("--judge-model", default="qwen2.5:3b")
    ap.add_argument("--retrieval", default="rerank")
    ap.add_argument("--sample", type=int)
    ap.add_argument("--no-judge", action="store_true")
    a = ap.parse_args()

    from clausecompass.generation.llm import OllamaProvider
    run(OllamaProvider(a.gen_model), OllamaProvider(a.judge_model), a.config,
        a.retrieval, a.sample, judge=not a.no_judge)
