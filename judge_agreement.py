"""Task 6 - validate the LLM judge against hand labels.

An unvalidated LLM judge is a vibe with a decimal point. gen_runner.py already
writes evals/judge_pairs.json with human_supported=null, so there is no separate
label-maker script: sample 40, fill them in, score.

  uv run python judge_agreement.py sample   # -> evals/judge_labels.json
  uv run python judge_agreement.py score
"""

import json
import random
import sys
from pathlib import Path

PAIRS = Path("evals/judge_pairs-final.json")   # the full 148-question 3b run
LABELS = Path("evals/judge_labels.json")
N = 40


def sample() -> None:
    pairs = [p for p in json.loads(PAIRS.read_text())
             if p["judge_supported"] is not None]
    random.seed(42)
    # Stratify by verdict so the sample can measure BOTH error directions.
    # An all-"supported" sample cannot detect a lenient judge.
    yes = [p for p in pairs if p["judge_supported"]]
    no = [p for p in pairs if not p["judge_supported"]]
    take_no = min(len(no), N // 2)
    picked = random.sample(no, take_no) + random.sample(yes, min(len(yes), N - take_no))
    random.shuffle(picked)
    LABELS.write_text(json.dumps(picked, indent=2))
    print(f"{len(picked)} pairs -> {LABELS} "
          f"({take_no} judged unsupported, {len(picked)-take_no} supported)")
    print(f"  pool: {len(pairs)} judged pairs ({len(no)} unsupported, {len(yes)} supported)")


def score() -> None:
    raw = json.loads(LABELS.read_text())
    labels = raw["labels"] if isinstance(raw, dict) else raw
    done = [l for l in labels if l["human_supported"] is not None]
    if not done:
        print(f"no labels filled in yet in {LABELS}")
        return
    agree = sum(1 for l in done if l["human_supported"] == l["judge_supported"])
    fp = sum(1 for l in done if l["judge_supported"] and not l["human_supported"])
    fn = sum(1 for l in done if not l["judge_supported"] and l["human_supported"])
    print(f"judge agreement: {agree}/{len(done)} = {100*agree//len(done)}%")
    print(f"  judge too lenient (said supported, human said not): {fp}")
    print(f"  judge too strict  (said not, human said supported): {fn}")
    verdict = ("trust it" if 100*agree/len(done) >= 90 else
               "usable - quote agreement alongside every faithfulness number"
               if 100*agree/len(done) >= 85 else
               "tighten the judge prompt and re-measure" if 100*agree/len(done) >= 75 else
               "DO NOT report faithfulness as a headline; report deterministic validation only")
    print(f"  verdict: {verdict}")


if __name__ == "__main__":
    {"sample": sample, "score": score}[sys.argv[1] if len(sys.argv) > 1 else "score"]()
