"""Task 5 - measure metadata extraction accuracy.

Two ground-truth sources, with deliberately different provenance:

contract_type - CUAD encodes the type in every filename
                ('...EX-10-DISTRIBUTOR AGREEMENT'). That is dataset-provided
                metadata, independent of the contract body the classifier
                reads, so it is legitimate ground truth - and it covers all
                510 CUAD documents rather than a 50-document sample.

governing_law - no such free signal, so it needs reading. labels_todo.json is
                generated with the relevant excerpts; whoever fills it in must
                not look at metadata.json first. Provenance is recorded in the
                file so the resulting number is never overstated.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

INTERIM = Path("data/interim")

# Canonical taxonomy: collapses CUAD's 144 raw title strings into 26 labels
# covering 95% of the corpus. Order matters - first substring hit wins.
CANON = [
    ("STRATEGIC ALLIANCE", "Strategic Alliance"), ("SPONSORSHIP", "Sponsorship"),
    ("DISTRIBUTOR", "Distribution"), ("DISTRIBUTION", "Distribution"),
    ("ENDORSEMENT", "Endorsement"), ("DEVELOPMENT", "Development"),
    ("OUTSOURCING", "Outsourcing"), ("CO-BRANDING", "Branding"),
    ("BRANDING", "Branding"), ("JOINT VENTURE", "Joint Venture"),
    ("JOINT FILING", "Joint Filing"), ("INTELLECTUAL PROPERTY", "IP"),
    ("TRADEMARK", "IP"), ("RESELLER", "Reseller"), ("AGENCY", "Agency"),
    ("FRANCHISE", "Franchise"), ("AFFILIATE", "Affiliate"),
    ("COLLABORATION", "Collaboration"), ("COOPERATION", "Cooperation"),
    ("CONSULTING", "Consulting"), ("PROMOTION", "Promotion"),
    ("MARKETING", "Marketing"), ("MAINTENANCE", "Maintenance"),
    ("HOSTING", "Hosting"), ("MANUFACTURING", "Manufacturing"),
    ("SUPPLY", "Supply"), ("TRANSPORTATION", "Transportation"),
    ("LICENSING", "License"), ("LICENSE", "License"),
    ("SERVICES", "Services"), ("SERVICE", "Services"),
]


def canon_type(title_type: str | None) -> str | None:
    if not title_type:
        return None
    t = title_type.upper()
    for key, val in CANON:
        if key in t:
            return val
    return None


def score_contract_type(meta: dict) -> None:
    rows = [(k, canon_type(v.get("title_contract_type")), v)
            for k, v in meta.items() if v.get("source") == "cuad"]
    labeled = [(k, truth, v) for k, truth, v in rows if truth]
    print(f"contract_type ground truth: {len(labeled)}/{len(rows)} CUAD docs "
          f"have a canonicalisable title type")

    for field, name in [("contract_type_tutorial", "tutorial 9 labels"),
                        ("contract_type", "corpus 26 labels")]:
        pred = [(v.get(field), truth) for _, truth, v in labeled]
        correct = sum(1 for p, t in pred if p == t)
        unknown = sum(1 for p, _ in pred if p == "Unknown")
        n = len(pred)
        print(f"  {name:<20} {correct}/{n} = {100*correct//max(n,1)}%"
              f"   (Unknown: {unknown}, {100*unknown//max(n,1)}%)")

    print("\n  most common confusions under the corpus label set:")
    conf = Counter((t, v.get("contract_type")) for _, t, v in labeled
                   if v.get("contract_type") != t)
    for (t, p), c in conf.most_common(8):
        print(f"    {c:3d}  truth {t:<20} -> predicted {p}")


def make_labels_todo(n: int, seed: int) -> None:
    docs = json.loads((INTERIM / "documents.json").read_text())
    random.seed(seed)
    sample = random.sample(docs, n)
    rows = []
    for d in sample:
        # Governing law sits in the final Miscellaneous section, not the
        # preamble, so show the regions that actually mention it.
        # Rank candidate windows instead of taking the first hit. The first
        # "law" mention in a contract is nearly always "applicable law"
        # boilerplate; the governing-law clause is the one that NAMES a
        # jurisdiction, so score for that.
        trig = (r"(?:govern(?:ed|ing)|choice\s+of\s+law|applicable\s+law"
                r"|construed\s+in\s+accordance|law\s+and\s+jurisdiction)")
        NAMES = re.compile(r"laws?\s+of\s*(?:the\s*)?"
                           r"(?:State|Commonwealth|Province|District)?\s*of?\s*[A-Z]")
        cands = [m.group(0).replace("\n", " ")
                 for m in re.finditer(rf".{{0,90}}{trig}.{{0,400}}", d["text"], re.I)]
        def rank(e):
            return (2 if NAMES.search(e) else 0) + \
                   (1 if re.search(r"(?i)govern(?:ed|ing)\s+law", e) else 0)
        excerpts = sorted(cands, key=rank, reverse=True)[:3]
        rows.append({
            "doc_id": d["doc_id"],
            "source": d["source"],
            "governing_law_excerpts": excerpts or ["(no 'govern' language found)"],
            "text_head": d["text"][:600].replace("\n", " "),
            "true_governing_law": "",
        })
    out = {
        "_provenance": "UNLABELLED. Fill true_governing_law by reading the "
                       "excerpts only. Do not consult metadata.json first - "
                       "that is the system being measured.",
        "_seed": seed,
        "rows": rows,
    }
    (INTERIM / "labels_todo.json").write_text(json.dumps(out, indent=2))
    print(f"wrote {INTERIM/'labels_todo.json'} with {len(rows)} unlabelled rows")


def score_governing_law() -> None:
    """Score dev and held-out separately.

    'none' in the truth column means the document genuinely has no
    governing-law clause (amendments, joint filings, redacted clauses). For
    those, an EMPTY prediction is the correct answer - so an empty string and
    the literal 'none' must be treated as equal. Scoring them as a mismatch
    understated accuracy by 18 points (80% vs 98% on the dev set) purely
    through label encoding.
    """
    pred = json.loads((INTERIM / "metadata.json").read_text())

    def norm(s: str | None) -> str:
        v = re.sub(r"\s+", " ", s or "").strip().lower()
        return "none" if v in ("", "none") else v

    for fname, name in [("labels_done.json", "DEV (scoring was tuned here)"),
                        ("labels_holdout_done.json", "HELD-OUT (never tuned on)")]:
        path = INTERIM / fname
        if not path.exists():
            print(f"governing_law {name}: {fname} absent - skipping")
            continue
        doc = json.loads(path.read_text())
        rows = [r for r in doc["rows"] if r.get("true_governing_law", "").strip()]

        exact = near = miss = wrong = 0
        errs = []
        for r in rows:
            t = norm(r["true_governing_law"])
            p = norm(pred.get(r["doc_id"], {}).get("governing_law"))
            if p == t:
                exact += 1
            elif t != "none" and p != "none" and (t in p or p in t):
                near += 1                      # "new york" vs "new york state"
                errs.append((r["doc_id"][:34], t, p + "  [NEAR]"))
            elif p == "none":
                miss += 1                      # failed to extract
                errs.append((r["doc_id"][:34], t, "(missing)"))
            else:
                wrong += 1                     # extracted the wrong jurisdiction
                errs.append((r["doc_id"][:34], t, p))
        n = len(rows)
        print(f"governing_law {name}: n={n}")
        print(f"  strict  {exact}/{n} = {100*exact//n}%")
        print(f"  lenient {exact+near}/{n} = {100*(exact+near)//n}%  "
              f"(+{near} normalisation-only)")
        print(f"  errors: {miss} missing, {wrong} wrong jurisdiction")
        for a, b, c in errs:
            print(f"    {a:<36} truth={b:<18} pred={c}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make-labels", type=int, default=0,
                    help="write labels_todo.json with N sampled docs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.make_labels:
        make_labels_todo(args.make_labels, args.seed)
        return 0

    meta = json.loads((INTERIM / "metadata.json").read_text())
    score_contract_type(meta)
    print()
    score_governing_law()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
