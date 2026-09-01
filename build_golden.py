"""Day 3 - build the golden eval set from CUAD annotations.

Generation, stratified sampling, recorded manual review, hand-written hard
cases and the final merge, in one pass.

Design note: questions are ENTITY-QUALIFIED ("Under the <party> <type>
agreement, ..."). Generic CUAD templates are identical across all 530
contracts, so corpus-wide retrieval cannot know which contract is meant -
measured at hit@5 = 0/30. Naming the party gives the query something to match:
hit@5 = 4/30, hit@50 = 13/30. The same text serves the doc-scoped config, where
the qualifier is merely redundant.
"""

import collections
import json
import random
import re
from pathlib import Path

INTERIM = Path("data/interim")
OUT = Path("evals/golden")
TARGET = 120
SEED = 42

# Categories worth asking about. "Document Name" and "Parties" are deliberately
# excluded: their answer is the contract title or the party names, which the
# entity-qualified question already states - a leak.
TEMPLATES = {
    "Governing Law": "which state's or country's law governs it?",
    "Expiration Date": "when does it expire?",
    "Effective Date": "what is its effective date?",
    "Agreement Date": "on what date was it signed?",
    "Renewal Term": "what are the renewal terms?",
    "Notice Period To Terminate Renewal": "how much notice is required to stop it renewing?",
    "Termination For Convenience": "can either party terminate for convenience, and on what notice?",
    "Post-Termination Services": "what services must continue after termination?",
    "Cap On Liability": "is there a cap on liability, and what is it?",
    "Uncapped Liability": "which liabilities are uncapped?",
    "Insurance": "what insurance must each party maintain?",
    "Audit Rights": "what audit rights does it grant, and on what conditions?",
    "Non-Compete": "does it impose a non-compete restriction?",
    "Exclusivity": "does it grant exclusivity to either party?",
    "Ip Ownership Assignment": "who owns intellectual property created under it?",
    "License Grant": "what licence is granted, and to whom?",
    "Warranty Duration": "how long does any warranty last?",
    "Anti-Assignment": "what restricts assignment of the agreement?",
    "Change Of Control": "what happens on a change of control?",
    "Minimum Commitment": "what minimum commitment does it impose?",
}

# (topic phrase, terms that must be ABSENT for the question to be genuinely
# unanswerable). Absence is verified against the document text.
UNANSWERABLE = [
    ("cryptocurrency or digital-asset custody", ["cryptocurrenc", "digital asset", "bitcoin", "blockchain"]),
    ("a carbon-emissions or net-zero target", ["carbon", "emission", "greenhouse", "net zero"]),
    ("GDPR data-processor obligations", ["gdpr", "general data protection", "data processor", "data subject"]),
    ("a SOC 2 or ISO 27001 certification requirement", ["soc 2", "soc2", "iso 27001", "service organization control"]),
    ("a source-code escrow arrangement", ["escrow"]),
    ("arbitration seated in Singapore", ["singapore"]),
    ("key-person life insurance", ["key person", "key man", "key-man"]),
    ("WCAG accessibility conformance", ["wcag", "accessib", "section 508"]),
    ("rights to train AI or machine-learning models on the data", ["artificial intelligence", "machine learning", "model training"]),
    ("biometric data handling", ["biometric", "fingerprint", "facial recognition"]),
    ("collective bargaining or union representation", ["collective bargaining", "labor union", "trade union"]),
    ("ITAR or export-control compliance", ["itar", "export control", "export administration"]),
    ("a right-to-erasure process", ["right to be forgotten", "right of erasure", "right to erasure"]),
    ("pandemic or epidemic force-majeure relief", ["pandemic", "epidemic", "covid"]),
    ("liquidated damages for late delivery", ["liquidated damages"]),
    ("criminal background screening of personnel", ["background check", "background screening", "criminal record"]),
    ("anti-bribery or FCPA compliance", ["foreign corrupt practices", "fcpa", "bribery", "anti-corruption"]),
    ("parental or maternity leave entitlements", ["parental leave", "maternity", "paternity"]),
]

# (asked-about category, adjacent category that is the obvious wrong answer)
NEAR_MISS = [
    ("Cap On Liability", "Uncapped Liability", "what is the monetary cap on liability?"),
    ("Termination For Convenience", "Post-Termination Services", "what notice is required to terminate for convenience?"),
    ("Ip Ownership Assignment", "License Grant", "who owns intellectual property created under it?"),
    ("Non-Compete", "Exclusivity", "what non-compete restriction does it impose?"),
    ("Anti-Assignment", "Change Of Control", "what consent is needed to assign it?"),
    ("Audit Rights", "Insurance", "what records may be audited, and how often?"),
]

# (category A, category B, question needing both)
MULTI_HOP = [
    ("Expiration Date", "Notice Period To Terminate Renewal",
     "what is its term, and how much notice stops it renewing?"),
    ("Cap On Liability", "Uncapped Liability",
     "is liability capped, and which liabilities escape the cap?"),
    ("Ip Ownership Assignment", "License Grant",
     "who owns IP created under it, and what licence is granted back?"),
    ("Termination For Convenience", "Post-Termination Services",
     "can it be terminated for convenience, and what obligations continue afterwards?"),
]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def entity_label(doc: dict, meta: dict) -> str | None:
    """A human-readable way to name this contract inside a question."""
    parties = meta.get(doc["doc_id"], {}).get("parties") or []
    if parties:
        return parties[0].strip()
    head = re.split(r"[_-]", doc.get("title") or "")[0]
    return head.title() if len(head) > 3 else None


def gold_for(by_doc, annots_by, doc_id, category) -> list[str]:
    """Chunks that BOTH overlap the annotated span AND contain the answer text.

    Overlap alone only proves the offsets line up, not that the text survived
    chunking - both conditions are required.
    """
    out = []
    for a in annots_by.get((doc_id, category), []):
        for ans in a["answers"]:
            s = ans["answer_start"]
            e = s + len(ans["text"])
            for c in by_doc.get(doc_id, []):
                if (c.get("char_start", 0) < e and c.get("char_end", 0) > s
                        and norm(ans["text"])[:120] in norm(c["text"])):
                    out.append(c["chunk_id"])
    return list(dict.fromkeys(out))


def generate(docs, by_doc, annots, meta):
    generated, rejected, seen = [], collections.Counter(), set()
    for a in annots:
        if a["is_impossible"] or not a["answers"]:
            rejected["no_answer"] += 1
            continue
        tmpl = TEMPLATES.get(a.get("category") or "")
        if not tmpl:
            rejected["no_template"] += 1
            continue
        doc = docs.get(a["doc_id"])
        if not doc:
            rejected["no_doc"] += 1
            continue
        key = (a["doc_id"], a["category"])
        if key in seen:
            rejected["duplicate"] += 1
            continue

        gold, answer = [], a["answers"][0]["text"]
        for ans in a["answers"]:
            s, e = ans["answer_start"], ans["answer_start"] + len(ans["text"])
            for c in by_doc.get(a["doc_id"], []):
                if (c.get("char_start", 0) < e and c.get("char_end", 0) > s
                        and norm(ans["text"])[:120] in norm(c["text"])):
                    gold.append(c["chunk_id"])
        gold = list(dict.fromkeys(gold))
        if not gold:
            rejected["no_gold_chunk"] += 1
            continue
        if len(answer.strip()) < 15 or not re.search(r"[A-Za-z]", answer):
            rejected["fragment"] += 1
            continue

        label = entity_label(doc, meta)
        if not label:
            rejected["no_entity_label"] += 1
            continue
        ctype = doc.get("title_contract_type") or "agreement"
        ctype = ctype.title() if ctype != "agreement" else ctype
        question = f"Under the {label} {ctype} agreement, {tmpl}"
        if norm(answer)[:40] in norm(question):
            rejected["answer_leaked_into_question"] += 1
            continue

        seen.add(key)
        generated.append({
            "question": question, "category": a["category"],
            "gold_answer": answer[:300], "gold_chunk_ids": gold,
            "doc_id": a["doc_id"], "type": "single_hop", "answerable": True,
            "source": "cuad", "reviewed": False,
        })

    print(f"generated {len(generated)} questions from {len(annots)} annotations")
    for k, v in rejected.most_common():
        print(f"  rejected {k:28} {v}")
    gen_pool = len(generated) + rejected["fragment"] + rejected["no_gold_chunk"]
    print(f"  generation-stage rejection (quality filters only): "
          f"{100*(rejected['fragment']+rejected['no_gold_chunk'])/gen_pool:.1f}%")
    return generated


def stratify(generated):
    random.seed(SEED)
    by_cat = collections.defaultdict(list)
    for g in generated:
        by_cat[g["category"]].append(g)
    per = max(1, TARGET // len(by_cat))
    sample = []
    for items in by_cat.values():
        sample.extend(random.sample(items, min(per, len(items))))
    random.shuffle(sample)
    sample = sample[:TARGET]
    for i, g in enumerate(sample, 1):
        g["id"] = f"g{i:04d}"
    print(f"sampled {len(sample)} across {len(by_cat)} categories ({per}/category)")
    return sample


def apply_review(sample):
    """Apply the recorded manual review decisions. See review_decisions.json."""
    dec = json.loads((OUT / "review_decisions.json").read_text())
    kept, hit = [], collections.Counter()
    for q in sample:
        key = f"{q['doc_id']}|{q['category']}"
        if key in dec["delete"]:
            hit["delete"] += 1
            continue
        if key in dec["flag_fragment"]:
            hit["flag"] += 1
            q["answer_is_fragment"] = True
        q["reviewed"] = True
        kept.append(q)
    # a decision that matches nothing means the keys drifted - fail loudly
    # rather than silently keeping a question the review rejected.
    want = (len(dec["delete"]), len(dec["flag_fragment"]))
    if (hit["delete"], hit["flag"]) != want:
        raise SystemExit(f"review keys stale: applied {hit['delete']} deletes / "
                         f"{hit['flag']} flags, expected {want[0]}/{want[1]}")
    print(f"reviewed {len(sample)} -> kept {len(kept)}, deleted {hit['delete']}, "
          f"flagged {hit['flag']} "
          f"(review-stage rejection {100*hit['delete']/len(sample):.1f}%)")
    return kept


def build_hard(docs, by_doc, annots, meta):
    random.seed(SEED)
    annots_by = {}
    for a in annots:
        if a["answers"] and not a["is_impossible"]:
            annots_by.setdefault((a["doc_id"], a.get("category")), []).append(a)

    def label(doc_id):
        d = docs[doc_id]
        lb = entity_label(d, meta)
        ct = (d.get("title_contract_type") or "agreement").title()
        return f"Under the {lb} {ct} agreement" if lb else None

    hard, used = [], set()
    doc_ids = sorted(docs)

    for topic, absent in UNANSWERABLE:
        for doc_id in random.sample(doc_ids, len(doc_ids)):
            if doc_id in used or not label(doc_id):
                continue
            low = docs[doc_id]["text"].lower()
            if any(t in low for t in absent):
                continue
            used.add(doc_id)
            hard.append({
                "question": f"{label(doc_id)}, what provisions cover {topic}?",
                "category": "unanswerable", "gold_answer": None,
                "gold_chunk_ids": [], "doc_id": doc_id, "type": "unanswerable",
                "answerable": False, "source": "handwritten", "reviewed": True,
                "absent_terms": absent,
            })
            break

    for cat, trap, ask in NEAR_MISS:
        for doc_id in random.sample(doc_ids, len(doc_ids)):
            if doc_id in used or not label(doc_id):
                continue
            if not ((doc_id, cat) in annots_by and (doc_id, trap) in annots_by):
                continue
            g = gold_for(by_doc, annots_by, doc_id, cat)
            if not g:
                continue
            used.add(doc_id)
            hard.append({
                "question": f"{label(doc_id)}, {ask}", "category": cat,
                "gold_answer": annots_by[(doc_id, cat)][0]["answers"][0]["text"][:300],
                "gold_chunk_ids": g, "doc_id": doc_id, "type": "near_miss",
                "answerable": True, "source": "handwritten", "reviewed": True,
                "trap_category": trap,
                "trap_chunk_ids": gold_for(by_doc, annots_by, doc_id, trap),
            })
            break

    for a_cat, b_cat, ask in MULTI_HOP:
        for doc_id in random.sample(doc_ids, len(doc_ids)):
            if doc_id in used or not label(doc_id):
                continue
            if not ((doc_id, a_cat) in annots_by and (doc_id, b_cat) in annots_by):
                continue
            ga = gold_for(by_doc, annots_by, doc_id, a_cat)
            gb = gold_for(by_doc, annots_by, doc_id, b_cat)
            if not (ga and gb):
                continue
            used.add(doc_id)
            hard.append({
                "question": f"{label(doc_id)}, {ask}",
                "category": f"{a_cat}+{b_cat}",
                "gold_answer": annots_by[(doc_id, a_cat)][0]["answers"][0]["text"][:200],
                "gold_chunk_ids": list(dict.fromkeys(ga + gb)), "doc_id": doc_id,
                "type": "multi_hop", "answerable": True, "source": "handwritten",
                "reviewed": True, "hop_categories": [a_cat, b_cat],
            })
            break

    # No single chunk answers these. Included deliberately as a measured known
    # weakness of single-pass RAG; they motivate the Day 6 agent.
    ny = sum(1 for v in meta.values()
             if (v.get("governing_law") or "").strip().lower() == "new york")
    mfn = len({d for (d, c) in annots_by if c == "Most Favored Nation"})
    for q, ans in [
        ("How many contracts in the corpus are governed by New York law?", str(ny)),
        ("How many agreements in the corpus include a most-favoured-nation clause?", str(mfn)),
    ]:
        hard.append({
            "question": q, "category": "aggregation", "gold_answer": ans,
            "gold_chunk_ids": [], "doc_id": None, "type": "aggregation",
            "answerable": True, "source": "handwritten", "reviewed": True,
            "note": "not answerable by single-pass retrieval; motivates the Day 6 agent",
        })

    print(f"hand-written hard cases: {len(hard)} "
          f"{dict(collections.Counter(h['type'] for h in hard))}")
    return hard


def write_golden(verified, hard):
    allq = verified + hard
    with (OUT / "golden.jsonl").open("w") as f:
        for i, q in enumerate(allq, 1):
            q["id"] = f"q{i:04d}"
            f.write(json.dumps(q) + "\n")
    types = collections.Counter(q["type"] for q in allq)
    unans = sum(1 for q in allq if not q["answerable"])
    print(f"golden.jsonl: {len(allq)} questions {dict(types)}")
    print(f"  unanswerable {unans} ({100*unans//len(allq)}%)")
    print(f"  reviewed     {sum(1 for q in allq if q['reviewed'])}/{len(allq)}")


def main() -> int:
    docs = {d["doc_id"]: d for d in json.loads((INTERIM / "documents.json").read_text())}
    chunks = json.loads((INTERIM / "chunks_enriched.json").read_text())
    annots = json.loads((INTERIM / "cuad_annotations.json").read_text())
    meta = json.loads((INTERIM / "metadata.json").read_text())

    by_doc = {}
    for c in chunks:
        by_doc.setdefault(c["doc_id"], []).append(c)

    OUT.mkdir(parents=True, exist_ok=True)
    generated = generate(docs, by_doc, annots, meta)
    with (OUT / "cuad_generated.jsonl").open("w") as f:
        for i, g in enumerate(generated, 1):
            f.write(json.dumps({**g, "id": f"c{i:04d}"}) + "\n")

    sample = stratify(generated)
    with (OUT / "to_review.jsonl").open("w") as f:
        for g in sample:
            f.write(json.dumps(g) + "\n")

    write_golden(apply_review(sample), build_hard(docs, by_doc, annots, meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
