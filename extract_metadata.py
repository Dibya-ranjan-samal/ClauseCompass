"""Task 4 - extract filterable metadata for every document.

Runs ONCE, offline, during ingestion. Never in the query path.

Phases are deliberately sequential and each pipeline is freed before the next
loads: this machine has 8 GB of RAM and holding the zero-shot and NER models
simultaneously pushes it into swap.

Two contract-type label sets are scored:
  LABELS_TUTORIAL - the 9 labels from the Day 2 brief, which match only 35/510
                    CUAD contracts (6%) because they were written for an
                    MSA-only EDGAR corpus.
  LABELS_CORPUS   - 26 labels derived from CUAD's own title metadata, covering
                    95% of the corpus.
Keeping both lets Task 5 report a real before/after instead of one number.
"""

import argparse
import gc
import json
import re
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_CTX = 8192
LOCAL_MODELS = Path("models")

INTERIM = Path("data/interim")

# --------------------------------------------------------------------------
# 4a. Regex extraction
# --------------------------------------------------------------------------
DATE = (r"(\d{1,2}(?:st|nd|rd|th)?\s+day\s+of\s+[A-Z][a-z]+,?\s+\d{4}"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]{2,9},?\s+\d{4}"      # 27th December 2024
        r"|[A-Z][a-z]{2,9}\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"      # December 27, 2024
        r"|\d{1,2}/\d{1,2}/\d{2,4})")

DATE_PATTERNS = [
    rf"dated\s+as\s+of\s+{DATE}",
    rf"[Ee]ffective\s+[Dd]ate[^\n]{{0,30}}?{DATE}",
    rf"entered\s+into\s+(?:as\s+of\s+)?(?:this\s+)?{DATE}",
    rf"made\s+(?:and\s+entered\s+into\s+)?(?:as\s+of\s+)?(?:this\s+)?{DATE}",
    rf"[Dd]ated:?\s+{DATE}",
    rf"[Aa]s\s+of\s+{DATE}",
]
# Last resort: contracts state their date early even without an anchor phrase.
# Trades precision for recall - anchored patterns are tried first and win.
DATE_BARE = re.compile(DATE)
DATE_BARE_WINDOW = 3000

# A jurisdiction name, optionally wrapped in a container noun
# ("the State of Delaware", "the Province of British Columbia").
# Case-flexible keywords, but the CAPTURE still demands a capital letter so the
# jurisdiction is a proper noun. Found by hand-labelling: two of fifty documents
# write "by the Laws of the State of Illinois" with a capital L, which a
# lowercase-only pattern silently misses.
# \s* rather than \s+ between keywords absorbs OCR damage - one CUAD contract
# reads "the laws ofthe State ofTexas" with the spaces eaten.
L = r"[Ll]aws?"
OF = r"\s*of\s*"
CONTAINER = rf"(?:[Tt]he\s*)?(?:[Ss]tate|[Cc]ommonwealth|[Pp]rovince|[Dd]istrict|[Rr]epublic){OF}"
JURIS = r"([A-Z][A-Za-z]+(?:\s+(?:and\s+)?[A-Z][A-Za-z]+){0,3})"

# One permissive candidate finder; precision comes from scoring, not from the
# pattern. The dominant error before scoring was picking a party's incorporation
# state ("a company organized under the laws of the State of Delaware") - four
# of eight errors on the 50-document sample were a spurious "Delaware".
LAW_CANDIDATE = rf"{L}{OF}(?:{CONTAINER})?{JURIS}"

# Governing-law clauses sit in the closing Miscellaneous section; incorporation
# jurisdictions sit in the opening recitals. Position is the strongest signal.
LAW_HEADING = re.compile(
    r"(?i)(govern(?:ing|ed)\s+law|choice\s+of\s+law|applicable\s+law"
    r"|law\s+and\s+jurisdiction|governing\s+laws)")
LAW_VERB = re.compile(
    r"(?i)(govern|construed|interpreted|enforced|accordance)")

# Party aliases: (the "Supplier") / ("Vendor")
ALIAS = re.compile(r'\(\s*(?:the\s+)?["“”\']([A-Z][A-Za-z .&\-]{2,30})["“”\']\s*\)')

# Container nouns and pronouns that mean the regex captured the wrapper, not
# the jurisdiction. "Province" appeared 10x before this list existed.
STOP_JURIS = {"The", "This", "Such", "Any", "Each", "Either", "Both", "It", "Its",
              "State", "Province", "Commonwealth", "District", "Republic",
              "United", "Country", "Jurisdiction"}

# "incorporated under the laws of Delaware" states where a PARTY was formed,
# not which law governs the contract. Reject matches in that context.
INCORP = re.compile(
    r"(?i)(incorporat\w*|organiz\w*|organis\w*|existing|formed|created|"
    r"chartered|duly)\b[^.]{0,40}$")



def extract_regex(text: str) -> dict:
    """Effective date lives in the preamble; governing law almost always lives
    in the final 'Miscellaneous' section. The brief searched only text[:20000]
    for both, which structurally cannot find most governing-law clauses."""
    head = text[:20000]
    out: dict = {}

    for p in DATE_PATTERNS:
        m = re.search(p, head)
        if m:
            out["effective_date"] = m.group(1).strip().rstrip(",")
            out["date_source"] = "anchored"
            break
    else:
        m = DATE_BARE.search(text[:DATE_BARE_WINDOW])
        if m:
            out["effective_date"] = m.group(1).strip().rstrip(",")
            out["date_source"] = "bare"

    best, best_score = None, -99
    n = max(len(text), 1)
    heads = [m.start() for m in LAW_HEADING.finditer(text)]
    for m in re.finditer(LAW_CANDIDATE, text):
        val = m.group(1).strip().rstrip(",").rstrip(".")
        if val.split()[0] in STOP_JURIS:
            continue
        before = text[max(0, m.start() - 60):m.start()]
        if INCORP.search(before):
            continue                        # party incorporation, not governing law
        score = 0.0
        if any(0 <= m.start() - h <= 400 for h in heads):
            score += 3                      # sits under a Governing Law heading
        score += 2 * (m.start() / n)        # later in the document is better
        if LAW_VERB.search(before):
            score += 1
        if score > best_score:
            best, best_score = val, score
    if best:
        out["governing_law"] = best

    out["defined_terms"] = list(dict.fromkeys(ALIAS.findall(head)))[:10]
    return out


# --------------------------------------------------------------------------
# 4b. Zero-shot contract type
# --------------------------------------------------------------------------
LABELS_TUTORIAL = {
    "non-disclosure agreement": "NDA", "master services agreement": "MSA",
    "data processing agreement": "DPA", "statement of work": "SOW",
    "employment agreement": "Employment", "software license agreement": "License",
    "lease agreement": "Lease", "supply agreement": "Supply",
    "distribution agreement": "Distribution",
}

LABELS_CORPUS = {
    "distribution agreement": "Distribution", "maintenance agreement": "Maintenance",
    "strategic alliance agreement": "Strategic Alliance",
    "sponsorship agreement": "Sponsorship", "development agreement": "Development",
    "license agreement": "License", "endorsement agreement": "Endorsement",
    "services agreement": "Services", "intellectual property agreement": "IP",
    "branding agreement": "Branding", "supply agreement": "Supply",
    "hosting agreement": "Hosting", "outsourcing agreement": "Outsourcing",
    "franchise agreement": "Franchise", "agency agreement": "Agency",
    "cooperation agreement": "Cooperation",
    "manufacturing agreement": "Manufacturing",
    "joint filing agreement": "Joint Filing", "reseller agreement": "Reseller",
    "transportation agreement": "Transportation", "marketing agreement": "Marketing",
    "promotion agreement": "Promotion", "collaboration agreement": "Collaboration",
    "consulting agreement": "Consulting", "joint venture agreement": "Joint Venture",
    "affiliate agreement": "Affiliate",
}

CONF_THRESHOLD = 0.45
CLS_CHARS = 1500


def classify_all(docs: list[dict], labels: dict, device: str) -> dict:
    from transformers import pipeline

    local = LOCAL_MODELS / "distilbert-mnli"
    clf = pipeline("zero-shot-classification",
                   model=str(local) if local.exists()
                   else "typeform/distilbert-base-uncased-mnli", device=device)
    names = list(labels)
    out = {}
    for i, d in enumerate(docs):
        r = clf(d["text"][:CLS_CHARS], candidate_labels=names)
        label, score = r["labels"][0], r["scores"][0]
        out[d["doc_id"]] = {
            "type": labels[label] if score >= CONF_THRESHOLD else "Unknown",
            "confidence": round(float(score), 3),
            "raw_label": label,
        }
        if i % 50 == 0:
            print(f"    classify {i}/{len(docs)}", flush=True)
    del clf
    gc.collect()
    return out


NER_CHARS = 3000
GENERIC_ORG = {"agreement", "company", "parties", "party", "exhibit", "the company"}

# EDGAR exhibits open with filing boilerplate ("EX-10.5 5 a05-19878_1ex10d5.htm
# MATERIAL CONTRACTS ... CONFIDENTIAL TREATMENT REQUESTED"), so a fixed
# text[:3000] window can miss the party recital entirely - two of the first
# three documents returned no parties at all before this.
# Tried strongest-first. A bare "among"/"between" matches ordinary prose all
# through a contract, which sent one window 20k chars past the recital.
PARTY_ANCHORS = [
    re.compile(r"(?i)\bby\s+and\s+between\b"),
    re.compile(r"(?i)\bby\s+and\s+among\b"),
    re.compile(r"(?i)\bentered\s+into\s+(?:by\s+)?(?:and\s+)?between\b"),
    re.compile(r"(?i)\bthis\s+agreement[^.]{0,80}?\bbetween\b"),
]


def preamble_window(text: str, size: int = NER_CHARS) -> str:
    """Centre the window on the party recital rather than on character 0."""
    for rx in PARTY_ANCHORS:
        m = rx.search(text[:40000])
        if m:
            return text[max(0, m.start() - 400):max(0, m.start() - 400) + size]
    return text[:size]


def _ollama(prompt: str, max_tokens: int = 40) -> str:
    r = httpx.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0.0, "num_ctx": OLLAMA_CTX,
                    "num_predict": max_tokens},
    }, timeout=180.0)
    r.raise_for_status()
    return r.json()["response"].strip()


def classify_all_ollama(docs: list[dict], labels: dict) -> dict:
    """Fallback for classify_all when the NLI weights cannot be fetched.

    This is still zero-shot classification - the model has seen none of these
    contracts and is given only a label list - but it is instruction-following
    rather than NLI entailment, so the two are not interchangeable and the
    substitution is recorded in metadata.json.
    """
    names = sorted(set(labels.values()))
    menu = ", ".join(names)
    out = {}
    for i, d in enumerate(docs):
        prompt = (f"Classify this contract into exactly one category.\n"
                  f"Categories: {menu}\n"
                  f"Reply with the category name alone, or Unknown if none fit.\n\n"
                  f"Contract excerpt:\n{d['text'][:CLS_CHARS]}\n\nCategory:")
        try:
            raw = _ollama(prompt)
        except Exception as e:
            print(f"    ollama failed on {d['doc_id'][:36]}: {type(e).__name__}")
            out[d["doc_id"]] = {"type": "Unknown", "confidence": None, "raw_label": None}
            continue
        first = raw.splitlines()[0].strip().strip(".\"'")
        match = next((n for n in names if n.lower() == first.lower()), None)
        if match is None:                     # accept a substring hit
            match = next((n for n in names if n.lower() in first.lower()), None)
        out[d["doc_id"]] = {
            "type": match or "Unknown",
            "confidence": None,               # no calibrated score from generation
            "raw_label": first[:60],
        }
        if i % 50 == 0:
            print(f"    classify(ollama) {i}/{len(docs)}", flush=True)
    return out


def parties_all_ollama(docs: list[dict]) -> dict:
    out = {}
    for i, d in enumerate(docs):
        # Do NOT offer an explicit "reply NONE if unsure" escape hatch. Measured:
        # with that clause qwen2.5:3b answered NONE for documents whose parties
        # were plainly present in the window; without it, the same model and the
        # same window extracted them correctly. Same failure mode as Day 1 F1 -
        # naming the easy way out makes a small model take it.
        prompt = ("Read the contract text below. Which companies are entering "
                  "into this agreement?\nList their names, one per line, maximum "
                  "5. Output only names.\n\n"
                  f"{preamble_window(d['text'])}\n\nCompanies:")
        try:
            raw = _ollama(prompt, max_tokens=120)
        except Exception:
            out[d["doc_id"]] = []
            continue
        names, seen = [], set()
        NOISE = ("none", "not applicable", "n/a", "here are", "there are",
                 "companies:", "company names")
        # The model sometimes separates names with <br> instead of newlines,
        # which collapses them into one over-long line the length filter drops.
        raw = re.sub(r"(?i)<br\s*/?>|\s+\|\s+", "\n", raw)
        for line in raw.splitlines():
            w = line.strip().strip("`").lstrip("-*0123456789. ").strip().strip('"')
            k = w.lower()
            if not w or len(w) < 4 or len(w) > 80:
                continue
            if k in GENERIC_ORG or k in seen or any(n in k for n in NOISE):
                continue
            seen.add(k)
            names.append(w)
        out[d["doc_id"]] = names[:5]
        if i % 50 == 0:
            print(f"    parties(ollama) {i}/{len(docs)}", flush=True)
    return out


# --------------------------------------------------------------------------
# 4c. NER parties
# --------------------------------------------------------------------------
def parties_all(docs: list[dict], device: str) -> dict:
    from transformers import pipeline

    local = LOCAL_MODELS / "bert-base-NER"
    ner = pipeline("token-classification",
                   model=str(local) if local.exists() else "dslim/bert-base-NER",
                   aggregation_strategy="simple", device=device)
    out = {}
    for i, d in enumerate(docs):
        try:
            ents = ner(preamble_window(d["text"]))
        except Exception as e:                    # keep going; record the gap
            print(f"    NER failed on {d['doc_id'][:40]}: {type(e).__name__}")
            out[d["doc_id"]] = []
            continue
        seen, orgs = set(), []
        for e in ents:
            if e["entity_group"] != "ORG":
                continue
            w = e["word"].strip(" .,;:")
            k = w.lower()
            if len(w) > 3 and k not in seen and k not in GENERIC_ORG:
                seen.add(k)
                orgs.append(w)
        out[d["doc_id"]] = orgs[:5]
        if i % 50 == 0:
            print(f"    ner {i}/{len(docs)}", flush=True)
    del ner
    gc.collect()
    return out


# --------------------------------------------------------------------------
def enrich_chunks(meta: dict) -> None:
    """Denormalise metadata onto chunks so vector search can filter on it."""
    chunks = json.loads((INTERIM / "chunks_structural.json").read_text())
    keep = ("effective_date", "governing_law", "contract_type", "type_confidence",
            "contract_type_tutorial", "parties", "defined_terms")
    for c in chunks:
        m = meta.get(c["doc_id"], {})
        for k in keep:
            c[k] = m.get(k)
    (INTERIM / "chunks_enriched.json").write_text(json.dumps(chunks, indent=2))
    print(f"Enriched {len(chunks)} chunks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N docs (smoke test)")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--skip-models", action="store_true", help="regex only")
    ap.add_argument("--backend", choices=["auto", "nli", "ollama"], default="auto",
                    help="auto: use the local NLI weights if present, else Ollama")
    ap.add_argument("--from-metadata", action="store_true",
                    help="skip extraction; enrich chunks from the committed metadata.json (CI)")
    args = ap.parse_args()

    if args.from_metadata:
        enrich_chunks(json.loads((INTERIM / "metadata.json").read_text()))
        return 0

    docs = json.loads((INTERIM / "documents.json").read_text())
    if args.limit:
        docs = docs[:args.limit]
    print(f"{len(docs)} documents")

    print("  phase 1/4: regex")
    meta = {d["doc_id"]: extract_regex(d["text"]) for d in docs}

    backend = args.backend
    if backend == "auto":
        w = LOCAL_MODELS / "distilbert-mnli" / "model.safetensors"
        backend = "nli" if (w.exists() and w.stat().st_size > 250_000_000) else "ollama"
        print(f"  backend: {backend} (auto-selected)")

    if not args.skip_models:
        if backend == "nli":
            print("  phase 2/4: zero-shot NLI (tutorial labels)")
            tut = classify_all(docs, LABELS_TUTORIAL, args.device)
            print("  phase 3/4: zero-shot NLI (corpus labels)")
            cor = classify_all(docs, LABELS_CORPUS, args.device)
            print("  phase 4/4: NER parties (bert-base-NER)")
            par = parties_all(docs, args.device)
        else:
            print("  phase 2/4: zero-shot via Ollama (tutorial labels)")
            tut = classify_all_ollama(docs, LABELS_TUTORIAL)
            print("  phase 3/4: zero-shot via Ollama (corpus labels)")
            cor = classify_all_ollama(docs, LABELS_CORPUS)
            print("  phase 4/4: party extraction via Ollama")
            par = parties_all_ollama(docs)
    else:
        tut = cor = {d["doc_id"]: {"type": "Unknown", "confidence": 0.0} for d in docs}
        par = {d["doc_id"]: [] for d in docs}

    for d in docs:
        i = d["doc_id"]
        meta[i].update({
            "source": d["source"],
            "title_contract_type": d.get("title_contract_type"),
            "contract_type_tutorial": tut[i]["type"],
            "type_confidence_tutorial": tut[i]["confidence"],
            "contract_type": cor[i]["type"],           # corpus labels are the default
            "type_confidence": cor[i]["confidence"],
            "parties": par[i],
        })

    for v in meta.values():
        v["extraction_backend"] = "regex-only" if args.skip_models else backend
    (INTERIM / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote metadata for {len(meta)} docs")

    enrich_chunks(meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
