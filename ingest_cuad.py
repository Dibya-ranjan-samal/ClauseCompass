"""Task 1 - load CUAD (Contract Understanding Atticus Dataset) into our format.

CUAD: 510 real commercial contracts with ~13.8k clause annotations across 41
categories, labelled by law students and reviewed by attorneys. CC BY 4.0.
Cite: Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for Legal
Contract Review" (NeurIPS 2021).

We read CUAD_v1.json directly rather than using `datasets`. The Hugging Face
repo `theatticusproject/cuad-qa` ships only a loading script, and dataset
loading scripts are disabled in datasets>=3, so that path is dead. The raw
SQuAD-format JSON in `theatticusproject/cuad` is the durable source.

Download once:
  curl -sL -o data/external/CUAD_v1.json \
    https://huggingface.co/datasets/theatticusproject/cuad/resolve/main/CUAD_v1/CUAD_v1.json
"""

import hashlib
import json
import re
from pathlib import Path

SRC = Path("data/external/CUAD_v1.json")
OUT = Path("data/interim")
OUT.mkdir(parents=True, exist_ok=True)


MAX_ID = 120


def safe_doc_id(title: str) -> str:
    """CUAD titles are filenames. Keep them readable but filesystem/ID safe.

    Truncating alone is not safe: two Playboy contracts share their first 120
    characters and differ only in a trailing '1'/'2', so a plain truncation
    silently collapsed them and dropped a 117k-char contract. Long names get a
    hash of the full title appended, which is unique and stable across runs.
    """
    t = re.sub(r"\.txt$", "", title.strip())
    t = re.sub(r"[^A-Za-z0-9._-]+", "_", t).strip("_")
    if len(t) <= MAX_ID:
        return f"cuad_{t}"
    digest = hashlib.sha1(title.encode()).hexdigest()[:8]
    return f"cuad_{t[:MAX_ID - 9]}_{digest}"


def contract_type_from_title(title: str) -> str:
    """CUAD encodes the contract type at the end of the filename, e.g.
    'LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT'.

    This is dataset-provided metadata, independent of the contract body, so it
    serves as ground truth for evaluating the text-based classifier in Task 5.
    """
    t = re.sub(r"\.txt$", "", title.strip())
    tail = re.split(r"[-_]", t)[-1].strip()
    tail = re.sub(r"\d+$", "", tail).strip()          # 'Agreement2' -> 'Agreement'
    tail = re.sub(r"\s*AGREEMENT\s*$", "", tail, flags=re.I).strip()
    return tail.upper() or "UNKNOWN"


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC} - see the download command in this file's docstring")
        return 1

    data = json.loads(SRC.read_text())["data"]

    documents, annotations = [], []
    seen: set[str] = set()
    for entry in data:
        title = entry["title"]
        doc_id = safe_doc_id(title)
        if doc_id in seen:          # must not happen: would mean silent data loss
            raise SystemExit(f"duplicate doc_id {doc_id!r} from title {title!r}")
        seen.add(doc_id)

        # Every CUAD entry has exactly one paragraph holding the full contract.
        text = "\n".join(p["context"] for p in entry["paragraphs"])
        documents.append({
            "doc_id": doc_id,
            "text": text,
            "source": "cuad",
            "title": title,
            "title_contract_type": contract_type_from_title(title),
        })

        for para in entry["paragraphs"]:
            for qa in para["qas"]:
                # CUAD asks all 41 category questions of every contract, so most
                # are unanswerable. Those are gold for Day 5's refusal accuracy -
                # keep them, flagged, rather than dropping them.
                annotations.append({
                    "doc_id": doc_id,
                    "qa_id": qa["id"],
                    "question": qa["question"],
                    "category": qa["question"].split('"')[1]
                                if '"' in qa["question"] else None,
                    "is_impossible": bool(qa.get("is_impossible")),
                    "answers": qa["answers"],   # [{'text':…, 'answer_start': int}]
                })

    (OUT / "cuad_documents.json").write_text(json.dumps(documents, indent=2))
    (OUT / "cuad_annotations.json").write_text(json.dumps(annotations, indent=2))

    answerable = [a for a in annotations if not a["is_impossible"]]
    spans = sum(len(a["answers"]) for a in annotations)
    chars = sum(len(d["text"]) for d in documents)
    cats = {a["category"] for a in annotations if a["category"]}

    print(f"{len(documents)} contracts, {len(annotations)} annotations")
    print(f"  answerable: {len(answerable)}  unanswerable: {len(annotations)-len(answerable)}")
    print(f"  answer spans: {spans}   categories: {len(cats)}")
    print(f"  corpus size: {chars:,} chars (mean {chars//len(documents):,}/contract)")

    bad = [a for a in answerable
           if not all(isinstance(x.get("answer_start"), int) for x in a["answers"])]
    print(f"  answer_start integrity: {len(answerable)-len(bad)}/{len(answerable)} ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
