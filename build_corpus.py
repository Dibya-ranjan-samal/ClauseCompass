"""Task 2 - merge the EDGAR and CUAD corpora into one document store.

Every document carries a `source` field. Day 4 reports retrieval metrics per
source: EDGAR docs come from messy filing HTML while CUAD ships clean text, so
they are expected to behave differently and that difference is a finding, not
noise.

parse.py is left alone as the Day 1 EDGAR-only baseline; this script supersedes
it for the merged pipeline.
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup

RAW = Path("data/raw")
INTERIM = Path("data/interim")
MIN_CHARS = 2000


def parse_edgar_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n").replace("\xa0", " ")
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def main() -> int:
    docs = []

    for f in sorted(RAW.glob("*.html")):
        text = parse_edgar_html(f)
        if len(text) < MIN_CHARS:
            print(f"  SKIP {f.name}: only {len(text)} chars")
            continue
        docs.append({
            "doc_id": f.stem,
            "text": text,
            "source": "edgar",
            "title": f.stem,
            "title_contract_type": None,   # EDGAR filenames carry no type
        })

    cuad_path = INTERIM / "cuad_documents.json"
    if not cuad_path.exists():
        print("missing cuad_documents.json - run ingest_cuad.py first")
        return 1
    docs.extend(json.loads(cuad_path.read_text()))

    ids = [d["doc_id"] for d in docs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate doc_id across sources")

    (INTERIM / "documents.json").write_text(json.dumps(docs, indent=2))

    by_source: dict[str, int] = {}
    chars_by_source: dict[str, int] = {}
    for d in docs:
        by_source[d["source"]] = by_source.get(d["source"], 0) + 1
        chars_by_source[d["source"]] = chars_by_source.get(d["source"], 0) + len(d["text"])

    print(f"{len(docs)} documents total: {by_source}")
    for s in sorted(by_source):
        n, c = by_source[s], chars_by_source[s]
        print(f"  {s:<6} {n:>4} docs  {c:>12,} chars  mean {c//n:>7,}")
    print(f"  {'TOTAL':<6} {len(docs):>4} docs  {sum(chars_by_source.values()):>12,} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
