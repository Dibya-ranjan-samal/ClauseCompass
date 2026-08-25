"""Task 2 - parse the raw EDGAR HTML exhibits into plain text."""

import json
from pathlib import Path

from bs4 import BeautifulSoup

raw_dir = Path("data/raw")
out_dir = Path("data/interim")
out_dir.mkdir(parents=True, exist_ok=True)

MIN_CHARS = 2000  # below this the parser stripped too much, or it is the wrong page

docs = []
suspect = []
for f in sorted(raw_dir.glob("*.html")):
    html = f.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    # EDGAR exhibits are table-heavy, so get_text leaves a lot of blank lines
    # and non-breaking spaces behind. Normalise both.
    text = text.replace("\xa0", " ")
    text = "\n".join(line.strip() for line in text.split("\n") if line.strip())
    docs.append({"doc_id": f.stem, "text": text})
    flag = "  <-- SUSPECT" if len(text) < MIN_CHARS else ""
    if flag:
        suspect.append(f.name)
    print(f"{f.name}: {len(text)} chars{flag}")

(out_dir / "documents.json").write_text(json.dumps(docs, indent=2))
print(f"\nSaved {len(docs)} documents to data/interim/documents.json")
if suspect:
    print(f"WARNING: {len(suspect)} doc(s) under {MIN_CHARS} chars: {', '.join(suspect)}")
