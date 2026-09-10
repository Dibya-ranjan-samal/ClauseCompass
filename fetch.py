"""Task 1 - collect real EX-10.x contracts from SEC EDGAR full-text search.

Replaces the manual browser Cmd+S loop. Queries the same index that
https://www.sec.gov/edgar/search/ uses, keeps only EX-10.x exhibits, and
downloads them to data/raw/ as contract_01.html, contract_02.html, ...
"""

import os
import re
import sys
import time
from pathlib import Path

import httpx

# SEC requires a declared contact in the User-Agent for automated access and
# rate-limits or blocks requests without one. Set SEC_CONTACT to an email
# address you control:  export SEC_CONTACT="you@example.com"
CONTACT = os.environ.get("SEC_CONTACT", "").strip()
UA = f"ClauseCompass research {CONTACT}"
SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data"

QUERY = '"master services agreement"'
TARGET = 20            # how many contracts we want
PAGE = 10              # EDGAR full-text search page size
MAX_PAGES = 40         # safety stop on pagination
MIN_BYTES = 50_000     # below this it is a stub, not a contract
MAX_BYTES = 2_000_000  # above this it is usually a whole registration statement
SLEEP = 0.15           # SEC fair-access: stay well under 10 req/sec

raw_dir = Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

# Only actual executed-contract exhibits. EX-99.x (press releases),
# EX-101.x (XBRL), EX-4.x (securities) and cover forms are all rejected.
EX10 = re.compile(r"^EX-10(\.\d+)*$", re.IGNORECASE)

# A real contract body has this vocabulary; a cover page or press release does not.
MARKERS = ("whereas", "shall", "agreement", "section")


def search(client: httpx.Client, offset: int) -> list[dict]:
    r = client.get(
        SEARCH_URL,
        params={"q": QUERY, "from": offset, "hits": PAGE},
        headers={"User-Agent": UA, "Accept": "application/json"},
    )
    r.raise_for_status()
    return r.json().get("hits", {}).get("hits", [])


def doc_url(hit: dict) -> str | None:
    """Build the Archives URL from the hit id: '<accession>:<filename>'."""
    if ":" not in hit["_id"]:
        return None
    accession, filename = hit["_id"].split(":", 1)
    cik = hit["_source"]["ciks"][0].lstrip("0")
    return f"{ARCHIVE}/{cik}/{accession.replace('-', '')}/{filename}"


def looks_like_contract(html: str) -> bool:
    low = html.lower()
    return sum(m in low for m in MARKERS) >= 3


def main() -> int:
    # Fail loudly rather than sending SEC an empty contact, which gets the IP
    # rate-limited or blocked partway through a run.
    if not CONTACT:
        print('SEC_CONTACT is not set. SEC requires a real contact address in the\n'
              'User-Agent for automated access. Set it first:\n'
              '  export SEC_CONTACT="you@example.com"', file=sys.stderr)
        return 2

    saved = 0
    seen_ciks: set[str] = set()   # one contract per filer keeps the corpus diverse
    seen_docs: set[str] = set()   # same exhibit is often attached to several filings
    skipped = {"not_ex10": 0, "dup_filer": 0, "too_small": 0, "too_big": 0,
               "not_contract": 0, "http_error": 0}

    with httpx.Client(timeout=45.0, follow_redirects=True) as client:
        for page in range(MAX_PAGES):
            if saved >= TARGET:
                break
            try:
                hits = search(client, page * PAGE)
            except httpx.HTTPError as e:
                print(f"  search page {page} failed: {e}", file=sys.stderr)
                break
            if not hits:
                break

            for hit in hits:
                if saved >= TARGET:
                    break
                src = hit["_source"]
                ftype = (src.get("file_type") or "").strip()

                if not EX10.match(ftype):
                    skipped["not_ex10"] += 1
                    continue
                if not src.get("ciks"):
                    continue
                cik = src["ciks"][0]
                if cik in seen_ciks:
                    skipped["dup_filer"] += 1
                    continue

                url = doc_url(hit)
                if not url or url in seen_docs:
                    continue
                seen_docs.add(url)

                time.sleep(SLEEP)
                try:
                    resp = client.get(url, headers={"User-Agent": UA})
                    resp.raise_for_status()
                except httpx.HTTPError as e:
                    skipped["http_error"] += 1
                    print(f"  skip {ftype}: fetch failed ({e.__class__.__name__})")
                    continue

                html = resp.text
                n = len(resp.content)
                if n < MIN_BYTES:
                    skipped["too_small"] += 1
                    continue
                if n > MAX_BYTES:
                    skipped["too_big"] += 1
                    continue
                if not looks_like_contract(html):
                    skipped["not_contract"] += 1
                    continue

                saved += 1
                seen_ciks.add(cik)
                out = raw_dir / f"contract_{saved:02d}.html"
                out.write_text(html, encoding="utf-8")

                name = (src.get("display_names") or ["?"])[0].split("  (")[0]
                print(f"  contract_{saved:02d}.html  {n/1024:7.0f} KB  {ftype:<9} "
                      f"{src.get('file_date','?')}  {name[:42]}")

            time.sleep(SLEEP)

    print(f"\nSaved {saved} contracts to {raw_dir}/")
    print("Skipped: " + ", ".join(f"{k}={v}" for k, v in skipped.items() if v))
    if saved < 15:
        print(f"\nWARNING: only {saved} contracts - Day 1 wants 15-20.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
