"""Task 3 - structural clause chunking.

Replaces Day 1's fixed-width character windows. Contracts are numbered, so
split on section boundaries and keep the section number and heading as
metadata. chunk.py is kept as the naive baseline to beat on Day 4.

Four corrections to the straightforward version, each one measured:

1. TWO heading patterns, not one. Requiring the heading to sit alone on its own
   line only covered 57% of documents; many contracts run the body inline after
   the number ("1. Services and Compensation. Consultant agrees to..."). Adding
   the inline form lifts structural coverage to ~70%.

2. THE PREAMBLE IS A CHUNK. Starting the first span at the first section match
   silently discarded everything before it - 1,832,665 chars, a mean of 6,008
   per document, holding the parties, recitals, dates and sometimes the
   governing law. Worse, 16.6% of CUAD's 13,823 answerable spans live there, so
   Day 3 recall would have been capped at 83.4% with no way to diagnose it.

3. SHORT SECTIONS MERGE FORWARD instead of being dropped. Dropping anything
   under MIN_CHARS loses text outright; merging keeps every character.

4. MAX_CHARS IS ENFORCED. split_long() only breaks on sentence boundaries, so a
   long table with no sentence punctuation passed through whole - the worst case
   was a 19,249-char "chunk" against a 4,000 limit.
"""

import json
import re
from collections import Counter
from pathlib import Path

# Heading alone on its line: "7.2 Indemnification". \s (not [ \t]) is deliberate -
# it also catches the number and heading split across two lines, which is common
# in text extracted from filing tables. Tightening it to [ \t] cost 12 points of
# structural coverage.
SECTION_OWN_LINE = re.compile(
    r"^\s*(?:(?:Section|Article|ARTICLE|SECTION)\s+)?"
    r"(\d+(?:\.\d+)*)\s*[.\)]?\s+"
    r"([A-Z][A-Za-z ,\-/]{2,80})?\s*[.:]?\s*$",
    re.MULTILINE,
)
# Heading inline, body follows:  "1. Services and Compensation. Consultant agrees..."
SECTION_INLINE = re.compile(
    r"^[ \t]*(?:(?:Section|Article|ARTICLE|SECTION)[ \t]+)?"
    r"(\d+(?:\.\d+){0,3})[ \t]*[.\)][ \t]+"
    r"([A-Z][A-Za-z0-9 ,\-/&']{2,60})[.:](?=[ \t\n])",
    re.MULTILINE,
)
PATTERNS = (SECTION_OWN_LINE, SECTION_INLINE)

MIN_CHARS, MAX_CHARS = 200, 4000
FALLBACK_SIZE, FALLBACK_STRIDE = 1000, 850
MIN_SECTIONS = 3


def _line_start(text: str, pos: int) -> int:
    return text.rfind("\n", 0, pos) + 1


def find_sections(text: str) -> list[tuple[int, str, str]]:
    """(start, section_number, heading) for every detected boundary.

    Deduped on the start of the line holding the section number, not on the raw
    match offset: the two patterns consume leading whitespace differently, so
    the same boundary would otherwise be recorded twice at different offsets.
    """
    found: dict[int, tuple[str, str]] = {}
    for rx in PATTERNS:
        for m in rx.finditer(text):
            key = _line_start(text, m.start(1))
            found.setdefault(key, (m.group(1), (m.group(2) or "").strip()))
    return [(s, n, h) for s, (n, h) in sorted(found.items())]


def hard_split(s: str, max_chars: int) -> list[str]:
    """Chop at max_chars, folding a runt tail back into the previous piece so we
    never emit a 1-character chunk (which a 4001-char input otherwise does)."""
    # Consequence: the merged piece can exceed max_chars by up to MIN_CHARS.
    # That is deliberate - slightly over the cap beats emitting a runt chunk.
    parts = [s[i:i + max_chars] for i in range(0, len(s), max_chars)]
    if len(parts) > 1 and len(parts[-1]) < MIN_CHARS:
        tail = parts.pop()          # pop first: parts[-2] would shift under us
        parts[-1] += tail
    return parts


def split_long(text: str, max_chars: int) -> list[str]:
    """Split on sentence boundaries, then hard-split anything still oversized."""
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.;])\s+", text):
        if len(cur) + len(sent) > max_chars and cur:
            parts.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        parts.append(cur)
    out: list[str] = []
    for p in parts:
        out.extend(hard_split(p, max_chars) if len(p) > max_chars else [p])
    return out


def merge_short(spans: list[tuple[int, int, str | None, str | None]],
                min_chars: int, text: str) -> list[tuple[int, int, str | None, str | None]]:
    """Fold spans below min_chars into the following span, keeping all text.
    A trailing short span merges backwards instead."""
    out: list[list] = []
    for start, end, num, head in spans:
        if out and len(text[out[-1][0]:out[-1][1]].strip()) < min_chars:
            out[-1][1] = end                      # extend previous, keep its label
        else:
            out.append([start, end, num, head])
    if len(out) > 1 and len(text[out[-1][0]:out[-1][1]].strip()) < min_chars:
        tail = out.pop()
        out[-1][1] = tail[1]
    return [tuple(x) for x in out]


def chunk_document(doc: dict) -> list[dict]:
    text = doc["text"]
    sections = find_sections(text)

    if len(sections) < MIN_SECTIONS:
        strategy = "fallback"
        spans = [(i, min(i + FALLBACK_SIZE, len(text)), None, None)
                 for i in range(0, len(text), FALLBACK_STRIDE)]
    else:
        strategy = "structural"
        spans = []
        # Fix 2: everything before the first section heading is the preamble.
        if sections[0][0] > 0:
            spans.append((0, sections[0][0], None, "PREAMBLE"))
        for i, (start, num, head) in enumerate(sections):
            end = sections[i + 1][0] if i + 1 < len(sections) else len(text)
            spans.append((start, end, num, head or None))

    # Applies to both strategies: the fallback path also produces a runt final
    # window (a 3-char tail) that must be folded in rather than dropped.
    spans = merge_short(spans, MIN_CHARS, text)

    chunks: list[dict] = []
    for idx, (start, end, num, heading) in enumerate(spans):
        body = text[start:end].strip()
        if not body:
            continue
        pieces = split_long(body, MAX_CHARS) if len(body) > MAX_CHARS else [body]
        for p_i, piece in enumerate(pieces):
            chunks.append({
                # idx keeps this unique: a section number can legitimately recur
                # in a contract (table of contents, exhibits, restated parts),
                # which collided 644 chunk_ids when the number alone was used.
                "chunk_id": f"{doc['doc_id']}::c{idx:03d}::s{num if num is not None else 'NA'}::p{p_i}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": piece,
                "section_number": num,
                "section_heading": heading,
                "char_start": start,
                "char_end": end,
                "part": p_i,
                "n_parts": len(pieces),
                "chunking_strategy": strategy,
                "is_preamble": heading == "PREAMBLE",
            })
    return chunks


def main() -> int:
    docs = json.loads(Path("data/interim/documents.json").read_text())
    all_chunks: list[dict] = []
    strat = Counter()
    strat_by_source: dict[str, Counter] = {}

    for d in docs:
        cs = chunk_document(d)
        all_chunks.extend(cs)
        s = cs[0]["chunking_strategy"] if cs else "empty"
        strat[s] += 1
        strat_by_source.setdefault(d["source"], Counter())[s] += 1

    ids = [c["chunk_id"] for c in all_chunks]
    dups = len(ids) - len(set(ids))
    Path("data/interim/chunks_structural.json").write_text(json.dumps(all_chunks, indent=2))

    lens = sorted(len(c["text"]) for c in all_chunks)
    n = len(all_chunks)
    print(f"{n} chunks from {len(docs)} docs")
    print(f"strategy: {dict(strat)} -> {100*strat['structural']//len(docs)}% structural")
    for src in sorted(strat_by_source):
        c = strat_by_source[src]
        print(f"  {src:<6} {dict(c)} -> {100*c['structural']//sum(c.values())}% structural")
    print(f"chunk chars - min {lens[0]}, median {lens[n//2]}, "
          f"p90 {lens[int(n*0.9)]}, max {lens[-1]}  (MAX_CHARS={MAX_CHARS})")
    print(f"over MAX_CHARS: {sum(1 for x in lens if x > MAX_CHARS)}")
    print(f"preamble chunks: {sum(1 for c in all_chunks if c['is_preamble'])}")
    print(f"with section heading: {sum(1 for c in all_chunks if c['section_heading'])}/{n}")
    print(f"duplicate chunk_ids: {dups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
