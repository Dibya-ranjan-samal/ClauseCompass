# 02 — Ingestion & Parsing Real Documents

The least glamorous module and the one that most often destroys an evaluation
silently. Every downstream metric inherits the errors made here.

---

## 1. What it is

**Ingestion** is turning source documents into clean text with stable character
positions. Two jobs:

1. Extract the words, discard the markup.
2. **Preserve character offsets** — or knowingly account for changing them.

The second job is the one people don't realise exists.

## 2. Why it exists

Real corpora do not arrive as clean text. Yours arrived as:

- **SEC EDGAR filings** — HTML written by 1990s–2020s filing software, using
  `<table>` for layout, full of `&nbsp;`, inline styles and page furniture.
- **CUAD** — a JSON file where each contract is plain text *plus annotations that
  reference character positions in that exact string*.

Those two sources demand opposite treatment, and understanding why is the point
of this module.

## 3. Intuition: offsets are a contract with the future

When CUAD says

> the governing-law answer starts at character **4,120** and runs **380**
> characters

that number is only meaningful relative to **one exact string**. Delete a
newline at position 100 and every offset after it is wrong by one. Collapse runs
of whitespace and they're wrong by thousands.

So the text you chunk must be **byte-identical** to the text the annotations were
computed against. Any normalisation you apply is a silent, corpus-wide corruption
of your ground truth.

This is the trap the Day 3 brief warns about, and the reason your project dodged
it: `ingest_cuad.py` writes CUAD's `context` field through **unchanged**, while
`parse_edgar_html()` normalises aggressively — because EDGAR documents have no
annotations to protect.

```
CUAD    ->  raw context, untouched      ->  offsets remain valid
EDGAR   ->  parsed + normalised         ->  no offsets exist to break
```

Two sources, two policies, one corpus. Getting this right is why 99% of your
answers landed inside a chunk that contained them.

## 4. Worked example — what HTML parsing actually removes

An EDGAR exhibit fragment:

```html
<table><tr><td><font style="font-size:10pt">7.2&nbsp;&nbsp;Indemnification.</font></td></tr>
<tr><td><font>The Supplier shall indemnify&nbsp;the Customer.</font></td></tr></table>
<script>trackPageView();</script>
```

Naive `.get_text()` on that yields:

```
7.2\xa0\xa0Indemnification.
The Supplier shall indemnify\xa0the Customer.
trackPageView();
```

Three separate problems:

1. **`trackPageView();`** — JavaScript is now "contract text". It will be
   chunked, embedded, and become retrievable.
2. **`\xa0`** — the non-breaking space. To Python this is **not whitespace**:
   `"\xa0".strip()` returns `"\xa0"`, and `"a\xa0b".split()` gives `["a\xa0b"]`.
   So it survives cleaning and reaches the tokeniser as a strange character.
3. **Table-per-line layout** — EDGAR uses tables for indentation, so `get_text()`
   emits a newline per cell. Raw output is mostly blank lines.

The fix, from `build_corpus.py`:

```python
soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
for tag in soup(["script", "style"]):
    tag.decompose()                       # (1) delete, don't just skip
text = soup.get_text("\n").replace("\xa0", " ")   # (2) real spaces
return "\n".join(l.strip() for l in text.split("\n") if l.strip())  # (3) drop blanks
```

Line by line:

- **`decompose()`** removes the tag *and its contents* from the tree. Compare
  `extract()` (removes and returns it) and `.string = ""` (leaves the tag). If you
  merely iterate and skip, the text still appears in `get_text()`.
- **`errors="ignore"`** — filings contain mojibake from decades of encoding
  changes. Failing the whole document over one bad byte is worse than dropping
  the byte.
- **`.replace("\xa0", " ")`** — before any stripping, or the strip won't work.
- **The final join** collapses the table-induced blank lines.

## 5. Choosing a parser

`BeautifulSoup(html, PARSER)` — the second argument matters more than it looks.

| Parser | Speed | Leniency | Use when |
|---|---|---|---|
| `lxml` | fastest | high — repairs broken markup | **real-world scraped HTML** |
| `html.parser` | slow | medium, stdlib only | no C dependencies allowed |
| `html5lib` | slowest | browser-exact | you need exact browser DOM semantics |

EDGAR HTML from 1997 is malformed in creative ways — unclosed tags, stray
attributes, mismatched nesting. `lxml` repairs it and moves on. This project uses
`lxml` for exactly that reason.

## 6. Selecting the right documents — filtering is part of ingestion

Ingestion isn't only *how* you parse, it's *what* you parse. `fetch.py` queries
the EDGAR full-text search API, and the reject counts are instructive:

| Filter | Rejected | Why it matters |
|---|---|---|
| `file_type` must match `EX-10.x` | **953** | Full-text search returns whatever file contains the phrase — mostly press releases (`EX-99.x`), XBRL data (`EX-101.x`), and cover forms |
| One contract per filer (CIK) | **158** | One company dominated the results — Synacor appeared **6× in the first 19 hits** |
| Size 50 KB – 2 MB | 15 | Under 50 KB is usually "incorporated by reference", not a contract |
| Must contain contract vocabulary | 4 | Cheap sanity check on `whereas`, `shall`, `agreement`, `section` |

Of roughly 1,150 hits examined, **83% were not contracts at all.** And without
the per-filer dedupe, the "20-contract corpus" would have been about six
companies wearing twenty hats.

**The transferable lesson:** corpus *diversity* matters more than corpus *size*.
Twenty contracts from twenty filers exercises more clause variation than a
hundred from six.

## 7. Verifying ingestion before you trust it

Never proceed on the assumption that parsing worked. Two checks, both cheap.

### Check 1 — length sanity

```python
MIN_CHARS = 2000
flag = "  <-- SUSPECT" if len(text) < MIN_CHARS else ""
```

A "parsed contract" of 300 characters means you saved a search-results page, or
the parser ate everything. All 20 of your EDGAR documents passed
(13,206–460,219 chars).

### Check 2 — offset alignment, the important one

For the annotated corpus, verify that annotations still point at their text.
Before writing a line of the golden-set generator, you measured:

```python
hit = [c for c in by_doc[doc_id]
       if c["char_start"] < answer_end and c["char_end"] > answer_start]
ok += any(norm(answer)[:120] in norm(c["text"]) for c in hit)
```

Result: **6,697 / 6,702 = 99%** of answerable annotations were found inside a
chunk that overlaps their span.

Note the shape of that assertion. It checks two things at once:

- **overlap** — the arithmetic lines up
- **containment** — the *text* survived

Overlap alone proves only that your bookkeeping is consistent. Containment
proves the answer is actually there. Module 05 develops this into a full
correctness argument.

## 8. Failure modes

| Failure | Symptom | Detect | Fix |
|---|---|---|---|
| Normalised annotated text | gold chunks don't contain their answers | assert `answer in chunk_text` | keep annotated text byte-exact |
| Whitespace collapse | *consistent small drift*; answers land one chunk early | spot-check 20 offsets by hand | never normalise before offset use |
| `script`/`style` not removed | JS becomes retrievable "clause text" | grep the parsed corpus for `function`, `{` | `decompose()` |
| `\xa0` survives | odd tokens, `.strip()` appears broken | `text.count("\xa0")` after cleaning | replace before stripping |
| Saved the wrong page | 300-char "contracts" | minimum-length assertion | the length check above |
| Saved `.mhtml` not `.html` | parser returns near-nothing | file-size and length checks | save as HTML only |
| Corpus concentration | eval looks good, generalises badly | count distinct filers/authors | dedupe by source entity |
| Encoding failure | exception mid-ingest, or mojibake | `errors="ignore"` plus a spot read | tolerate bad bytes, log counts |

**The pattern worth naming:** every failure in this table is *silent*. None
raises an exception. Each produces a corpus that looks fine and an evaluation
that is quietly meaningless. Ingestion is where you must replace trust with
assertions.

## 9. Compared with

| | Normalise aggressively | Preserve byte-exact |
|---|---|---|
| Good for | display, keyword search, embedding quality | anything with character offsets |
| Cost | destroys annotation alignment | keeps oddities that reach the tokeniser |
| Your usage | EDGAR (no annotations) | CUAD (13,823 annotated spans) |

| | HTML parsing | PDF extraction |
|---|---|---|
| Structure available | explicit tags (though often abused for layout) | none — only glyph positions |
| Reading order | mostly the DOM order | must be inferred from coordinates |
| Typical failure | layout tables, page furniture | column interleaving, ligatures, OCR errors |
| Offsets | stable once parsed | fragile; extraction order can vary by library |

Your project deliberately used the text sources and excluded OCR-quality
documents. That's a limitation stated in the README rather than a silent
exclusion — which is the right way to handle it.

## 10. Interview angle

**"Walk me through your ingestion pipeline."** — Weak answers say "I used
BeautifulSoup". A strong answer names the *decision*: two sources with opposite
requirements, because one carries character-level annotations. EDGAR gets
normalised because nothing depends on its offsets; CUAD is passed through
untouched because 13,823 annotations reference exact positions. Then give the
verification: 99% of annotations confirmed inside a containing chunk.

**"What's the most dangerous bug in a RAG pipeline?"** — A good answer is the
silent-corruption class, and ingestion is where it starts. Nothing throws;
retrieval still returns k results; the eval produces stable plausible numbers
that mean nothing. My defence was assertions rather than inspection: a minimum
length per document, and an overlap-*and*-containment check on every annotation.

**"How did you pick your documents?"** — Mention that 83% of full-text search
hits weren't contracts, and that deduping by filer took the corpus from ~6
companies to 20. Diversity over volume.

## 11. Test yourself

**Q1.** Why is `"\xa0".strip()` equal to `"\xa0"` rather than `""`, and why does
that matter for a RAG pipeline?

<details><summary>Answer</summary>

`str.strip()` removes characters Python classifies as whitespace. U+00A0
(non-breaking space) is *not* in that set, so it survives. It matters because
cleaning code that looks correct (`line.strip()`, `" ".join(text.split())`)
leaves these characters in place; they then reach the tokeniser as unusual
sub-word units, and can appear inside quotes you later try to string-match for
citation validation.

</details>

**Q2.** You collapse runs of whitespace in a corpus that has character-level
annotations. Describe precisely what the downstream symptom looks like — and why
it is worse than a crash.

<details><summary>Answer</summary>

Every offset after the first collapsed run shifts left by the number of removed
characters, and the shift *accumulates* through the document. So annotations point
at text slightly before their real target — increasingly so, deeper into the
document.

The symptom is a **consistent small drift**: gold chunks are often the neighbour
of the correct chunk rather than the correct one. Metrics drop by a plausible
amount rather than to zero, so it reads as "retrieval needs tuning" instead of
"ground truth is corrupt". A crash would be *better*, because you'd fix it in
minutes instead of optimising against a broken target for days.

</details>

**Q3.** Why `decompose()` rather than iterating over tags and skipping `script`
elements?

<details><summary>Answer</summary>

Skipping during iteration doesn't change the parse tree — a later `get_text()`
call walks the tree afresh and includes the script contents. `decompose()`
removes the node and its children from the tree entirely, so any subsequent text
extraction cannot see it. The general principle: mutate the source of truth, not
one traversal of it.

</details>

**Q4.** Your EDGAR search returns 1,150 hits and you keep 20. A colleague says
you're wasting 98% of your data. Respond.

<details><summary>Answer</summary>

They aren't data for this task. 953 of the rejects were the wrong *document type*
— press releases announcing that a deal happened (`EX-99.x`), XBRL financial
tables, cover forms — none of which contain contract clauses. Keeping them would
add retrievable noise that competes with real clauses at query time.

A further 158 were duplicate filers. Keeping those would concentrate the corpus
on a handful of companies, so the eval would measure performance on six drafting
styles while claiming twenty. The filtering isn't waste, it's the difference
between a corpus and a pile of files.

</details>

**Q5.** Harder. You inherit a RAG system with an eval set built from a PDF
corpus. Retrieval metrics look reasonable but answer quality is poor. Give three
ingestion-level hypotheses and the cheapest check for each.

<details><summary>Answer</summary>

1. **Reading-order corruption.** Multi-column PDFs extracted column-interleaved,
   so sentences are spliced from two columns. *Check:* read 5 random chunks and
   see whether the prose is coherent; grep for sentences that end mid-clause and
   resume in an unrelated topic.
2. **Offset/annotation drift.** The gold spans were computed against a different
   extraction than the one indexed. *Check:* assert the gold answer string
   appears inside its gold chunk — this is the overlap-AND-containment test; if
   containment fails while overlap passes, the extractions differ.
3. **Page furniture embedded in chunks.** Headers, footers and confidentiality
   legends repeat on every page and end up inside otherwise-good chunks,
   inflating similarity between unrelated parts of the same document. *Check:*
   count the most frequent repeated lines across the corpus; measure pairwise
   similarity of random chunks *within* a document (in this project, exactly this
   produced pairs at 0.94–0.99 similarity — module 04).

Notice all three produce plausible retrieval metrics: the gold chunk can be
retrieved correctly while its *contents* are garbled, which is precisely why
retrieval metrics alone are insufficient and module 15 exists.

</details>

---

**Previous:** [01 — Why RAG exists](01-why-rag-exists.md) ·
**Next:** [03 — Chunking fundamentals](03-chunking-fundamentals.md)
