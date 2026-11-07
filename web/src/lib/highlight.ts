export type QuoteMatch = { start: number; end: number; ratio: number };

/**
 * Locate `quote` inside `text` the way the backend validator does
 * (generation/answer.py quote_ratio: lowercase, whitespace collapsed). Returns
 * the range in the ORIGINAL text. An exact match has ratio 1; otherwise the
 * longest contiguous match is returned with ratio = its share of the quote,
 * provided it covers at least `minRatio` of it.
 */
export function findQuote(text: string, quote: string | null | undefined, minRatio = 0.5): QuoteMatch | null {
  if (!quote) return null;
  const q = quote.replace(/\s+/g, " ").trim().toLowerCase();
  if (!q) return null;

  // Normalised copy of text plus a map from each normalised char to its origin.
  let norm = "";
  const map: number[] = [];
  let prevSpace = true;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (/\s/.test(ch)) {
      if (prevSpace) continue;
      norm += " ";
      map.push(i);
      prevSpace = true;
    } else {
      norm += ch.toLowerCase();
      map.push(i);
      prevSpace = false;
    }
  }
  const toRange = (at: number, len: number, ratio: number) => ({
    start: map[at],
    end: map[at + len - 1] + 1,
    ratio,
  });

  const at = norm.indexOf(q);
  if (at >= 0) return toRange(at, q.length, 1);

  // Longest common substring, O(|q|·|norm|) with one rolling row.
  let best = 0;
  let bestEnd = 0;
  let prev = new Uint16Array(q.length + 1);
  let cur = new Uint16Array(q.length + 1);
  for (let j = 1; j <= norm.length; j++) {
    const c = norm[j - 1];
    for (let i = 1; i <= q.length; i++) {
      const v = q[i - 1] === c ? prev[i - 1] + 1 : 0;
      cur[i] = v;
      if (v > best) {
        best = v;
        bestEnd = j;
      }
    }
    [prev, cur] = [cur, prev];
  }
  const ratio = best / q.length;
  return ratio >= minRatio ? toRange(bestEnd - best, best, ratio) : null;
}
