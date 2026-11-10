// Small pure helpers for showing patent text.

export interface Segment {
  text: string;
  hit: boolean;
}

/** Split text into plain and highlighted pieces. Spans are [start, end) character
 * offsets from the API; bad, overlapping or out-of-range spans are skipped. */
export function segments(text: string, spans: [number, number][], from = 0, to = text.length): Segment[] {
  const out: Segment[] = [];
  let pos = from;
  for (const [a, b] of [...spans].sort((x, y) => x[0] - y[0])) {
    if (a < pos || b > to || b <= a) continue;
    if (a > pos) out.push({ text: text.slice(pos, a), hit: false });
    out.push({ text: text.slice(a, b), hit: true });
    pos = b;
  }
  if (pos < to) out.push({ text: text.slice(pos, to), hit: false });
  return out;
}

export interface Window {
  from: number;
  to: number;
  clippedStart: boolean;
  clippedEnd: boolean;
}

/** A ~`width`-character window that starts a little before the first highlight,
 * snapped to word boundaries. */
export function snippetWindow(text: string, spans: [number, number][], width = 320): Window {
  if (text.length <= width) return { from: 0, to: text.length, clippedStart: false, clippedEnd: false };
  const first = spans.length ? Math.min(...spans.map((s) => s[0])) : 0;
  let from = Math.max(0, first - Math.floor(width / 4));
  let to = Math.min(text.length, from + width);
  from = Math.max(0, to - width);
  if (from > 0) {
    const space = text.indexOf(" ", from);
    if (space !== -1 && space < first) from = space + 1;
  }
  if (to < text.length) {
    const space = text.lastIndexOf(" ", to);
    if (space > from) to = space;
  }
  return { from, to, clippedStart: from > 0, clippedEnd: to < text.length };
}

const SMALL_WORDS = new Set(["a", "an", "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the", "to", "with"]);

/** "PNEUMATIC TIRE AND METHOD" -> "Pneumatic Tire and Method". Mixed-case titles are kept. */
export function displayTitle(title: string | null): string {
  if (!title) return "(untitled)";
  if (title !== title.toUpperCase()) return title;
  return title
    .toLowerCase()
    .split(/(\s+)/)
    .map((word, i) => (i > 0 && SMALL_WORDS.has(word) ? word : word.charAt(0).toUpperCase() + word.slice(1)))
    .join("");
}

/** DOM id for a passage inside the patent view: "Claim 7 (part 2/3)" -> "claim-7". */
export function anchorFor(label: string): string {
  const claim = /^Claim (\d+)/.exec(label);
  if (claim) return `claim-${claim[1]}`;
  const para = /^Description ¶(\d+)/.exec(label);
  if (para) return `para-${para[1]}`;
  return "abstract";
}

export function formatMs(ms: number): string {
  if (ms >= 1000) return `${(ms / 1000).toFixed(2)} s`;
  if (ms >= 10) return `${Math.round(ms)} ms`;
  return `${ms.toFixed(1)} ms`;
}

export function formatCount(n: number): string {
  return n.toLocaleString("en-US");
}

/** Model id -> short display name: "mixedbread-ai/mxbai-rerank-base-v1" -> "mxbai-rerank-base-v1". */
export function shortModel(name: string | null): string {
  return name ? (name.split("/").pop() ?? name) : "none";
}
