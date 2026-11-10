// The search form's state, and how it maps to a query string.
//
// The browser URL and the API request use the SAME parameter names
// (?q=...&cpc=B60B&cpc=B60C&top_k=10), so a search can be shared as a link and
// the address bar shows exactly what was sent to GET /api/search.

import type { Kind, Mode, RerankChoice, Strategy } from "../api/types";

export interface SearchState {
  q: string;
  cpc: string[];
  keyword: string[];
  title: string;
  titleExact: boolean;
  dateFrom: string; // YYYY-MM-DD or ""
  dateTo: string;
  mode: Mode;
  strategy: Strategy;
  kinds: Kind[]; // [] = all kinds
  topK: number;
  rerank: RerankChoice;
}

export const MODES: Mode[] = ["hybrid", "bm25", "dense"];
export const STRATEGIES: Strategy[] = ["pre", "post"];
export const KINDS: Kind[] = ["abstract", "claim", "description"];
export const RERANK_CHOICES: RerankChoice[] = ["auto", "on", "off"];
export const MAX_TOP_K = 100;

export const DEFAULT_STATE: SearchState = {
  q: "",
  cpc: [],
  keyword: [],
  title: "",
  titleExact: false,
  dateFrom: "",
  dateTo: "",
  mode: "hybrid",
  strategy: "pre",
  kinds: [],
  topK: 10,
  rerank: "auto",
};

const DATE = /^\d{4}-\d{2}-\d{2}$/;

function pick<T extends string>(value: string | null, allowed: readonly T[], fallback: T): T {
  return value !== null && (allowed as readonly string[]).includes(value) ? (value as T) : fallback;
}

function clean(values: string[]): string[] {
  return [...new Set(values.map((v) => v.trim()).filter(Boolean))];
}

/** State -> query string. Defaults and empty values are left out. */
export function toParams(s: SearchState): URLSearchParams {
  const p = new URLSearchParams();
  if (s.q.trim()) p.set("q", s.q.trim());
  clean(s.cpc).forEach((c) => p.append("cpc", c));
  clean(s.keyword).forEach((k) => p.append("keyword", k));
  if (s.title.trim()) {
    p.set("title", s.title.trim());
    if (s.titleExact) p.set("title_exact", "true");
  }
  if (s.dateFrom) p.set("date_from", s.dateFrom);
  if (s.dateTo) p.set("date_to", s.dateTo);
  if (s.mode !== DEFAULT_STATE.mode) p.set("mode", s.mode);
  if (s.strategy !== DEFAULT_STATE.strategy) p.set("strategy", s.strategy);
  s.kinds.forEach((k) => p.append("kinds", k));
  if (s.topK !== DEFAULT_STATE.topK) p.set("top_k", String(s.topK));
  if (s.rerank !== DEFAULT_STATE.rerank) p.set("rerank", s.rerank);
  return p;
}

/** Query string -> state. Anything unknown or malformed falls back to the default. */
export function fromParams(p: URLSearchParams): SearchState {
  const topK = Number.parseInt(p.get("top_k") ?? "", 10);
  const date = (key: string) => {
    const v = p.get(key) ?? "";
    return DATE.test(v) ? v : "";
  };
  return {
    q: p.get("q") ?? "",
    cpc: clean(p.getAll("cpc")),
    keyword: clean(p.getAll("keyword")),
    title: p.get("title") ?? "",
    titleExact: p.get("title_exact") === "true",
    dateFrom: date("date_from"),
    dateTo: date("date_to"),
    mode: pick(p.get("mode"), MODES, DEFAULT_STATE.mode),
    strategy: pick(p.get("strategy"), STRATEGIES, DEFAULT_STATE.strategy),
    kinds: KINDS.filter((k) => p.getAll("kinds").includes(k)),
    topK: Number.isFinite(topK) ? Math.min(Math.max(topK, 1), MAX_TOP_K) : DEFAULT_STATE.topK,
    rerank: pick(p.get("rerank"), RERANK_CHOICES, DEFAULT_STATE.rerank),
  };
}

/** How many filters are set (for the "Filters (3)" badge). */
export function activeFilterCount(s: SearchState): number {
  return (
    s.cpc.length +
    s.keyword.length +
    (s.title.trim() ? 1 : 0) +
    (s.dateFrom ? 1 : 0) +
    (s.dateTo ? 1 : 0) +
    (s.kinds.length ? 1 : 0)
  );
}

/** The same state with every filter cleared (query and search options kept). */
export function withoutFilters(s: SearchState): SearchState {
  return { ...s, cpc: [], keyword: [], title: "", titleExact: false, dateFrom: "", dateTo: "", kinds: [] };
}
