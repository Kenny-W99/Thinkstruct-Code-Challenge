import { useCallback, useEffect, useRef, useState } from "react";
import { api as defaultApi, type Api } from "./api/client";
import type { Mode, RerankChoice } from "./api/types";
import { FilterPanel } from "./components/FilterPanel";
import { Intro } from "./components/Intro";
import { Masthead } from "./components/Masthead";
import { PatentView } from "./components/PatentView";
import { ResultList } from "./components/ResultList";
import { SearchBar } from "./components/SearchBar";
import { Segmented } from "./components/Segmented";
import { useResource } from "./hooks/useResource";
import { useSearch } from "./hooks/useSearch";
import { fromParams, toParams, type SearchState } from "./lib/query";

// Must match RERANK_AUTO_MAX_WORDS in src/patent_search/config.py.
export const RERANK_AUTO_MAX_WORDS = 40;
const TOP_K_OPTIONS = [10, 20, 50];

function readUrl(): SearchState {
  return fromParams(new URLSearchParams(window.location.search));
}

function willRerank(s: SearchState, rerankerLoaded: boolean): boolean {
  if (s.rerank === "on") return true;
  return s.rerank === "auto" && rerankerLoaded && s.q.trim().split(/\s+/).length <= RERANK_AUTO_MAX_WORDS;
}

export default function App({ client = defaultApi }: { client?: Api }) {
  const [form, setForm] = useState<SearchState>(readUrl);
  const [open, setOpen] = useState<{ doc: string; anchor?: string } | null>(null);
  const { run, reset, status, data, error, pending } = useSearch(client.search);
  const health = useResource("health", client.health);
  const facets = useResource("facets", client.facets);
  const hasSearched = useRef(false);

  /** Put the state in the address bar and run it (an empty query clears the results). */
  const go = useCallback(
    (next: SearchState, history: "push" | "replace") => {
      const qs = toParams(next).toString();
      window.history[history === "push" ? "pushState" : "replaceState"](null, "", qs ? `?${qs}` : window.location.pathname);
      hasSearched.current = Boolean(next.q.trim());
      if (hasSearched.current) run(next);
      else reset();
    },
    [run, reset],
  );

  useEffect(() => {
    const initial = readUrl();
    if (initial.q.trim()) go(initial, "replace");
    const onPop = () => {
      const s = readUrl();
      setForm(s);
      hasSearched.current = Boolean(s.q.trim());
      if (hasSearched.current) run(s);
      else reset();
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
    // Once on mount: read the URL and listen for back/forward.
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Filters and options re-run the current search right away. */
  const onOptions = (next: SearchState) => {
    setForm(next);
    if (hasSearched.current && next.q.trim()) go(next, "replace");
  };

  const modes = health.data?.modes ?? ["hybrid", "bm25", "dense"];
  const rerankerLoaded = Boolean(health.data?.reranker);
  const loading = status === "loading";
  const closeDrawer = useCallback(() => setOpen(null), []);

  return (
    <div className="app">
      <Masthead health={health.data} error={health.error} />

      <div className="query">
        <SearchBar
          value={form.q}
          onChange={(q) => setForm({ ...form, q })}
          onSubmit={() => go(form, "push")}
          onExample={(q) => {
            const next = { ...form, q };
            setForm(next);
            go(next, "push");
          }}
          busy={loading}
        />
        <div className="options">
          <Segmented<Mode>
            label="Match by"
            value={form.mode}
            onChange={(mode) => onOptions({ ...form, mode })}
            options={[
              { value: "hybrid", label: "Both", title: "Keywords and meaning, merged (recommended)", disabled: !modes.includes("hybrid") },
              { value: "bm25", label: "Keywords", title: "BM25 only" },
              { value: "dense", label: "Meaning", title: "Embeddings only", disabled: !modes.includes("dense") },
            ]}
          />
          <Segmented<RerankChoice>
            label="Phase 2 rerank"
            value={form.rerank}
            onChange={(rerank) => onOptions({ ...form, rerank })}
            options={[
              { value: "auto", label: "Auto", title: `Rerank queries of ${RERANK_AUTO_MAX_WORDS} words or fewer` },
              { value: "on", label: "On", disabled: !rerankerLoaded },
              { value: "off", label: "Off" },
            ]}
          />
          <label className="topk">
            <span>Show</span>
            <select value={form.topK} onChange={(e) => onOptions({ ...form, topK: Number(e.target.value) })}>
              {TOP_K_OPTIONS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <p className="options-hint">
            Auto reranks queries of {RERANK_AUTO_MAX_WORDS} words or fewer. In testing, reranking helped short
            descriptions but hurt long claim-style queries.
          </p>
        </div>
      </div>

      <div className="layout">
        <FilterPanel state={form} facets={facets.data} onChange={onOptions} />

        <main className={`results ${loading && data ? "is-stale" : ""}`} aria-busy={loading}>
          {loading && (
            <div className="progress" role="status">
              <span className="progress-bar" />
              {pending && willRerank(pending, rerankerLoaded)
                ? "Searching… phase 2 reads the top 30 patents with a cross-encoder (a few seconds on CPU)."
                : "Searching…"}
            </div>
          )}
          {error && (
            <p className="error-banner" role="alert">
              {error}
            </p>
          )}
          {data ? (
            <ResultList
              response={data}
              totalPatents={facets.data?.n_patents ?? null}
              onOpen={(doc, anchor) => setOpen({ doc, anchor })}
            />
          ) : loading ? (
            <div className="skeletons" aria-hidden>
              {[0, 1, 2].map((i) => (
                <div key={i} className="skeleton" />
              ))}
            </div>
          ) : (
            !error && <Intro />
          )}
        </main>
      </div>

      <PatentView doc={open?.doc ?? null} anchor={open?.anchor} load={client.patent} onClose={closeDrawer} />
    </div>
  );
}
