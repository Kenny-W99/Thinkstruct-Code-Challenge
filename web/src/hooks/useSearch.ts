import { useCallback, useEffect, useRef, useState } from "react";
import type { SearchResponse } from "../api/types";
import type { SearchState } from "../lib/query";

export type SearchFn = (state: SearchState, signal: AbortSignal) => Promise<SearchResponse>;

export interface SearchStatus {
  status: "idle" | "loading" | "done" | "error";
  data: SearchResponse | null; // the last good answer (kept on screen while the next one loads)
  error: string | null;
  pending: SearchState | null; // what is being searched right now
}

const IDLE: SearchStatus = { status: "idle", data: null, error: null, pending: null };

/** Runs searches. Starting a new search cancels the one in flight, so a slow
 * old answer (e.g. a 3 s reranked query) can never replace a newer one. */
export function useSearch(searchFn: SearchFn) {
  const [state, setState] = useState<SearchStatus>(IDLE);
  const controller = useRef<AbortController | null>(null);

  const run = useCallback(
    (query: SearchState) => {
      controller.current?.abort();
      const mine = new AbortController();
      controller.current = mine;
      setState((s) => ({ ...s, status: "loading", error: null, pending: query }));
      searchFn(query, mine.signal).then(
        (data) => {
          if (!mine.signal.aborted) setState({ status: "done", data, error: null, pending: null });
        },
        (err: unknown) => {
          if (mine.signal.aborted) return;
          const message = err instanceof Error ? err.message : String(err);
          setState((s) => ({ ...s, status: "error", error: message, pending: null }));
        },
      );
    },
    [searchFn],
  );

  const reset = useCallback(() => {
    controller.current?.abort();
    setState(IDLE);
  }, []);

  useEffect(() => () => controller.current?.abort(), []);

  return { ...state, run, reset };
}
