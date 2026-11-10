import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SearchResponse } from "../api/types";
import { DEFAULT_STATE } from "../lib/query";
import { useResource } from "./useResource";
import { useSearch, type SearchFn } from "./useSearch";

const answer = (query: string) => ({ query, results: [] }) as unknown as SearchResponse;

/** A search function whose calls we resolve by hand, in any order. */
function manualSearch() {
  const calls: { q: string; signal: AbortSignal; resolve: (r: SearchResponse) => void; reject: (e: Error) => void }[] = [];
  const fn: SearchFn = (state, signal) =>
    new Promise((resolve, reject) => calls.push({ q: state.q, signal, resolve, reject }));
  return { fn, calls };
}

describe("useSearch", () => {
  it("a slow old answer never replaces a newer one", async () => {
    const { fn, calls } = manualSearch();
    const { result } = renderHook(() => useSearch(fn));

    act(() => result.current.run({ ...DEFAULT_STATE, q: "first" }));
    act(() => result.current.run({ ...DEFAULT_STATE, q: "second" }));
    expect(calls[0].signal.aborted).toBe(true);
    expect(result.current.status).toBe("loading");

    await act(async () => calls[1].resolve(answer("second")));
    await act(async () => calls[0].resolve(answer("first"))); // arrives late
    expect(result.current.status).toBe("done");
    expect(result.current.data?.query).toBe("second");
  });

  it("keeps the last answer on screen while loading, and reports errors", async () => {
    const { fn, calls } = manualSearch();
    const { result } = renderHook(() => useSearch(fn));
    act(() => result.current.run({ ...DEFAULT_STATE, q: "one" }));
    await act(async () => calls[0].resolve(answer("one")));

    act(() => result.current.run({ ...DEFAULT_STATE, q: "two" }));
    expect(result.current.data?.query).toBe("one");
    expect(result.current.pending?.q).toBe("two");

    await act(async () => calls[1].reject(new Error("Query is empty.")));
    expect(result.current.status).toBe("error");
    expect(result.current.error).toBe("Query is empty.");
  });

  it("cancels the request when the component goes away", () => {
    const { fn, calls } = manualSearch();
    const { result, unmount } = renderHook(() => useSearch(fn));
    act(() => result.current.run({ ...DEFAULT_STATE, q: "x" }));
    unmount();
    expect(calls[0].signal.aborted).toBe(true);
  });
});

describe("useResource", () => {
  it("loads per key and cancels on key change", async () => {
    const load = vi.fn((signal: AbortSignal) => Promise.resolve(signal));
    const { result, rerender } = renderHook(({ k }) => useResource(k, load), { initialProps: { k: "a" as string | null } });
    await waitFor(() => expect(result.current.loading).toBe(false));
    const first = result.current.data!;
    rerender({ k: "b" });
    expect(first.aborted).toBe(true);
    await waitFor(() => expect(result.current.data).not.toBe(first));
    rerender({ k: null });
    expect(result.current).toEqual({ data: null, error: null, loading: false });
  });
});
