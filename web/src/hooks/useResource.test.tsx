import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useResource } from "./useResource";

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

describe("useResource", () => {
  it("loads once per key", async () => {
    const load = vi.fn().mockResolvedValue("ok");
    const { result } = renderHook(() => useResource("k", load));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(result.current).toEqual({ data: "ok", error: null, loading: false });
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("without retryMs, a failure stays", async () => {
    const load = vi.fn().mockRejectedValue(new Error("HTTP 502"));
    const { result } = renderHook(() => useResource("k", load));
    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(result.current.error).toBe("HTTP 502");
    expect(load).toHaveBeenCalledTimes(1);
  });

  it("with retryMs, retries until the load works, keeping the error meanwhile", async () => {
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Error("HTTP 502"))
      .mockRejectedValueOnce(new Error("HTTP 502"))
      .mockResolvedValue("ok");
    const { result } = renderHook(() => useResource("k", load, 1000));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(result.current.error).toBe("HTTP 502");

    await act(() => vi.advanceTimersByTimeAsync(999));
    expect(load).toHaveBeenCalledTimes(1); // not before retryMs
    await act(() => vi.advanceTimersByTimeAsync(1));
    expect(result.current).toMatchObject({ data: null, error: "HTTP 502" }); // 2nd attempt failed too

    await act(() => vi.advanceTimersByTimeAsync(1000));
    expect(result.current).toEqual({ data: "ok", error: null, loading: false });
    expect(load).toHaveBeenCalledTimes(3);

    await act(() => vi.advanceTimersByTimeAsync(10_000));
    expect(load).toHaveBeenCalledTimes(3); // stops once it works
  });
});
