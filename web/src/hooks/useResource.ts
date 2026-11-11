import { useEffect, useState } from "react";

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Load something once per `key` (null = load nothing). A changed key cancels the old request.
 *
 * With `retryMs`, a failed load is tried again after that long until it works,
 * keeping the error visible meanwhile. The API can still be starting when the page
 * opens (the first Docker start builds the index for minutes, and nginx answers 502). */
export function useResource<T>(
  key: string | null,
  load: (signal: AbortSignal) => Promise<T>,
  retryMs?: number,
): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ data: null, error: null, loading: key !== null });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const controller = new AbortController();
    let retry: number | undefined;
    // A retry keeps the last error on screen instead of flashing back to "loading".
    setState((s) => (attempt === 0 ? { data: null, error: null, loading: true } : { ...s, loading: true }));
    load(controller.signal).then(
      (data) => !controller.signal.aborted && setState({ data, error: null, loading: false }),
      (err: unknown) => {
        if (controller.signal.aborted) return;
        setState({ data: null, error: err instanceof Error ? err.message : String(err), loading: false });
        if (retryMs) retry = window.setTimeout(() => setAttempt((n) => n + 1), retryMs);
      },
    );
    return () => {
      controller.abort();
      window.clearTimeout(retry);
    };
    // Re-run only when the key changes or a retry is due (`load` is expected to be stable for a key).
  }, [key, attempt]); // eslint-disable-line react-hooks/exhaustive-deps

  return state;
}
