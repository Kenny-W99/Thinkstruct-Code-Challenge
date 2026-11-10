import { useEffect, useState } from "react";

export interface Resource<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

/** Load something once per `key` (null = load nothing). A changed key cancels the old request. */
export function useResource<T>(key: string | null, load: (signal: AbortSignal) => Promise<T>): Resource<T> {
  const [state, setState] = useState<Resource<T>>({ data: null, error: null, loading: key !== null });

  useEffect(() => {
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    const controller = new AbortController();
    setState({ data: null, error: null, loading: true });
    load(controller.signal).then(
      (data) => !controller.signal.aborted && setState({ data, error: null, loading: false }),
      (err: unknown) =>
        !controller.signal.aborted &&
        setState({ data: null, error: err instanceof Error ? err.message : String(err), loading: false }),
    );
    return () => controller.abort();
    // Re-run only when the key changes (`load` is expected to be stable for a key).
  }, [key]);

  return state;
}
