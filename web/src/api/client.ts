// Typed client for the REST API. Every call takes an optional AbortSignal so
// the UI can cancel a request whose answer is no longer wanted.

import { toParams, type SearchState } from "../lib/query";
import type { Facets, Health, Patent, SearchResponse } from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

interface ValidationItem {
  loc?: (string | number)[];
  msg?: string;
}

/** A readable message from an error body: FastAPI sends {detail: "..."} for our
 * 400/404s and {detail: [{loc, msg}, ...]} for 422 validation errors. */
export function errorMessage(status: number, body: unknown): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length) {
    return (detail as ValidationItem[])
      .map((d) => {
        const field = d.loc?.filter((x) => x !== "query").join(".");
        return field ? `${field}: ${d.msg ?? "invalid"}` : (d.msg ?? "invalid");
      })
      .join("; ");
  }
  return `Request failed (HTTP ${status})`;
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${BASE}${path}`, { signal, headers: { Accept: "application/json" } });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, errorMessage(res.status, body));
  }
  return (await res.json()) as T;
}

export function searchUrl(state: SearchState): string {
  return `/api/search?${toParams(state).toString()}`;
}

export const api = {
  health: (signal?: AbortSignal) => getJson<Health>("/api/health", signal),
  facets: (signal?: AbortSignal) => getJson<Facets>("/api/facets", signal),
  search: (state: SearchState, signal?: AbortSignal) => getJson<SearchResponse>(searchUrl(state), signal),
  patent: (doc: string, signal?: AbortSignal) => getJson<Patent>(`/api/patents/${encodeURIComponent(doc)}`, signal),
};

export type Api = typeof api;
