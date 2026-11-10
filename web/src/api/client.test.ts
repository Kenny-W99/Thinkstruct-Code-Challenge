import { afterEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_STATE } from "../lib/query";
import { ApiError, api, errorMessage, searchUrl } from "./client";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(status: number, body: unknown) {
  const fetchMock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("errorMessage", () => {
  it("uses a plain detail string (our 400 and 404 errors)", () => {
    expect(errorMessage(400, { detail: "Not a valid classification prefix: '60B'" })).toBe(
      "Not a valid classification prefix: '60B'",
    );
  });

  it("flattens FastAPI validation errors (422)", () => {
    const body = { detail: [{ loc: ["query", "top_k"], msg: "Input should be greater than or equal to 1" }] };
    expect(errorMessage(422, body)).toBe("top_k: Input should be greater than or equal to 1");
  });

  it("falls back to the status code", () => {
    expect(errorMessage(502, null)).toBe("Request failed (HTTP 502)");
  });
});

describe("api", () => {
  it("builds the search URL from the state", () => {
    expect(searchUrl({ ...DEFAULT_STATE, q: "spoke", cpc: ["B60B", "B60C"] })).toBe("/api/search?q=spoke&cpc=B60B&cpc=B60C");
  });

  it("passes the abort signal and returns the JSON", async () => {
    const fetchMock = stubFetch(200, { results: [] });
    const controller = new AbortController();
    await expect(api.search({ ...DEFAULT_STATE, q: "spoke" }, controller.signal)).resolves.toEqual({ results: [] });
    expect(fetchMock).toHaveBeenCalledWith("/api/search?q=spoke", expect.objectContaining({ signal: controller.signal }));
  });

  it("throws ApiError with the status and message", async () => {
    stubFetch(404, { detail: "Patent 1 is not in this dataset." });
    const err = await api.patent("1").catch((e: unknown) => e);
    expect(err).toBeInstanceOf(ApiError);
    expect(err).toMatchObject({ status: 404, message: "Patent 1 is not in this dataset." });
  });

  it("encodes patent numbers in the path", async () => {
    const fetchMock = stubFetch(200, {});
    await api.patent("a/b");
    expect(fetchMock.mock.calls[0][0]).toBe("/api/patents/a%2Fb");
  });
});
