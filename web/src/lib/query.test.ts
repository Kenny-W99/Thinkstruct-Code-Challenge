import { describe, expect, it } from "vitest";
import { DEFAULT_STATE, activeFilterCount, fromParams, toParams, withoutFilters, type SearchState } from "./query";

const full: SearchState = {
  q: "tire tread that reduces noise",
  cpc: ["B60C", "B60B 1/04"],
  keyword: ["groove"],
  title: "pneumatic tire",
  titleExact: true,
  dateFrom: "2024-02-01",
  dateTo: "2024-12-31",
  mode: "bm25",
  strategy: "post",
  kinds: ["claim", "description"],
  topK: 25,
  rerank: "off",
  rerankCandidates: 100,
};

describe("toParams / fromParams", () => {
  it("round-trips every field", () => {
    expect(fromParams(toParams(full))).toEqual(full);
  });

  it("uses the API's parameter names and repeats list params", () => {
    const p = toParams(full);
    expect(p.getAll("cpc")).toEqual(["B60C", "B60B 1/04"]);
    expect(p.get("title_exact")).toBe("true");
    expect(p.get("date_from")).toBe("2024-02-01");
    expect(p.get("top_k")).toBe("25");
    expect(p.get("rerank_candidates")).toBe("100");
  });

  it("leaves out defaults and blanks", () => {
    const p = toParams({ ...DEFAULT_STATE, q: "  wheel  ", cpc: ["", " B60B ", "B60B"], title: "   ", titleExact: true });
    expect(p.toString()).toBe("q=wheel&cpc=B60B");
  });

  it("falls back to defaults for unknown or malformed values", () => {
    const s = fromParams(
      new URLSearchParams("q=x&mode=fuzzy&top_k=999&date_from=yesterday&kinds=figure&rerank=maybe&rerank_candidates=lots"),
    );
    expect(s).toEqual({ ...DEFAULT_STATE, q: "x", topK: 100 });
    expect(fromParams(new URLSearchParams("top_k=0")).topK).toBe(1);
  });
});

describe("the README's two-phase example URL", () => {
  it("survives a round trip", () => {
    const qs = "q=bicycle+spoke&top_k=30&rerank=on&rerank_candidates=100";
    const s = fromParams(new URLSearchParams(qs));
    expect(s).toMatchObject({ topK: 30, rerank: "on", rerankCandidates: 100 });
    expect(toParams(s).toString()).toBe(qs);
  });
});

describe("filter helpers", () => {
  it("counts active filters", () => {
    expect(activeFilterCount(DEFAULT_STATE)).toBe(0);
    expect(activeFilterCount(full)).toBe(7); // 2 cpc + 1 keyword + title + 2 dates + kinds
  });

  it("clears filters but keeps the query and options", () => {
    const cleared = withoutFilters(full);
    expect(activeFilterCount(cleared)).toBe(0);
    expect(cleared.q).toBe(full.q);
    expect(cleared.mode).toBe("bm25");
    expect(cleared.rerank).toBe("off");
  });
});
