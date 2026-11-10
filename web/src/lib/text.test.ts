import { describe, expect, it } from "vitest";
import { anchorFor, displayTitle, formatMs, segments, shortModel, snippetWindow } from "./text";

describe("segments", () => {
  it("splits text around highlight spans", () => {
    expect(segments("a tire tread", [[2, 6]])).toEqual([
      { text: "a ", hit: false },
      { text: "tire", hit: true },
      { text: " tread", hit: false },
    ]);
  });

  it("sorts spans and skips overlapping or out-of-range ones", () => {
    const parts = segments("abcdefgh", [
      [4, 6],
      [0, 2],
      [1, 3], // overlaps the first
      [7, 99], // past the end
    ]);
    expect(parts.filter((p) => p.hit).map((p) => p.text)).toEqual(["ab", "ef"]);
    expect(parts.map((p) => p.text).join("")).toBe("abcdefgh");
  });

  it("respects a window", () => {
    expect(segments("0123456789", [[1, 2], [5, 6]], 3, 8)).toEqual([
      { text: "34", hit: false },
      { text: "5", hit: true },
      { text: "67", hit: false },
    ]);
  });
});

describe("snippetWindow", () => {
  const long = Array.from({ length: 200 }, (_, i) => `w${i}`).join(" ");

  it("returns the whole text when short", () => {
    expect(snippetWindow("short text", [[0, 5]])).toEqual({ from: 0, to: 10, clippedStart: false, clippedEnd: false });
  });

  it("starts near the first highlight, on a word boundary", () => {
    const at = long.indexOf("w150");
    const w = snippetWindow(long, [[at, at + 4]], 120);
    expect(w.from).toBeLessThan(at);
    expect(w.to).toBeGreaterThan(at + 4);
    expect(long[w.from - 1]).toBe(" ");
    expect(w.clippedStart && w.clippedEnd).toBe(true);
  });
});

describe("displayTitle", () => {
  it("title-cases ALL CAPS titles but keeps small words lower", () => {
    expect(displayTitle("PNEUMATIC TIRE AND METHOD OF MAKING THE SAME")).toBe("Pneumatic Tire and Method of Making the Same");
  });
  it("keeps mixed-case titles and handles missing ones", () => {
    expect(displayTitle("Trailer Hitch System")).toBe("Trailer Hitch System");
    expect(displayTitle(null)).toBe("(untitled)");
  });
});

describe("anchorFor", () => {
  it("maps passage labels to ids in the patent view", () => {
    expect(anchorFor("Claim 7")).toBe("claim-7");
    expect(anchorFor("Claim 12 (part 2/3)")).toBe("claim-12");
    expect(anchorFor("Description ¶4")).toBe("para-4");
    expect(anchorFor("Title + Abstract")).toBe("abstract");
  });
});

describe("formatting", () => {
  it("formats milliseconds at three scales", () => {
    expect(formatMs(0.42)).toBe("0.4 ms");
    expect(formatMs(14.7)).toBe("15 ms");
    expect(formatMs(2573)).toBe("2.57 s");
  });
  it("shortens model names", () => {
    expect(shortModel("mixedbread-ai/mxbai-rerank-base-v1")).toBe("mxbai-rerank-base-v1");
    expect(shortModel(null)).toBe("none");
  });
});
