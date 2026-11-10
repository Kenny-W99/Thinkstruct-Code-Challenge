import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { DEFAULT_STATE } from "../lib/query";
import { facets, hit, patent, response } from "../test/fixtures";
import { FilterPanel } from "./FilterPanel";
import { PatentView } from "./PatentView";
import { RankChange } from "./RankChange";
import { ResultCard } from "./ResultCard";
import { Summary } from "./ResultList";
import { TagInput } from "./TagInput";
import { TimingBar, stagesOf } from "./TimingBar";

describe("ResultCard", () => {
  it("shows the title, metadata and highlighted passage", () => {
    render(<ResultCard hit={hit()} onOpen={() => {}} />);
    expect(screen.getByRole("button", { name: "Wheel Rim" })).toBeInTheDocument();
    expect(screen.getByText("US 20240000004")).toBeInTheDocument();
    const marks = document.querySelectorAll("mark");
    expect([...marks].map((m) => m.textContent)).toEqual(["carbon", "fiber"]);
    expect(screen.getByText("keyword #1 · meaning #3")).toBeInTheDocument();
  });

  it("opens the patent at the clicked passage", async () => {
    const onOpen = vi.fn();
    render(<ResultCard hit={hit()} onOpen={onOpen} />);
    await userEvent.click(screen.getByTitle("Show in the full patent"));
    expect(onOpen).toHaveBeenCalledWith("20240000004", "claim-1");
    await userEvent.click(screen.getByRole("button", { name: "Wheel Rim" }));
    expect(onOpen).toHaveBeenLastCalledWith("20240000004");
  });

  it("shows the rerank score when phase 2 ran", () => {
    const reranked = hit({ rank: 1, first_stage_rank: 5 });
    reranked.passages[0] = { ...reranked.passages[0], rerank_score: 2.417 };
    render(<ResultCard hit={reranked} onOpen={() => {}} />);
    expect(screen.getByText("rerank 2.42")).toBeInTheDocument();
    expect(screen.getByText("from #5")).toBeInTheDocument();
  });
});

describe("RankChange", () => {
  it("shows up, down, same, or nothing", () => {
    const { container, rerender } = render(<RankChange rank={1} firstStageRank={5} />);
    expect(container.textContent).toBe("▲ 4from #5");
    rerender(<RankChange rank={6} firstStageRank={2} />);
    expect(container.querySelector(".down")?.textContent).toBe("▼ 4from #2");
    rerender(<RankChange rank={3} firstStageRank={3} />);
    expect(container.textContent).toBe("= #3");
    rerender(<RankChange rank={3} firstStageRank={null} />);
    expect(container.textContent).toBe("");
  });
});

describe("TimingBar", () => {
  it("splits time into phase 1 and phase 2", () => {
    const t = { embed: 12, bm25: 1, dense: 1, group: 1, rerank: 2500, total: 2515 };
    expect(stagesOf(t)).toMatchObject({ phase1: 15, phase2: 2500, total: 2515 });
    expect(stagesOf(t).stages.map((s) => s.key)).toEqual(["embed", "bm25", "dense", "group", "rerank"]);
    render(<TimingBar timings={t} />);
    expect(screen.getByText("2.52 s")).toBeInTheDocument();
    expect(screen.getByText("2.50 s")).toBeInTheDocument(); // phase 2
  });

  it("keeps stages it does not know about", () => {
    expect(stagesOf({ embed: 1, mystery: 2, total: 3 }).stages.map((s) => s.key)).toEqual(["embed", "mystery"]);
  });
});

describe("Summary", () => {
  it("explains filters and the rerank decision", () => {
    const r = response({ candidate_patents: 2, rerank_note: "Reranking skipped: a 90-word, claim-style query." });
    render(<Summary response={r} totalPatents={640} />);
    expect(screen.getByText(/patents pass the filters/).textContent).toContain("2 of 640 patents pass the filters (filtered first)");
    expect(screen.getByText(/Reranking skipped/)).toBeInTheDocument();
  });

  it("names the reranker when phase 2 ran", () => {
    render(<Summary response={response({ reranker: "mixedbread-ai/mxbai-rerank-base-v1" })} totalPatents={640} />);
    expect(screen.getByText("mxbai-rerank-base-v1")).toBeInTheDocument();
  });
});

describe("TagInput", () => {
  it("adds on Enter, removes with the × button and Backspace", async () => {
    const onChange = vi.fn();
    const { rerender } = render(<TagInput label="Words" values={[]} onChange={onChange} normalize={(v) => v.toUpperCase()} />);
    await userEvent.type(screen.getByRole("textbox"), "b60c{Enter}");
    expect(onChange).toHaveBeenLastCalledWith(["B60C"]);

    rerender(<TagInput label="Words" values={["B60C", "B60B"]} onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "Remove B60C" }));
    expect(onChange).toHaveBeenLastCalledWith(["B60B"]);
    await userEvent.type(screen.getByRole("textbox"), "{Backspace}");
    expect(onChange).toHaveBeenLastCalledWith(["B60C"]);
  });
});

describe("FilterPanel", () => {
  it("toggles subclasses from the facets and clears everything", async () => {
    const onChange = vi.fn();
    const { rerender } = render(<FilterPanel state={DEFAULT_STATE} facets={facets} onChange={onChange} />);
    await userEvent.click(screen.getByRole("checkbox", { name: /B60B/ }));
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_STATE, cpc: ["B60B"] });

    rerender(<FilterPanel state={{ ...DEFAULT_STATE, cpc: ["B60B", "B60C 11/03"], keyword: ["groove"] }} facets={facets} onChange={onChange} />);
    expect(screen.getByText("3")).toHaveClass("count-badge");
    expect(screen.getByText("B60C 11/03")).toHaveClass("chip"); // typed prefix shown as a chip
    await userEvent.click(screen.getByRole("button", { name: "Clear all" }));
    expect(onChange).toHaveBeenLastCalledWith(DEFAULT_STATE);
  });

  it("switches between pre- and post-filtering", async () => {
    const onChange = vi.fn();
    render(<FilterPanel state={DEFAULT_STATE} facets={null} onChange={onChange} />);
    await userEvent.click(screen.getByRole("radio", { name: "Filter after" }));
    expect(onChange).toHaveBeenLastCalledWith({ ...DEFAULT_STATE, strategy: "post" });
  });
});

describe("PatentView", () => {
  it("loads the patent, marks the target passage and closes on Escape", async () => {
    const onClose = vi.fn();
    const load = vi.fn().mockResolvedValue(patent);
    render(<PatentView doc="20240000004" anchor="claim-2" load={load} onClose={onClose} />);
    const dialog = await screen.findByRole("dialog", { name: "Wheel Rim" });
    expect(load).toHaveBeenCalledWith("20240000004", expect.any(AbortSignal));
    const claim2 = dialog.querySelector("#claim-2")!;
    expect(claim2).toHaveClass("target", "dependent");
    expect(within(claim2 as HTMLElement).getByText(/↳ claim 1/)).toBeInTheDocument();
    await userEvent.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalled();
  });

  it("shows API errors", async () => {
    render(<PatentView doc="1" load={() => Promise.reject(new Error("Patent 1 is not in this dataset."))} onClose={() => {}} />);
    expect(await screen.findByText("Patent 1 is not in this dataset.")).toBeInTheDocument();
  });

  it("renders nothing when closed", () => {
    const { container } = render(<PatentView doc={null} load={vi.fn()} onClose={() => {}} />);
    expect(container).toBeEmptyDOMElement();
  });
});
