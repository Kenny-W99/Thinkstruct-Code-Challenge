import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import { ApiError } from "./api/client";
import type { SearchState } from "./lib/query";
import { fakeApi, health, hit, response } from "./test/fixtures";

beforeEach(() => window.history.replaceState(null, "", "/"));
afterEach(() => vi.restoreAllMocks());

describe("App", () => {
  it("shows the index status and the intro before any search", async () => {
    render(<App client={fakeApi()} />);
    expect(await screen.findByText("640")).toBeInTheDocument();
    expect(screen.getByText("bge-small-en-v1.5")).toBeInTheDocument();
    expect(screen.getByRole("region", { name: "How search works" })).toBeInTheDocument();
  });

  it("searches on Enter and puts the query in the URL", async () => {
    const search = vi.fn().mockResolvedValue(response());
    render(<App client={fakeApi({ search })} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Search query" }), "carbon fiber spokes{Enter}");
    expect(await screen.findByRole("button", { name: "Wheel Rim" })).toBeInTheDocument();
    expect(search.mock.calls[0][0]).toMatchObject({ q: "carbon fiber spokes", rerank: "auto" });
    expect(window.location.search).toBe("?q=carbon+fiber+spokes");
  });

  it("re-runs the search when a filter changes, and the URL carries it", async () => {
    const search = vi.fn().mockResolvedValue(response());
    render(<App client={fakeApi({ search })} />);
    await userEvent.click(screen.getByRole("button", { name: "bicycle wheel spoke made of carbon fiber" }));
    await screen.findByRole("button", { name: "Wheel Rim" });
    await userEvent.click(await screen.findByRole("checkbox", { name: /B60B/ }));
    await waitFor(() => expect(search).toHaveBeenCalledTimes(2));
    expect((search.mock.calls[1][0] as SearchState).cpc).toEqual(["B60B"]);
    expect(new URLSearchParams(window.location.search).getAll("cpc")).toEqual(["B60B"]);
  });

  it("waits for typing to pause before re-running the search for the Title filter", async () => {
    const search = vi.fn().mockResolvedValue(response());
    render(<App client={fakeApi({ search })} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Search query" }), "spoke{Enter}");
    await screen.findByRole("button", { name: "Wheel Rim" });
    await userEvent.type(screen.getByRole("textbox", { name: "Title" }), "wheel");
    expect(search).toHaveBeenCalledTimes(1); // nothing sent per keystroke
    await waitFor(() => expect(search).toHaveBeenCalledTimes(2));
    expect((search.mock.calls[1][0] as SearchState).title).toBe("wheel");
  });

  it("runs the search in the URL on load", async () => {
    window.history.replaceState(null, "", "/?q=spoke&mode=bm25&rerank=off");
    const search = vi.fn().mockResolvedValue(response({ mode: "bm25" }));
    render(<App client={fakeApi({ search })} />);
    await screen.findByRole("button", { name: "Wheel Rim" });
    expect(search.mock.calls[0][0]).toMatchObject({ q: "spoke", mode: "bm25", rerank: "off" });
    expect(screen.getByRole("radio", { name: "Keywords" })).toBeChecked();
  });

  it("shows the API's message for a bad request", async () => {
    const search = vi.fn().mockRejectedValue(new ApiError(400, "Not a valid classification prefix: '60B'"));
    render(<App client={fakeApi({ search })} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Search query" }), "wheel{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("Not a valid classification prefix: '60B'");
  });

  it("marks the previous results as old when the next search fails", async () => {
    const search = vi
      .fn()
      .mockResolvedValueOnce(response())
      .mockRejectedValueOnce(new ApiError(400, "Not a valid classification prefix: '60B'"));
    const { container } = render(<App client={fakeApi({ search })} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Search query" }), "wheel{Enter}");
    await screen.findByRole("button", { name: "Wheel Rim" });
    expect(container.querySelector("main.results")).not.toHaveClass("is-stale");

    await userEvent.type(screen.getByRole("textbox", { name: "More classification prefixes" }), "60B{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("from the previous search");
    expect(container.querySelector("main.results")).toHaveClass("is-stale");
  });

  it("shows a top_k from a shared URL even if the menu doesn't list it", async () => {
    window.history.replaceState(null, "", "/?q=spoke&top_k=30&rerank_candidates=100");
    const search = vi.fn().mockResolvedValue(response());
    render(<App client={fakeApi({ search })} />);
    await screen.findByRole("button", { name: "Wheel Rim" });
    expect(screen.getByRole("combobox")).toHaveValue("30");
    expect(search.mock.calls[0][0]).toMatchObject({ topK: 30, rerankCandidates: 100 });
  });

  it("shows rank changes after reranking and opens the full patent", async () => {
    const reranked = response({ reranker: "mixedbread-ai/mxbai-rerank-base-v1", results: [hit({ first_stage_rank: 4 })] });
    render(<App client={fakeApi({ search: () => Promise.resolve(reranked) })} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Search query" }), "carbon spokes{Enter}");
    expect(await screen.findByText("from #4")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Wheel Rim" }));
    expect(await screen.findByRole("dialog")).toHaveTextContent("light alloy wheel rim");
  });

  it("disables options the server can't run", async () => {
    const keywordOnly = fakeApi({ health: () => Promise.resolve({ ...health, modes: ["bm25"], reranker: null }) });
    render(<App client={keywordOnly} />);
    await waitFor(() => expect(screen.getByRole("radio", { name: "Meaning" })).toBeDisabled());
    expect(screen.getByRole("radio", { name: "On" })).toBeDisabled();
  });
});
