// Small fake API answers shared by the component and app tests.

import type { Api } from "../api/client";
import type { Facets, Health, Patent, PatentHit, SearchResponse } from "../api/types";

export const hit = (over: Partial<PatentHit> = {}): PatentHit => ({
  rank: 1,
  doc_number: "20240000004",
  title: "WHEEL RIM",
  cpc: "B60B 2700",
  pub_date: "2024-02-22",
  year: 2024,
  has_description: true,
  score: 0.03,
  first_stage_rank: null,
  passages: [
    {
      row: 7,
      kind: "claim",
      label: "Claim 1",
      text: "A wheel rim comprising carbon fiber spokes.",
      score: 0.03,
      bm25_rank: 1,
      dense_rank: 3,
      highlights: [
        [23, 29],
        [30, 35],
      ],
      rerank_score: null,
    },
  ],
  ...over,
});

export const response = (over: Partial<SearchResponse> = {}): SearchResponse => ({
  query: "carbon fiber spokes",
  note: null,
  mode: "hybrid",
  strategy: "pre",
  results: [hit()],
  timings_ms: { embed: 12, filter: 0.1, bm25: 0.5, dense: 0.9, fuse: 0.1, group: 0.8, total: 14.4 },
  candidate_patents: null,
  candidate_passages: 900,
  excluded_no_cpc: 0,
  reranker: null,
  rerank_note: null,
  ...over,
});

export const health: Health = {
  status: "ok",
  dataset: "small",
  n_patents: 640,
  n_passages: 32929,
  passages_by_kind: { abstract: 645, claim: 10563, description: 21721 },
  embedder: "BAAI/bge-small-en-v1.5",
  reranker: "mixedbread-ai/mxbai-rerank-base-v1",
  modes: ["hybrid", "bm25", "dense"],
};

export const facets: Facets = {
  n_patents: 640,
  no_cpc: 0,
  subclasses: [
    { code: "B60C", count: 318 },
    { code: "B60B", count: 298 },
  ],
  date_min: "2024-02-15",
  date_max: "2025-05-01",
  year_min: 2024,
  year_max: 2025,
};

export const patent: Patent = {
  doc_number: "20240000004",
  title: "WHEEL RIM",
  abstract: "A light alloy wheel rim with spokes made of carbon fiber.",
  cpc: "B60B 2700",
  cpc_subclass: "B60B",
  pub_date: "2024-02-22",
  year: 2024,
  filename: "US20240000004A1-20240222.XML",
  has_description: true,
  claims: [
    { number: 1, text: "A wheel rim comprising carbon fiber spokes.", depends_on: [], independent: true },
    { number: 2, text: "The wheel rim of claim 1, wherein the hub is aluminum.", depends_on: [1], independent: false },
  ],
  description: ["The carbon fiber spokes connect the rim to the hub."],
};

/** A fake client; override any method per test. */
export function fakeApi(over: Partial<Api> = {}): Api {
  return {
    health: () => Promise.resolve(health),
    facets: () => Promise.resolve(facets),
    search: () => Promise.resolve(response()),
    patent: () => Promise.resolve(patent),
    ...over,
  };
}
