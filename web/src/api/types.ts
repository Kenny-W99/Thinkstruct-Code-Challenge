// Mirrors src/patent_search/api/schemas.py. Keep the two in sync.

export type Kind = "abstract" | "claim" | "description";
export type Mode = "hybrid" | "bm25" | "dense";
export type Strategy = "pre" | "post";
export type RerankChoice = "auto" | "on" | "off";

export interface Passage {
  row: number;
  kind: Kind;
  label: string;
  text: string;
  score: number;
  bm25_rank: number | null;
  dense_rank: number | null;
  highlights: [number, number][];
  rerank_score: number | null;
}

export interface PatentHit {
  rank: number;
  doc_number: string;
  title: string | null;
  cpc: string | null;
  pub_date: string | null;
  year: number | null;
  has_description: boolean;
  score: number;
  passages: Passage[];
  first_stage_rank: number | null;
}

export interface SearchResponse {
  query: string;
  note: string | null;
  mode: Mode;
  strategy: Strategy;
  results: PatentHit[];
  timings_ms: Record<string, number>;
  candidate_patents: number | null;
  candidate_passages: number;
  excluded_no_cpc: number;
  reranker: string | null;
  rerank_note: string | null;
}

export interface Claim {
  number: number;
  text: string;
  depends_on: number[];
  independent: boolean;
}

export interface Patent {
  doc_number: string;
  title: string | null;
  abstract: string | null;
  cpc: string | null;
  cpc_subclass: string | null;
  pub_date: string | null;
  year: number | null;
  filename: string;
  has_description: boolean;
  claims: Claim[];
  description: string[];
}

export interface Facets {
  n_patents: number;
  no_cpc: number;
  subclasses: { code: string; count: number }[];
  date_min: string | null;
  date_max: string | null;
  year_min: number | null;
  year_max: number | null;
}

export interface Health {
  status: "ok";
  dataset: string | null;
  n_patents: number;
  n_passages: number;
  passages_by_kind: Record<string, number>;
  embedder: string | null;
  reranker: string | null;
  modes: Mode[];
}
