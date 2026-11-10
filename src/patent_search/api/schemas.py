"""The JSON shapes of the REST API (Pydantic models).

They mirror the engine's dataclasses in `results.py`; FastAPI uses them to
validate responses and to write the OpenAPI docs at /docs.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["abstract", "claim", "description"]
Mode = Literal["hybrid", "bm25", "dense"]
Strategy = Literal["pre", "post"]
RerankChoice = Literal["auto", "on", "off"]


class PassageOut(BaseModel):
    row: int
    kind: Kind
    label: str = Field(description='e.g. "Claim 7" or "Description ¶12"')
    text: str
    score: float = Field(description="phase-1 score (RRF, BM25 or cosine)")
    bm25_rank: int | None
    dense_rank: int | None
    highlights: list[tuple[int, int]] = Field(description="character spans of matched query words")
    rerank_score: float | None = None


class PatentHitOut(BaseModel):
    rank: int
    doc_number: str
    title: str | None
    cpc: str | None
    pub_date: date | None
    year: int | None
    has_description: bool
    score: float
    passages: list[PassageOut]
    first_stage_rank: int | None = Field(None, description="rank before reranking (phase 1)")


class SearchOut(BaseModel):
    query: str = Field(description="the text that was searched (a patent ID is expanded to its text)")
    note: str | None
    mode: Mode
    strategy: Strategy
    results: list[PatentHitOut]
    timings_ms: dict[str, float]
    candidate_patents: int | None = Field(description="patents passing the filters (null = no filter)")
    candidate_passages: int
    excluded_no_cpc: int
    reranker: str | None = Field(None, description="model name when phase 2 ran")
    rerank_note: str | None = Field(None, description='why rerank="auto" skipped phase 2')


class ClaimOut(BaseModel):
    number: int
    text: str
    depends_on: list[int]
    independent: bool


class PatentOut(BaseModel):
    doc_number: str
    title: str | None
    abstract: str | None
    cpc: str | None
    cpc_subclass: str | None
    pub_date: date | None
    year: int | None
    filename: str
    has_description: bool
    claims: list[ClaimOut]
    description: list[str] = Field(description="description paragraphs (empty if the index did not keep them)")


class SubclassCount(BaseModel):
    code: str
    count: int


class FacetsOut(BaseModel):
    n_patents: int
    no_cpc: int = Field(description="patents with no classification code")
    subclasses: list[SubclassCount] = Field(description="CPC subclasses, most patents first")
    date_min: date | None
    date_max: date | None
    year_min: int | None
    year_max: int | None


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"
    dataset: str | None
    n_patents: int
    n_passages: int
    passages_by_kind: dict[str, int]
    embedder: str | None
    reranker: str | None
    modes: list[Mode] = Field(description="search modes this server can run")
