"""The shapes a search returns. Shared by the engine, the reranker, the CLI and the API."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class PassageHit:
    row: int
    kind: str  # "abstract" | "claim" | "description"
    label: str  # "Claim 7", "Description ¶12"
    text: str
    score: float  # phase-1 score (RRF, BM25 or cosine)
    bm25_rank: int | None
    dense_rank: int | None
    highlights: list[tuple[int, int]]  # character spans of matched query words
    rerank_score: float | None = None  # phase-2 (cross-encoder) score, if reranked


@dataclass
class PatentHit:
    rank: int
    doc_number: str
    title: str | None
    cpc: str | None
    pub_date: str | None
    year: int | None
    has_description: bool
    score: float
    passages: list[PassageHit] = field(default_factory=list)
    first_stage_rank: int | None = None  # rank before reranking ("was #14, now #2")


@dataclass
class SearchResponse:
    query: str
    note: str | None
    mode: str
    strategy: str
    results: list[PatentHit]
    timings_ms: dict[str, float]
    candidate_patents: int | None  # None = no metadata filter
    candidate_passages: int
    excluded_no_cpc: int
    reranker: str | None = None  # model name when phase 2 ran
    rerank_note: str | None = None  # why "auto" skipped reranking

    def to_dict(self) -> dict:
        return asdict(self)
