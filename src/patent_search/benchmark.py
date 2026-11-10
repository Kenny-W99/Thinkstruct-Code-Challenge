"""Part 1 timing: how long does a search take, with and without filters?

For every (filter case × strategy) we run the same queries several times and
report the median / p95 search time.

- "pre"  = smart pre-filter: lookup tables find matching patents first, then
           only their passages are scored.
- "post" = naive post-filter: score everything, keep the top N, then drop the
           ones that fail the filter (can also return too few results).

Query vectors are computed once up front (CachedEmbedder), so the numbers show
the cost of *searching*, which is what filters change. The query-embedding
cost is measured separately, because it is the same for every case.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

from .embed import Embedder
from .filters import SearchFilters, compile_filters
from .search import SearchEngine

BENCH_QUERIES = (
    "tire tread that reduces road noise",
    "spoke made of carbon fiber",
    "trailer hitch with a locking pin",
    "wheel bearing with a speed sensor",
    "airless tire with flexible spokes",
    "battery pack with liquid cooling plates",
    "wireless charging coil for a phone",
    "neural network that classifies images",
    "encrypting data sent over a network",
    "drug delivery with nanoparticles",
    "solar panel mounting bracket",
    "electric motor stator winding",
    "heated vehicle seat",
    "medical imaging of the heart",
    "touch screen display panel",
    "valve that controls fluid flow",
)


@dataclass(frozen=True)
class FilterCase:
    name: str
    filters: SearchFilters | None


DEFAULT_CASES = (
    FilterCase("no filter", None),
    FilterCase("CPC B60B (wheels)", SearchFilters(cpc=["B60B"])),
    FilterCase("CPC G06F (computing)", SearchFilters(cpc=["G06F"])),
    FilterCase("CPC B60C + keyword 'tread'", SearchFilters(cpc=["B60C"], keywords=["tread"])),
    FilterCase("keyword 'vehicle'", SearchFilters(keywords=["vehicle"])),
    FilterCase("title contains 'wheel'", SearchFilters(title="wheel")),
    FilterCase("CPC B60 + from 2024", SearchFilters(cpc=["B60"], date_from=date(2024, 1, 1))),
)


@dataclass
class TimingRow:
    case: str
    strategy: str  # "pre", "post" or "-" (no filter)
    patents_passing: int
    passages_scored: int
    median_ms: float
    p95_ms: float
    median_filter_ms: float
    median_scoring_ms: float  # BM25 + vector scoring
    avg_results: float  # patents returned (top_k = 10)

    def to_dict(self) -> dict:
        return asdict(self)


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0


def time_case(
    engine: SearchEngine,
    queries: Sequence[str],
    case: FilterCase,
    strategy: str,
    *,
    mode: str = "hybrid",
    repeats: int = 3,
    top_k: int = 10,
) -> TimingRow:
    totals, filter_ms, scoring_ms, n_results = [], [], [], []
    scored = 0
    for _ in range(repeats):
        for q in queries:
            resp = engine.search(q, case.filters, strategy=strategy, mode=mode, top_k=top_k)
            t = resp.timings_ms
            totals.append(t["total"] - t.get("embed", 0.0))
            filter_ms.append(t["filter"] + t.get("post_filter", 0.0))
            scoring_ms.append(t.get("bm25", 0.0) + t.get("dense", 0.0))
            n_results.append(len(resp.results))
            scored = resp.candidate_passages
    passing = (
        engine.index.metadata.candidates(compile_filters(case.filters)).patents.size
        if case.filters is not None
        else len(engine.index.patents)
    )
    return TimingRow(
        case=case.name,
        strategy=strategy if case.filters is not None else "-",
        patents_passing=int(passing),
        passages_scored=int(scored),
        median_ms=statistics.median(totals),
        p95_ms=_p95(totals),
        median_filter_ms=statistics.median(filter_ms),
        median_scoring_ms=statistics.median(scoring_ms),
        avg_results=sum(n_results) / len(n_results),
    )


def run_benchmark(
    engine: SearchEngine,
    queries: Sequence[str] = BENCH_QUERIES,
    cases: Sequence[FilterCase] = DEFAULT_CASES,
    strategies: Sequence[str] = ("pre", "post"),
    *,
    mode: str = "hybrid",
    repeats: int = 3,
) -> list[TimingRow]:
    engine.warmup()
    for q in queries[:3]:  # warm caches
        engine.search(q, mode=mode)
    rows = []
    for case in cases:
        for strategy in strategies if case.filters is not None else ("pre",):
            rows.append(time_case(engine, queries, case, strategy, mode=mode, repeats=repeats))
    return rows


def measure_query_embedding(embedder: Embedder, queries: Sequence[str] = BENCH_QUERIES, repeats: int = 3) -> dict:
    """Median / p95 ms to embed ONE query (what every search pays once)."""
    import time

    embedder.encode(["warm up"], is_query=True)
    ms = []
    for _ in range(repeats):
        for q in queries:
            t = time.perf_counter()
            embedder.encode([q], is_query=True)
            ms.append((time.perf_counter() - t) * 1000)
    return {"median_ms": statistics.median(ms), "p95_ms": _p95(ms)}


def format_timing_markdown(rows: Sequence[TimingRow]) -> str:
    base = next((r.median_ms for r in rows if r.strategy == "-"), None)
    lines = [
        "| Filter | Strategy | Patents passing | Passages scored | Median ms | p95 ms | Filter ms | Scoring ms | Results | vs no filter |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        ratio = f"{r.median_ms / base:.2f}×" if base else "-"
        lines.append(
            f"| {r.case} | {r.strategy} | {r.patents_passing:,} | {r.passages_scored:,} | {r.median_ms:.2f} | "
            f"{r.p95_ms:.2f} | {r.median_filter_ms:.2f} | {r.median_scoring_ms:.2f} | {r.avg_results:.1f} | {ratio} |"
        )
    return "\n".join(lines) + "\n"
