"""Part 3 evaluation: does phase 2 (reranking) actually find better patents?

Two tasks, each with one known correct patent per query:

1. "claims" - claim -> its own description (no labels needed).
   For every patent with a description, its first independent claim is the
   query. We search ONLY description paragraphs, indexed WITHOUT titles, so the
   answer can't simply match the claim, abstract or title. This copies a real
   claim-mapping job (find the text that supports a claim). It favors exact
   word overlap, because a claim reuses its own description's words.

2. "plain" - plain-English idea -> patent.
   benchmarks/plain_english_queries.jsonl holds 48 one-sentence descriptions
   of patents, written in everyday words (by the AI assistant that helped build
   this project, from each patent's title + abstract). They are searched
   against the normal index. This is the other half of our problem statement,
   and where meaning (not exact words) matters.

Honest limit for both: one correct patent per query. Real prior-art search has several.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

from .chunk import ChunkConfig
from .load import Patent
from .results import SearchResponse
from .search import SearchEngine

EVAL_CHUNK_CONFIG = ChunkConfig(include_description=True, title_context=False)
DEFAULT_KS = (1, 10, 30, 100)


@dataclass(frozen=True)
class EvalQuery:
    doc_number: str  # the correct answer
    text: str  # the claim used as the query


@dataclass(frozen=True)
class Setup:
    name: str
    mode: str = "hybrid"
    rerank: bool = False


SETUPS = {
    "bm25": Setup("BM25 only", "bm25"),
    "dense": Setup("Embeddings only", "dense"),
    "hybrid": Setup("Hybrid (RRF)", "hybrid"),
    "rerank": Setup("Hybrid + cross-encoder rerank", "hybrid", rerank=True),
}


@dataclass
class SetupResult:
    name: str
    n_queries: int
    recall: dict[int, float]  # k -> share of queries whose patent is in the top k
    mrr: float  # mean of 1/rank (0 if not found in the top max(k))
    median_ms: float  # search time per query (query embedding excluded)
    p95_ms: float
    median_rerank_ms: float | None = None
    ranks: list[int | None] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("ranks")
        return d


def build_eval_queries(patents: Sequence[Patent], limit: int | None = None, min_words: int = 8) -> list[EvalQuery]:
    """One query per patent with a description: its first independent claim
    with at least `min_words` words (very short claims say almost nothing)."""
    queries: list[EvalQuery] = []
    for p in patents:
        if not p.paragraphs:
            continue
        claim = next((c for c in p.claims if c.is_independent and len(c.text.split()) >= min_words), None)
        if claim is not None:
            queries.append(EvalQuery(p.doc_number, claim.text))
        if limit and len(queries) >= limit:
            break
    return queries


def load_query_file(path: Path | str, known_docs: set[str] | None = None) -> list[EvalQuery]:
    """Read {"doc_number", "query"} lines. Raises if a query points to a patent
    that is not in the index (a typo would silently count as a miss)."""
    queries = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if known_docs is not None and row["doc_number"] not in known_docs:
            raise ValueError(f"{path}:{n}: patent {row['doc_number']} is not in the index")
        queries.append(EvalQuery(row["doc_number"], row["query"]))
    return queries


def rank_of(doc_number: str, response: SearchResponse) -> int | None:
    return next((h.rank for h in response.results if h.doc_number == doc_number), None)


def recall_at(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: Sequence[int | None], cutoff: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= cutoff) / len(ranks) if ranks else 0.0


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0


def run_setup(
    engine: SearchEngine,
    queries: Sequence[EvalQuery],
    setup: Setup,
    ks: Sequence[int] = DEFAULT_KS,
    *,
    kinds: Sequence[str] | None = ("description",),
    retrieve_n: int = 1000,
    rerank_passages: int = 1,
    rerank_candidates: int | None = None,
) -> SetupResult:
    """Search every query with one setup. `retrieve_n` is large so that phase 1
    returns ~100 patents from paragraph-level matches."""
    depth = max(ks)
    ranks: list[int | None] = []
    times: list[float] = []
    rerank_times: list[float] = []
    for q in queries:
        resp = engine.search(
            q.text,
            mode=setup.mode,
            rerank=setup.rerank,
            kinds=list(kinds) if kinds else None,
            top_k=depth,
            passages_per_patent=1,
            retrieve_n=retrieve_n,
            rerank_candidates=rerank_candidates or depth,
            rerank_passages=rerank_passages,
        )
        ranks.append(rank_of(q.doc_number, resp))
        t = resp.timings_ms
        times.append(t["total"] - t.get("embed", 0.0))
        if "rerank" in t:
            rerank_times.append(t["rerank"])
    return SetupResult(
        name=setup.name,
        n_queries=len(queries),
        recall={k: recall_at(ranks, k) for k in ks},
        mrr=mean_reciprocal_rank(ranks, depth),
        median_ms=statistics.median(times) if times else 0.0,
        p95_ms=_p95(times),
        median_rerank_ms=statistics.median(rerank_times) if rerank_times else None,
        ranks=ranks,
    )


def format_markdown(results: Sequence[SetupResult], ks: Sequence[int] = DEFAULT_KS, notes: str = "") -> str:
    head = "| Setup | " + " | ".join(f"Recall@{k}" for k in ks) + " | MRR | Median ms | p95 ms |"
    sep = "|---|" + "---:|" * (len(ks) + 3)
    lines = [head, sep]
    for r in results:
        cells = [f"{r.recall[k]:.3f}" for k in ks] + [f"{r.mrr:.3f}", f"{r.median_ms:.1f}", f"{r.p95_ms:.1f}"]
        lines.append(f"| {r.name} | " + " | ".join(cells) + " |")
    if notes:
        lines += ["", notes]
    return "\n".join(lines) + "\n"


def compare_ranks(before: SetupResult, after: SetupResult) -> dict[str, int]:
    """How many queries moved up / down / stayed after reranking (missing = rank depth+1)."""
    depth = 10**9
    up = down = same = 0
    for a, b in zip(before.ranks, after.ranks):
        a, b = a or depth, b or depth
        up += b < a
        down += b > a
        same += b == a
    return {"improved": up, "worse": down, "unchanged": same}
