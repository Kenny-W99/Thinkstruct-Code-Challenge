import pytest

from patent_search.evaluate import (
    SetupResult,
    build_eval_queries,
    compare_ranks,
    format_markdown,
    mean_reciprocal_rank,
    rank_of,
    recall_at,
)
from patent_search.load import load_patents
from patent_search.results import PatentHit, SearchResponse


def test_recall_and_mrr():
    ranks = [1, 3, None, 12]
    assert recall_at(ranks, 1) == 0.25
    assert recall_at(ranks, 10) == 0.5
    assert recall_at(ranks, 100) == 0.75
    assert mean_reciprocal_rank(ranks, cutoff=10) == pytest.approx((1 + 1 / 3) / 4)
    assert recall_at([], 10) == 0 and mean_reciprocal_rank([], 10) == 0


def test_eval_queries_use_first_long_independent_claim_of_patents_with_a_description(small_dir):
    patents, _ = load_patents(small_dir)
    queries = build_eval_queries(patents, min_words=3)
    # the SPOKE patent has no description -> no query for it
    assert [q.doc_number for q in queries] == ["20240000001", "20240000003", "20240000004"]
    assert queries[0].text.startswith("a tread portion having circumferential grooves")
    assert len(build_eval_queries(patents, limit=2, min_words=3)) == 2
    assert all(q.doc_number != "20240000004" for q in build_eval_queries(patents, min_words=8))  # its claim is short


def _response(*docs):
    hits = [PatentHit(i, d, None, None, None, None, True, 1.0) for i, d in enumerate(docs, 1)]
    return SearchResponse("q", None, "hybrid", "pre", hits, {"total": 1.0}, None, 0, 0)


def test_rank_of():
    resp = _response("A", "B", "C")
    assert rank_of("B", resp) == 2 and rank_of("Z", resp) is None


def test_compare_ranks_counts_moves_and_treats_missing_as_worst():
    before = SetupResult("h", 4, {}, 0, 0, 0, ranks=[5, 1, None, 3])
    after = SetupResult("r", 4, {}, 0, 0, 0, ranks=[2, 1, 7, None])
    assert compare_ranks(before, after) == {"improved": 2, "worse": 1, "unchanged": 1}


def test_markdown_table():
    r = SetupResult("Hybrid", 10, {1: 0.5, 10: 0.9}, 0.6, 12.34, 20.0)
    md = format_markdown([r], ks=(1, 10), notes="note")
    assert "| Setup | Recall@1 | Recall@10 | MRR | Median ms | p95 ms |" in md
    assert "| Hybrid | 0.500 | 0.900 | 0.600 | 12.3 | 20.0 |" in md
    assert md.rstrip().endswith("note")
    assert "ranks" not in r.to_dict()
