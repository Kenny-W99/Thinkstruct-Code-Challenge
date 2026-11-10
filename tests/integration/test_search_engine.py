"""End-to-end: load -> chunk -> index -> search, with the offline hashing embedder."""

from datetime import date

import pytest

from patent_search.filters import SearchFilters
from patent_search.rerank import OverlapReranker
from patent_search.search import QueryError, SearchEngine


def docs(resp):
    return [hit.doc_number for hit in resp.results]


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_every_mode_finds_the_obvious_match(engine, mode):
    resp = engine.search("tire tread grooves that reduce road noise", mode=mode)
    assert docs(resp)[0] == "20240000001"


def test_result_shape_highlights_and_timings(engine):
    resp = engine.search("carbon fiber spokes", top_k=2)
    top = resp.results[0]
    assert top.doc_number == "20240000004" and top.rank == 1 and top.cpc == "B60B 2700"
    assert 1 <= len(top.passages) <= 3
    p = top.passages[0]
    assert p.label and p.kind in {"abstract", "claim", "description"}
    assert any(p.text[a:b].lower().startswith("carbon") for a, b in p.highlights)
    assert {"embed", "filter", "bm25", "dense", "fuse", "group", "total"} <= set(resp.timings_ms)
    assert resp.candidate_patents is None  # no filter used
    assert len(resp.results) <= 2


def test_patents_are_unique_and_ranked(engine):
    resp = engine.search("wheel spoke rim", top_k=10)
    assert len(docs(resp)) == len(set(docs(resp)))
    assert [h.rank for h in resp.results] == list(range(1, len(resp.results) + 1))


def test_cpc_filter_limits_results(engine):
    resp = engine.search("tire", SearchFilters(cpc=["B60B"]))
    assert set(docs(resp)) <= {"20240000002", "20240000004"}
    assert resp.candidate_patents == 2


def test_keyword_title_and_date_filters(engine):
    assert docs(engine.search("wheel", SearchFilters(keywords=["carbon"]))) == ["20240000004"]
    assert docs(engine.search("wheel", SearchFilters(title="spoke"))) == ["20240000002"]
    resp = engine.search("wheel tire hitch spoke", SearchFilters(date_from=date(2024, 2, 20)))
    assert set(docs(resp)) <= {"20240000003", "20240000004"}


def test_filter_with_no_matches_returns_nothing(engine):
    resp = engine.search("tire", SearchFilters(keywords=["zebra"]))
    assert resp.results == [] and resp.candidate_patents == 0


@pytest.mark.parametrize("mode", ["hybrid", "bm25", "dense"])
def test_pre_and_post_filter_agree_when_overfetch_covers_everything(engine, mode):
    f = SearchFilters(cpc=["B60"])
    pre = engine.search("wheel with spokes", f, mode=mode, strategy="pre")
    post = engine.search("wheel with spokes", f, mode=mode, strategy="post")
    assert docs(pre) == docs(post)
    assert "post_filter" in post.timings_ms


def test_naive_post_filter_can_lose_results_when_overfetch_is_small(engine):
    """The classic post-filter problem: the top-N overall are all tire passages,
    so filtering them to B60D (trailer hitch) afterwards leaves nothing."""
    f = SearchFilters(cpc=["B60D"])
    pre = engine.search("tire tread grooves noise", f, strategy="pre", mode="dense")
    post = engine.search("tire tread grooves noise", f, strategy="post", mode="dense", overfetch=2)
    assert docs(pre) == ["20240000003"]
    assert docs(post) == []


def test_kinds_limits_passage_types(engine):
    resp = engine.search("wheel tire spoke", kinds=["claim"])
    assert resp.results and all(p.kind == "claim" for h in resp.results for p in h.passages)


def test_patent_id_query_uses_its_text_and_excludes_itself(engine):
    resp = engine.search("20240000004")
    assert "20240000004" not in docs(resp)
    assert resp.note and "20240000004" in resp.note
    assert "carbon" in resp.query.lower()


def test_patent_claim_query(engine):
    resp = engine.search("US20240000001A1 claim 2")
    assert "grooves reduce road noise" in resp.query
    assert "20240000001" not in docs(resp)


@pytest.mark.parametrize(
    "query, message",
    [("", "empty"), ("99999999999", "not in this dataset"), ("20240000001 claim 42", "no claim 42")],
)
def test_query_errors(engine, query, message):
    with pytest.raises(QueryError, match=message):
        engine.search(query)


def test_bad_arguments(engine, small_index):
    with pytest.raises(QueryError):
        engine.search("tire", mode="magic")
    with pytest.raises(QueryError):
        engine.search("tire", strategy="sideways")
    with pytest.raises(QueryError, match="no embedder"):
        SearchEngine(small_index, None).search("tire", mode="hybrid")
    assert docs(SearchEngine(small_index, None).search("tire", mode="bm25"))[0] == "20240000001"


def test_warmup_is_safe_with_and_without_models(engine, small_index, embedder):
    engine.warmup()
    SearchEngine(small_index, None).warmup()
    SearchEngine(small_index, embedder, OverlapReranker()).warmup()


# ---------------------------------------------------------------- two-phase search


@pytest.fixture(scope="module")
def rerank_engine(small_index, embedder):
    return SearchEngine(small_index, embedder, OverlapReranker())


def test_rerank_marks_results_and_times_phase_two(rerank_engine):
    resp = rerank_engine.search("carbon fiber spokes", rerank=True, top_k=3)
    assert resp.reranker == "overlap" and "rerank" in resp.timings_ms
    assert resp.results[0].doc_number == "20240000004"
    assert all(h.first_stage_rank is not None for h in resp.results)
    assert all(h.passages[0].rerank_score is not None for h in resp.results)
    assert [h.rank for h in resp.results] == list(range(1, len(resp.results) + 1))
    assert len(resp.results) <= 3 and all(len(h.passages) <= 3 for h in resp.results)


class _CountingReranker(OverlapReranker):
    def __init__(self):
        self.seen = 0

    def score(self, query, passages):
        self.seen += len(passages)
        return super().score(query, passages)


@pytest.mark.parametrize("rerank_passages", [1, 2])
def test_reranker_reads_rerank_passages_per_patent_but_results_show_more(small_index, embedder, rerank_passages):
    counter = _CountingReranker()
    engine = SearchEngine(small_index, embedder, counter)
    plain = engine.search("wheel tire spokes grooves hub", passages_per_patent=3)
    resp = engine.search("wheel tire spokes grooves hub", rerank=True, rerank_passages=rerank_passages, passages_per_patent=3)
    assert counter.seen == sum(min(len(h.passages), rerank_passages) for h in plain.results)
    shown = {h.doc_number: len(h.passages) for h in plain.results}
    for hit in resp.results:
        assert len(hit.passages) == shown[hit.doc_number]  # unscored phase-1 passages are still shown
        scored = [p.rerank_score is not None for p in hit.passages]
        assert scored == sorted(scored, reverse=True)  # reranked passages first
        assert sum(scored) == min(rerank_passages, len(hit.passages))


def test_rerank_only_the_top_candidates_rest_keep_phase1_order(rerank_engine):
    full = rerank_engine.search("wheel spokes rim tire hitch", top_k=4)
    resp = rerank_engine.search("wheel spokes rim tire hitch", top_k=4, rerank=True, rerank_candidates=2)
    assert len(resp.results) == len(full.results)
    head, tail = resp.results[:2], resp.results[2:]
    assert {h.doc_number for h in head} == {h.doc_number for h in full.results[:2]}  # same 2, maybe reordered
    assert [h.doc_number for h in tail] == [h.doc_number for h in full.results[2:]]  # untouched
    assert all(h.passages[0].rerank_score is None for h in tail)
    assert [h.first_stage_rank for h in tail] == [3, 4][: len(tail)]
    assert [h.rank for h in resp.results] == list(range(1, len(resp.results) + 1))


def test_rerank_respects_filters(rerank_engine):
    resp = rerank_engine.search("wheel spokes", SearchFilters(cpc=["B60B"]), rerank=True)
    assert {h.doc_number for h in resp.results} <= {"20240000002", "20240000004"}


def test_without_rerank_nothing_is_marked(rerank_engine):
    resp = rerank_engine.search("carbon fiber spokes")
    assert resp.reranker is None and "rerank" not in resp.timings_ms
    assert all(h.first_stage_rank is None for h in resp.results)


def test_rerank_needs_a_reranker(engine):
    with pytest.raises(QueryError, match="reranker"):
        engine.search("tire", rerank=True)


def test_auto_rerank_runs_for_short_queries_and_skips_long_claim_style_ones(rerank_engine):
    short = rerank_engine.search("carbon fiber spokes", rerank="auto")
    assert short.reranker == "overlap" and short.rerank_note is None
    long_query = "a wheel rim comprising " + " ".join(["carbon fiber spokes connected to an aluminum hub"] * 6)
    skipped = rerank_engine.search(long_query, rerank="auto")
    assert skipped.reranker is None and "skipped" in skipped.rerank_note
    assert all(h.first_stage_rank is None for h in skipped.results)


def test_auto_rerank_without_a_reranker_just_skips(engine):
    resp = engine.search("tire", rerank="auto")
    assert resp.reranker is None and "not available" in resp.rerank_note


def test_rerank_value_is_validated(rerank_engine):
    with pytest.raises(QueryError, match="rerank must be"):
        rerank_engine.search("tire", rerank="sometimes")


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"kinds": ["figure"]}])
def test_invalid_options(engine, kwargs):
    with pytest.raises(QueryError):
        engine.search("tire", **kwargs)


def test_embedder_dimension_must_match(small_index):
    from patent_search.embed import HashingEmbedder

    with pytest.raises(ValueError, match="dim"):
        SearchEngine(small_index, HashingEmbedder(dim=7))


def test_rerank_sizes_are_validated(rerank_engine):
    with pytest.raises(QueryError, match="rerank_candidates"):
        rerank_engine.search("tire", rerank=True, rerank_candidates=0)
