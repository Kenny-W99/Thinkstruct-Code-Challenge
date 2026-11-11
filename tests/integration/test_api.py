"""The REST API end to end, through FastAPI's test client (offline models)."""

import pytest
from fastapi.testclient import TestClient

from patent_search.api import create_app
from patent_search.rerank import OverlapReranker
from patent_search.search import SearchEngine

CLAIM_STYLE = "a wheel rim comprising " + " ".join(["carbon fiber spokes connected to an aluminum hub"] * 6)


@pytest.fixture(scope="module")
def client(small_index, embedder):
    return TestClient(create_app(SearchEngine(small_index, embedder, OverlapReranker())))


@pytest.fixture(scope="module")
def plain_client(small_index):
    """No embedder and no reranker: keyword search only."""
    return TestClient(create_app(SearchEngine(small_index, None)))


def _search(client, **params):
    r = client.get("/api/search", params=params)
    assert r.status_code == 200, r.text
    return r.json()


# ------------------------------------------------------------ meta endpoints


def test_health(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_patents"] == 4 and body["reranker"] == "overlap"
    assert body["modes"] == ["hybrid", "bm25", "dense"]


def test_facets(client):
    body = client.get("/api/facets").json()
    assert body["subclasses"][0] == {"code": "B60B", "count": 2}
    assert body["date_min"] == "2024-02-15" and body["date_max"] == "2024-02-22"


def test_openapi_lists_the_endpoints(client):
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {"/api/health", "/api/facets", "/api/search", "/api/patents/{doc_number}"}


# ------------------------------------------------------------ search


def test_search_basic(client):
    body = _search(client, q="carbon fiber spokes", rerank="off")
    assert body["results"][0]["doc_number"] == "20240000004"
    assert body["candidate_patents"] is None and body["reranker"] is None
    assert body["timings_ms"]["total"] > 0
    first = body["results"][0]["passages"][0]
    assert {"kind", "label", "text", "highlights", "bm25_rank", "dense_rank"} <= set(first)


def test_search_repeated_cpc_is_any_of(client):
    body = client.get("/api/search?q=wheel&cpc=B60D&cpc=B60C&rerank=off").json()
    assert body["candidate_patents"] == 2
    assert {h["cpc"][:4] for h in body["results"]} <= {"B60C", "B60D"}


def test_search_keyword_and_dates(client):
    body = _search(client, q="wheel", keyword="carbon", date_from="2024-02-20", date_to="2024-02-28", rerank="off")
    assert [h["doc_number"] for h in body["results"]] == ["20240000004"]
    assert body["results"][0]["pub_date"] == "2024-02-22"


def test_search_title_exact(client):
    body = _search(client, q="tire", title="pneumatic tire", title_exact="true", rerank="off")
    assert [h["doc_number"] for h in body["results"]] == ["20240000001"]


def test_search_post_strategy_and_kinds(client):
    body = _search(client, q="spokes", cpc="B60B", strategy="post", kinds="claim", rerank="off")
    assert body["strategy"] == "post"
    assert all(p["kind"] == "claim" for h in body["results"] for p in h["passages"])


def test_search_by_patent_id_skips_that_patent(client):
    body = _search(client, q="20240000004", rerank="off")
    assert "20240000004" in body["note"]
    assert "20240000004" not in [h["doc_number"] for h in body["results"]]


def test_rerank_on_off_auto(client):
    on = _search(client, q="carbon fiber spokes", rerank="on")
    assert on["reranker"] == "overlap"
    assert all(h["passages"][0]["rerank_score"] is not None for h in on["results"])
    assert _search(client, q="carbon fiber spokes", rerank="off")["reranker"] is None
    auto_short = _search(client, q="carbon fiber spokes")  # auto is the default
    assert auto_short["reranker"] == "overlap"
    auto_long = _search(client, q=CLAIM_STYLE)
    assert auto_long["reranker"] is None and "skipped" in auto_long["rerank_note"]


def test_keyword_only_server(plain_client):
    body = _search(plain_client, q="carbon fiber", mode="bm25")
    assert body["results"][0]["doc_number"] == "20240000004"
    assert "not available" in body["rerank_note"]  # auto with no reranker loaded


# ------------------------------------------------------------ errors


@pytest.mark.parametrize(
    "params, message",
    [
        ({"q": "   "}, "empty"),
        ({"q": "wheel", "cpc": "60B"}, "60B"),
        ({"q": "99999999999"}, "not in this dataset"),
        ({"q": "20240000004 claim 9"}, "no claim 9"),
        ({"q": "wheel", "date_from": "2024-03-01", "date_to": "2024-02-01"}, "after"),
        ({"q": "wheel", "keyword": "the"}, "no searchable word"),
    ],
)
def test_bad_requests_are_400_with_a_message(client, params, message):
    r = client.get("/api/search", params=params)
    assert r.status_code == 400
    assert message in r.json()["detail"]


@pytest.mark.parametrize(
    "params",
    [
        {},  # q is required
        {"q": "wheel", "mode": "fuzzy"},
        {"q": "wheel", "top_k": 0},
        {"q": "wheel", "top_k": 101},
        {"q": "wheel", "date_from": "not-a-date"},
        {"q": "wheel", "kinds": "figure"},
        {"q": "wheel", "rerank": "maybe"},
        {"q": "wheel", "rerank_candidates": 0},
        {"q": "wheel", "rerank_candidates": 101},
    ],
)
def test_wrong_types_are_422(client, params):
    assert client.get("/api/search", params=params).status_code == 422


def test_engine_limits_are_400(plain_client):
    assert plain_client.get("/api/search", params={"q": "wheel", "mode": "dense"}).status_code == 400  # no embedder
    r = plain_client.get("/api/search", params={"q": "wheel", "mode": "bm25", "rerank": "on"})
    assert r.status_code == 400 and "reranker" in r.json()["detail"]


# ------------------------------------------------------------ patents


def test_get_patent(client):
    body = client.get("/api/patents/20240000001").json()
    assert body["title"] == "PNEUMATIC TIRE" and body["cpc_subclass"] == "B60C"
    assert [c["number"] for c in body["claims"]] == [1, 2, 5]
    assert body["claims"][1]["depends_on"] == [1]


def test_unknown_patent_is_404(client):
    r = client.get("/api/patents/12345678901")
    assert r.status_code == 404 and "not in this dataset" in r.json()["detail"]


def test_rerank_candidates_limits_phase_two(client):
    body = _search(client, q="wheel spokes rim tire hitch", rerank="on", rerank_candidates=2, top_k=4)
    reranked = [h for h in body["results"] if h["passages"][0]["rerank_score"] is not None]
    assert len(reranked) == 2 and len(body["results"]) == 4
