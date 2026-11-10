"""The API's plain conversion functions (no server needed)."""

from datetime import date

from patent_search.api.convert import facets_out, filters_from_params, health_out, patent_out
from patent_search.search import SearchEngine


def test_filters_from_params_drops_blanks():
    f = filters_from_params(cpc=["", " B60B ", "  "], keyword=["carbon", ""], title="  ", date_from=date(2024, 1, 1))
    assert f.cpc == ["B60B"]
    assert f.keywords == ["carbon"]
    assert f.title is None
    assert f.date_from == date(2024, 1, 1) and f.date_to is None


def test_filters_from_params_empty_means_no_filter():
    assert filters_from_params().is_empty()


def test_patent_out_keeps_claim_structure(small_index):
    tire = small_index.patents[small_index.doc_to_idx["20240000001"]]
    out = patent_out(tire)
    assert out.cpc_subclass == "B60C" and out.cpc == tire.cpc.display
    assert out.pub_date == date(2024, 2, 15)
    numbers = [c.number for c in out.claims]
    assert numbers == [1, 2, 5]  # claims 3-4 were canceled
    assert out.claims[0].independent and not out.claims[1].independent
    assert out.claims[1].depends_on == [1]
    assert len(out.description) == 2  # blank paragraphs dropped


def test_facets_out_counts_subclasses_most_first(small_index):
    facets = facets_out(small_index)
    assert facets.n_patents == 4 and facets.no_cpc == 0
    assert facets.subclasses[0].code == "B60B" and facets.subclasses[0].count == 2
    assert {s.code for s in facets.subclasses} == {"B60B", "B60C", "B60D"}
    assert (facets.date_min, facets.date_max) == (date(2024, 2, 15), date(2024, 2, 22))
    assert facets.year_min == facets.year_max == 2024


def test_health_out_lists_available_modes(small_index, embedder):
    assert health_out(SearchEngine(small_index, embedder)).modes == ["hybrid", "bm25", "dense"]
    keyword_only = health_out(SearchEngine(small_index, None))
    assert keyword_only.modes == ["bm25"] and keyword_only.reranker is None
    assert keyword_only.n_patents == 4
