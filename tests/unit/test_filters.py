from datetime import date

import pytest

from patent_search.filters import MetadataIndex, SearchFilters, compile_filters
from patent_search.load import load_patents, parse_record


@pytest.fixture(scope="module")
def patents(small_dir, large_dir):
    small, _ = load_patents(small_dir)
    large, _ = load_patents(large_dir)
    return small + large  # mixes both formats on purpose


@pytest.fixture(scope="module")
def meta(patents):
    return MetadataIndex(patents)


def _docs(meta_index, patents, **kwargs):
    found = meta_index.candidates(compile_filters(SearchFilters(**kwargs)))
    return {patents[i].doc_number for i in found.patents.tolist()}


def test_empty_filters():
    assert SearchFilters().is_empty()
    assert not SearchFilters(cpc=["B60B"]).is_empty()


def test_cpc_prefix_across_both_formats(meta, patents):
    assert _docs(meta, patents, cpc=["B60B"]) == {"20240000002", "20240000004", "20100000003"}
    assert _docs(meta, patents, cpc=["B60"]) == {"20240000001", "20240000002", "20240000003", "20240000004", "20100000003"}
    assert _docs(meta, patents, cpc=["B60D", "H02J"]) == {"20240000003", "20180000002"}  # any-of
    assert _docs(meta, patents, cpc=["B60B 3/02"]) == {"20100000003"}


def test_cpc_filter_reports_patents_without_a_code(meta):
    result = meta.candidates(compile_filters(SearchFilters(cpc=["B60"])))
    assert result.excluded_no_cpc == 1  # the battery patent has "N/A"


def test_keywords_all_must_appear_and_use_plural_folding(meta, patents):
    assert _docs(meta, patents, keywords=["carbon"]) == {"20240000004"}
    assert _docs(meta, patents, keywords=["spokes"]) == {"20240000002", "20240000004"}
    assert _docs(meta, patents, keywords=["spoke", "carbon"]) == {"20240000004"}
    assert _docs(meta, patents, keywords=["zebra"]) == set()


def test_title_contains_and_exact(meta, patents):
    assert _docs(meta, patents, title="wheel") == {"20240000004", "20100000003"}
    assert _docs(meta, patents, title="Wheel Rim", title_exact=True) == {"20240000004"}
    assert _docs(meta, patents, title="wheel", title_exact=True) == set()


def test_title_contains_is_a_true_substring_search():
    titles = ["FLYWHEEL ASSEMBLY", "wheel", "", "SPOKE WHEEL WHEEL", "rim"]
    patents = [parse_record({"doc_number": str(i), "title": t or "N/A", "abstract": "x"}) for i, t in enumerate(titles)]
    m = MetadataIndex(patents)
    found = lambda **kw: m.candidates(compile_filters(SearchFilters(**kw))).patents.tolist()  # noqa: E731
    assert found(title="wheel") == [0, 1, 3]  # inside a word, at the start, twice in one title
    assert found(title="heel ass") == [0]
    assert found(title="rim") == [4]  # the last title
    assert found(title="wheel\nrim") == []  # can't match across two titles
    assert found(title="WHEEL", title_exact=True) == [1]


def test_dates_use_exact_date_or_whole_year(meta, patents):
    assert _docs(meta, patents, date_from=date(2024, 2, 20)) == {"20240000003", "20240000004"}
    assert _docs(meta, patents, date_from=date(2015, 6, 1), date_to=date(2015, 6, 30)) == {"20150000001"}


def test_combined_filters(meta, patents):
    assert _docs(meta, patents, cpc=["B60B"], keywords=["carbon"], date_to=date(2024, 12, 31)) == {"20240000004"}


def test_passes_agrees_with_candidates(meta, patents):
    """The post-filter's per-patent check must give exactly the pre-filter's answer."""
    cases = [
        SearchFilters(cpc=["B60B"]),
        SearchFilters(keywords=["spokes"]),
        SearchFilters(title="wheel"),
        SearchFilters(date_from=date(2016, 1, 1)),
        SearchFilters(cpc=["B60"], keywords=["tire"], date_to=date(2024, 2, 16)),
    ]
    for f in cases:
        compiled = compile_filters(f)
        pre = set(meta.candidates(compiled).patents.tolist())
        post = {i for i in range(len(patents)) if meta.passes(i, compiled)}
        assert pre == post, f


def test_invalid_cpc_raises():
    with pytest.raises(ValueError):
        compile_filters(SearchFilters(cpc=["60B"]))


def test_backwards_date_range_raises():
    with pytest.raises(ValueError, match="after"):
        compile_filters(SearchFilters(date_from=date(2024, 3, 1), date_to=date(2024, 2, 1)))
    compile_filters(SearchFilters(date_from=date(2024, 2, 1), date_to=date(2024, 2, 1)))  # one day is fine
