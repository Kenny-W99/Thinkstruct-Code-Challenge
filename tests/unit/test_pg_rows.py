"""Part 2 PoC: rows and WHERE clauses (no database needed)."""

from datetime import date

import numpy as np
import pytest

from patent_search.filters import SearchFilters
from patent_search.load import load_patents
from patent_search.pg.rows import PATENT_COLUMNS, patent_row, poc_passages, vector_literal, where_clause


@pytest.fixture(scope="module")
def by_doc(small_dir, large_dir):
    small, _ = load_patents(small_dir)
    large, _ = load_patents(large_dir)
    return {p.doc_number: p for p in small + large}


def test_patent_row_matches_the_columns(by_doc):
    row = dict(zip(PATENT_COLUMNS, patent_row(by_doc["20240000001"], "a.json")))
    assert len(row) == len(PATENT_COLUMNS)
    assert (row["cpc_subclass"], row["cpc_group"], row["cpc_digits"]) == ("B60C", None, "1103")
    assert row["pub_date"] == date(2024, 2, 15) and row["pub_year"] == 2024
    assert row["n_claims"] == 3 and row["source_file"] == "a.json"

    wheel = dict(zip(PATENT_COLUMNS, patent_row(by_doc["20100000003"], "b.json")))
    assert (wheel["cpc_subclass"], wheel["cpc_group"], wheel["cpc_subgroup"]) == ("B60B", "3", "02")
    assert wheel["pub_date"] is None and wheel["pub_year"] == 2010

    battery = dict(zip(PATENT_COLUMNS, patent_row(by_doc["20150000001"], "c.json")))
    assert battery["cpc_raw"] is None and battery["title"] is None  # "N/A" values


def test_poc_passages_are_abstract_and_independent_claims(by_doc):
    pairs = poc_passages(by_doc["20240000001"])
    assert [ps.label for ps, _ in pairs] == ["Title + Abstract", "Claim 1"]  # claims 2 and 5 are dependent
    assert pairs[1][1].startswith("PNEUMATIC TIRE: ")  # title context, like the main index


def test_vector_literal():
    assert vector_literal(np.array([0.5, -0.25, 1e-7], dtype=np.float32)) == "[0.5,-0.25,1e-07]"


def test_no_filters_is_true():
    assert where_clause(None) == ("TRUE", {})
    assert where_clause(SearchFilters()) == ("TRUE", {})


def test_cpc_filters():
    sql, params = where_clause(SearchFilters(cpc=["B60", "H01M 10/6"]))
    assert sql.startswith("(p.cpc_subclass LIKE %(f0)s OR (p.cpc_subclass = %(f1)s")
    assert params["f0"] == "B60%" and params["f1"] == "H01M"
    assert "10" in params.values() and "6%" in params.values() and "106%" in params.values()


def test_text_and_date_filters():
    sql, params = where_clause(
        SearchFilters(keywords=["cooling", "plates"], title="50%_off", date_from=date(2015, 1, 1), date_to=date(2016, 1, 1))
    )
    assert "plainto_tsquery('english', %(f0)s)" in sql and params["f0"] == "cooling plates"
    assert params["f1"] == "%50\\%\\_off%"  # LIKE wildcards in the user's text are escaped
    assert "make_date(p.pub_year, 12, 31)) >= %(f2)s" in sql and "make_date(p.pub_year, 1, 1)) <= %(f3)s" in sql

    exact_sql, exact_params = where_clause(SearchFilters(title="  Pneumatic   TIRE ", title_exact=True))
    assert exact_sql == "lower(p.title) = %(f0)s" and exact_params["f0"] == "pneumatic tire"


def test_bad_filters_raise():
    with pytest.raises(ValueError):
        where_clause(SearchFilters(cpc=["60B"]))
    with pytest.raises(ValueError, match="after"):
        where_clause(SearchFilters(date_from=date(2024, 2, 1), date_to=date(2024, 1, 1)))
