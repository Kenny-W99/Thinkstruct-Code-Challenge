import pytest

from patent_search.cpc import parse_cpc, parse_cpc_query


@pytest.mark.parametrize(
    "raw, subclass, group, subgroup, digits",
    [
        ("B60B2700FI", "B60B", None, None, "2700"),  # small data, no separator
        ("B60C25138FI", "B60C", None, None, "25138"),
        ("A61B5/097", "A61B", "5", "097", "5097"),  # large data
        ("A01G009/02", "A01G", "9", "02", "902"),  # leading zeros in the group
        (" g06f16/00 ", "G06F", "16", "00", "1600"),
    ],
)
def test_parse_both_formats(raw, subclass, group, subgroup, digits):
    code = parse_cpc(raw)
    assert (code.subclass, code.group, code.subgroup, code.digits) == (subclass, group, subgroup, digits)


@pytest.mark.parametrize("raw", ["N/A", "", None, "garbage", 123])
def test_missing_or_invalid_codes_are_none(raw):
    assert parse_cpc(raw) is None


def test_display():
    assert parse_cpc("A61B5/097").display == "A61B 5/097"
    assert parse_cpc("B60B2700FI").display == "B60B 2700"


def test_query_parsing_normalizes_spaces_case_and_zeros():
    q = parse_cpc_query(" b60b 027/0 ")
    assert (q.prefix, q.group, q.subgroup, q.has_slash) == ("B60B", "27", "0", True)


@pytest.mark.parametrize("bad", ["", "60B", "B602", "B60B/00", "B6/1"])
def test_invalid_queries_raise(bad):
    with pytest.raises(ValueError):
        parse_cpc_query(bad)


@pytest.mark.parametrize(
    "query, code, expected",
    [
        ("B60B", "B60B2700FI", True),
        ("B60", "B60C1103FI", True),  # class-level prefix
        ("B", "B60D100FI", True),  # section-level prefix
        ("B60B", "B60C1103FI", False),
        ("A61B5", "A61B5/097", True),  # structured: exact group
        ("A61B5", "A61B50/20", False),  # group 50 is not group 5
        ("A61B 5/09", "A61B5/097", True),  # subgroup prefix
        ("A61B 5/1", "A61B5/097", False),
        ("B60B27", "B60B2700FI", True),  # unstructured: best-effort digit prefix
        ("B60B 27/00", "B60B2700FI", True),
        ("B60B", "N/A", False),  # missing code never matches
    ],
)
def test_matching(query, code, expected):
    assert parse_cpc_query(query).matches(parse_cpc(code)) is expected
