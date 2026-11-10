import pytest

from patent_search.clean import clean_text, is_missing, parse_claims


@pytest.mark.parametrize("value", [None, "", "  ", "N/A", "n/a", [], ["", "N/A"]])
def test_missing_values(value):
    assert is_missing(value)


@pytest.mark.parametrize("value", ["NA tire", "x", ["", "text"], 0])
def test_present_values(value):
    assert not is_missing(value)


def test_clean_text_unglues_figure_references_and_collapses_space():
    assert clean_text("As illustrated inFIG.1is a   tire\n") == "As illustrated in FIG.1 is a tire"
    assert clean_text("see FIGS.2A and 3") == "see FIGS.2A and 3"


def test_claims_small_data_style():
    claims = parse_claims(
        [
            "1 - 5 . (canceled)",
            "a tread portion; and a sidewall portion",  # number missing -> inferred 6
            "7 . The pneumatic tire according to claim 6 , wherein the rubber is soft.",
            "",
        ]
    )
    assert [(c.number, c.depends_on) for c in claims] == [(6, ()), (7, (6,))]
    assert claims[0].is_independent and not claims[1].is_independent
    assert claims[1].text.startswith("The pneumatic tire")


def test_claims_large_data_style():
    claims = parse_claims(["1. A method for X.", "2. The method of claim 1, wherein Y.", "3. (Cancelled)", "4. A system for Z."])
    assert [(c.number, c.depends_on) for c in claims] == [(1, ()), (2, (1,)), (4, ())]


def test_decimal_numbers_are_not_claim_numbers():
    (claim,) = parse_claims(["2.5 mm thick layer of rubber"])
    assert claim.number == 1 and claim.text.startswith("2.5 mm")


def test_dependency_on_multiple_claims_and_no_forward_refs():
    claims = parse_claims(["1. A.", "2. B.", "3. The device of claim 1 or 2.", "4. Unlike claim 9, a C."])
    assert claims[2].depends_on == (1, 2)
    assert claims[3].depends_on == ()  # a claim can't depend on a later claim
