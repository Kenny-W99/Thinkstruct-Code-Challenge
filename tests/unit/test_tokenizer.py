from patent_search.tokenizer import highlight_spans, normalize, tokenize


def test_tokenize_lowercases_drops_stopwords_and_numbers():
    assert tokenize("The Tire, wherein 2 grooves comprising RUBBER") == ["tire", "groove", "rubber"]


def test_plural_folding():
    assert normalize("tires") == "tire"
    assert normalize("batteries") == "battery"
    for word in ("process", "status", "axis", "gas"):
        assert normalize(word) == word


def test_glued_reference_numerals_split_from_words():
    assert tokenize("portion1 and sidewall2") == ["portion", "sidewall"]


def test_highlight_spans_use_the_same_normalization():
    text = "Tires with grooves; the tire."
    spans = highlight_spans(text, {"tire"})
    assert [text[a:b] for a, b in spans] == ["Tires", "tire"]
    assert highlight_spans(text, set()) == []
