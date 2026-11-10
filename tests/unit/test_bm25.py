import numpy as np

from patent_search.bm25 import BM25Index

DOCS = [
    ["tire", "tread", "groove"],
    ["wheel", "spoke", "rim"],
    ["tire", "tire", "noise"],
    ["battery", "cell"],
]


def test_ranks_matching_docs_and_zero_for_the_rest():
    s = BM25Index.build(DOCS).score(["tire", "noise"])
    assert s.shape == (4,)
    assert s[2] > s[0] > 0 and s[1] == 0 and s[3] == 0


def test_rare_terms_weigh_more_than_common_terms():
    idx = BM25Index.build(DOCS)
    assert idx.score(["noise"])[2] > idx.score(["tire"])[0]


def test_unknown_terms_and_empty_query_score_zero():
    idx = BM25Index.build(DOCS)
    assert not idx.score(["zebra"]).any() and not idx.score([]).any()


def test_rows_returns_same_scores_as_full_then_select():
    idx = BM25Index.build(DOCS)
    rows = np.array([3, 0, 2])
    np.testing.assert_allclose(idx.score(["tire"], rows), idx.score(["tire"])[rows])
    assert idx.score(["tire"], np.array([], dtype=np.int64)).shape == (0,)


def test_empty_corpus():
    idx = BM25Index.build([])
    assert idx.n_docs == 0 and idx.score(["tire"]).shape == (0,)


def test_save_load_round_trip(tmp_path):
    idx = BM25Index.build(DOCS)
    idx.save(tmp_path)
    again = BM25Index.load(tmp_path)
    np.testing.assert_allclose(again.score(["tire", "rim"]), idx.score(["tire", "rim"]))
