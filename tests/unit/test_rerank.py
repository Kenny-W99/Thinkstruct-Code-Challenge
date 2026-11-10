import numpy as np

from patent_search.rerank import OverlapReranker, make_reranker, rerank_hits
from patent_search.results import PassageHit, PatentHit


def passage(text: str, row: int = 0) -> PassageHit:
    return PassageHit(row, "claim", f"Claim {row}", text, 0.5, None, None, [])


def patent(rank: int, doc: str, *texts: str) -> PatentHit:
    return PatentHit(rank, doc, None, None, None, None, True, 1.0 / rank, [passage(t, i) for i, t in enumerate(texts)])


class FixedReranker:
    """Scores passages from a lookup table, to test the reordering logic exactly."""

    name = "fixed"

    def __init__(self, table: dict[str, float]):
        self.table = table

    def score(self, query, passages):
        return np.array([self.table[p] for p in passages], dtype=np.float32)


def test_overlap_reranker_scores_share_of_query_words():
    r = OverlapReranker()
    scores = r.score("tire tread noise", ["tire tread", "battery", "noisy tires tread noise"])
    np.testing.assert_allclose(scores, [2 / 3, 0, 1])
    assert not r.score("the of", ["tire"]).any()  # stopword-only query
    assert make_reranker("overlap").name == "overlap"


def test_rerank_reorders_patents_by_best_passage_and_keeps_old_rank():
    hits = [patent(1, "A", "a1", "a2"), patent(2, "B", "b1"), patent(3, "C", "c1")]
    out = rerank_hits("q", hits, FixedReranker({"a1": 0.1, "a2": 0.2, "b1": 0.9, "c1": 0.5}), top_k=10, keep_passages=3)
    assert [h.doc_number for h in out] == ["B", "C", "A"]
    assert [h.rank for h in out] == [1, 2, 3]
    assert [h.first_stage_rank for h in out] == [2, 3, 1]
    assert out[2].score == np.float32(0.2)
    assert [p.text for p in out[2].passages] == ["a2", "a1"]  # best passage first
    assert out[2].passages[0].rerank_score == np.float32(0.2)


def test_rerank_trims_to_top_k_and_keep_passages_without_touching_input():
    hits = [patent(1, "A", "a1", "a2"), patent(2, "B", "b1")]
    out = rerank_hits("q", hits, FixedReranker({"a1": 1, "a2": 2, "b1": 0}), top_k=1, keep_passages=1)
    assert [h.doc_number for h in out] == ["A"] and len(out[0].passages) == 1
    assert hits[0].first_stage_rank is None and len(hits[0].passages) == 2  # input unchanged


def test_ties_keep_phase1_order():
    hits = [patent(1, "A", "x"), patent(2, "B", "y")]
    out = rerank_hits("q", hits, FixedReranker({"x": 1, "y": 1}), top_k=2, keep_passages=1)
    assert [h.doc_number for h in out] == ["A", "B"]


def test_custom_passage_text_is_what_gets_scored():
    seen = []

    class Spy:
        name = "spy"

        def score(self, query, passages):
            seen.extend(passages)
            return np.zeros(len(passages), dtype=np.float32)

    rerank_hits("q", [patent(1, "A", "body")], Spy(), top_k=1, keep_passages=1, passage_text=lambda h, p: f"T: {p.text}")
    assert seen == ["T: body"]


def test_empty_input():
    assert rerank_hits("q", [], OverlapReranker(), top_k=5, keep_passages=1) == []
