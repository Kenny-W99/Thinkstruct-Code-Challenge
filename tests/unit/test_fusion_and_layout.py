import numpy as np

from patent_search.fusion import rrf, top_k_indices
from patent_search.index import rows_for_patents


def test_top_k_best_first_with_stable_ties():
    s = np.array([0.1, 0.9, 0.5, 0.9, 0.2])
    assert top_k_indices(s, 3).tolist() == [1, 3, 2]
    assert top_k_indices(s, 99).tolist() == [1, 3, 2, 4, 0]
    assert top_k_indices(s, 0).size == 0 and top_k_indices(np.array([]), 3).size == 0


def test_rrf_rewards_items_ranked_well_by_both_lists():
    items, scores = rrf([np.array([10, 20, 30]), np.array([20, 40, 10])], k=60)
    assert items[:2].tolist() == [20, 10]  # in both lists
    assert set(items.tolist()) == {10, 20, 30, 40}
    assert np.all(np.diff(scores) <= 0)
    assert abs(scores[0] - (1 / 62 + 1 / 61)) < 1e-6


def test_rrf_empty():
    items, scores = rrf([np.array([], dtype=np.int64)])
    assert items.size == 0 and scores.size == 0


def test_rows_for_patents_concatenates_ranges():
    starts = np.array([0, 5, 7, 10, 13])  # 4 patents
    assert rows_for_patents(np.array([1, 3]), starts).tolist() == [5, 6, 10, 11, 12]
    assert rows_for_patents(np.array([], dtype=np.int64), starts).size == 0
    empty_patent_starts = np.array([0, 2, 2, 4])
    assert rows_for_patents(np.array([1, 2]), empty_patent_starts).tolist() == [2, 3]
