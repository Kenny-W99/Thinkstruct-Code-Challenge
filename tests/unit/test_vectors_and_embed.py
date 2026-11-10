import numpy as np
import pytest

from patent_search.embed import CachedEmbedder, HashingEmbedder, l2_normalize, make_embedder
from patent_search.vectors import VectorIndex


def test_hashing_embedder_is_deterministic_normalized_and_word_based():
    e = HashingEmbedder(dim=64)
    a, b, c = e.encode(["tire tread noise", "noisy tires tread", "battery cooling plate"])
    np.testing.assert_allclose(e.encode(["tire tread noise"])[0], a)
    assert abs(np.linalg.norm(a) - 1) < 1e-6
    assert a @ b > a @ c  # shares words -> closer
    assert not e.encode([""]).any()  # empty text -> zero vector, no NaN


def test_cached_embedder_embeds_each_query_once():
    calls = []

    class Counting(HashingEmbedder):
        def encode(self, texts, *, is_query=False, show_progress=False):
            calls.append(list(texts))
            return super().encode(texts, is_query=is_query)

    cached = CachedEmbedder(Counting(dim=16))
    cached.prefill(["a b", "c d", "a b"])
    out = cached.encode(["c d", "a b"], is_query=True)
    np.testing.assert_allclose(out, HashingEmbedder(16).encode(["c d", "a b"]))
    cached.encode(["new"], is_query=True)
    cached.encode(["passage"])  # passages are never cached
    assert calls == [["a b", "c d"], ["new"], ["passage"]]
    assert cached.name == "hashing-16" and cached.dim == 16


def test_make_embedder_hashing():
    assert make_embedder("hashing-32").dim == 32


def test_l2_normalize_handles_zero_rows():
    out = l2_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    np.testing.assert_allclose(out, [[0.6, 0.8], [0, 0]])


def test_vector_scores_full_and_rows():
    m = l2_normalize(np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32))
    idx = VectorIndex(m)
    q = np.array([1, 0], dtype=np.float32)
    full = idx.score(q)
    assert full.argmax() == 0
    np.testing.assert_allclose(idx.score(q, np.array([2, 1])), full[[2, 1]])


def test_both_scoring_plans_give_identical_scores():
    rng = np.random.default_rng(0)
    idx = VectorIndex(l2_normalize(rng.normal(size=(50, 8)).astype(np.float32)))
    q = l2_normalize(rng.normal(size=(1, 8)).astype(np.float32))[0]
    few = np.array([3, 7, 9])  # below the threshold: copy the rows
    most = np.arange(5, 50)  # above it: score all, then pick
    np.testing.assert_allclose(idx.score(q, few), idx.score(q)[few], rtol=1e-6)
    np.testing.assert_allclose(idx.score(q, most), idx.score(q)[most], rtol=1e-6)


def test_vector_index_rejects_1d_and_round_trips(tmp_path):
    with pytest.raises(ValueError):
        VectorIndex(np.zeros(3))
    idx = VectorIndex(np.eye(3, dtype=np.float32))
    idx.save(tmp_path)
    np.testing.assert_array_equal(VectorIndex.load(tmp_path).matrix, idx.matrix)


def test_encode_with_progress_matches_one_call_and_reports_each_chunk():
    from patent_search.embed import HashingEmbedder, encode_with_progress

    emb = HashingEmbedder(32)
    texts = ["wheel rim", "tire tread", "trailer hitch", "spoke", "battery pack"]
    lines = []
    out = encode_with_progress(emb, texts, chunk_size=2, report=lines.append)
    assert np.allclose(out, emb.encode(texts))
    assert len(lines) == 3 and lines[-1].startswith("embedded 5/5 passages")
    assert encode_with_progress(emb, []).shape == (0, 32)
