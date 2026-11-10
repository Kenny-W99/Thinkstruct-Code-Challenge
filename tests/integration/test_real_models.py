"""Smoke tests with the REAL models. Skipped by default (they need a model
download); run them with:  pytest -m slow"""

import numpy as np
import pytest

pytestmark = pytest.mark.slow


def test_real_embedder_puts_similar_texts_closer():
    from patent_search.embed import SentenceTransformerEmbedder

    e = SentenceTransformerEmbedder(device="cpu")
    q = e.encode(["tire that is quiet on the road"], is_query=True)[0]
    near, far = e.encode(["a tread pattern that reduces road noise", "a battery pack with cooling plates"])
    assert e.dim == 384 and abs(np.linalg.norm(q) - 1) < 1e-4
    assert q @ near > q @ far


def test_real_cross_encoder_prefers_the_relevant_passage():
    from patent_search.rerank import CrossEncoderReranker

    r = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu", batch_size=2)
    scores = r.score(
        "bicycle spoke made from a fiber cord",
        ["The spoke rope is made of a textile fiber material.", "The battery is cooled by liquid.", "A tire tread."],
    )
    assert scores.shape == (3,) and scores.argmax() == 0
