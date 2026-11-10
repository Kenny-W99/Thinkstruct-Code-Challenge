"""Ranking helpers: top-k selection and Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Positions of the k highest scores, best first (ties broken by position).
    Uses argpartition, so it is O(n) instead of a full O(n log n) sort."""
    if k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)
    k = min(k, scores.size)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.lexsort((part, -scores[part]))]


def rrf(rankings: Sequence[np.ndarray], k: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Merge ranked lists by position: score(item) = sum over lists of 1 / (k + rank).

    Works on ranks, not raw scores, so BM25 scores (0..30) and cosine
    similarities (0..1) can be combined without any tuning.
    Returns (items best-first, their fused scores).
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(np.asarray(ranking).tolist(), start=1):
            fused[item] = fused.get(item, 0.0) + 1.0 / (k + rank)
    items = sorted(fused, key=lambda i: (-fused[i], i))
    return np.array(items, dtype=np.int64), np.array([fused[i] for i in items], dtype=np.float32)
