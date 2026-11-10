"""Exact vector search with a plain numpy matrix.

At 32k passages x 384 dims (~50 MB) one matrix multiply takes a few ms, so an
approximate index (FAISS/HNSW) would add complexity for no gain here. Part 2's
design doc covers what changes at millions of passages.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Above this share of selected rows, score all rows instead of copying the selected ones.
GATHER_MAX_FRACTION = 0.3


class VectorIndex:
    def __init__(self, matrix: np.ndarray):
        if matrix.ndim != 2:
            raise ValueError("vector matrix must be 2-D")
        self.matrix = matrix

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __len__(self) -> int:
        return self.matrix.shape[0]

    def score(self, query: np.ndarray, rows: np.ndarray | None = None) -> np.ndarray:
        """Cosine similarity (rows are L2-normalized) of every row, or only `rows`.

        A tiny query planner: copying the selected rows out of the matrix costs
        about as much per row as multiplying it. So when MOST rows are selected
        (a broad filter), scoring everything and picking the selected scores is
        cheaper than copying; when FEW are selected, copying wins by far.
        """
        q = query.astype(self.matrix.dtype, copy=False)
        if rows is None:
            return (self.matrix @ q).astype(np.float32, copy=False)
        if rows.size > GATHER_MAX_FRACTION * len(self):
            return (self.matrix @ q)[rows].astype(np.float32, copy=False)
        return (self.matrix[rows] @ q).astype(np.float32, copy=False)

    def save(self, directory: Path) -> None:
        np.save(directory / "vectors.npy", self.matrix)

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        return cls(np.load(directory / "vectors.npy"))
