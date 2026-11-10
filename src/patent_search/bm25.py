"""A small BM25 keyword index on a sparse matrix.

All BM25 math is done once at build time: matrix[d, t] holds the full BM25
weight of term t in passage d. A query score is then just the sum of the
query terms' columns, and the cost depends on how many passages contain
those terms, not on the total number of passages.
"""

from __future__ import annotations

import json
from array import array
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse


class BM25Index:
    def __init__(self, matrix: sparse.csc_matrix, vocab: dict[str, int], k1: float, b: float):
        self.matrix = matrix  # shape (n_passages, n_terms), CSC so column slices are fast
        self.vocab = vocab
        self.k1, self.b = k1, b

    @property
    def n_docs(self) -> int:
        return self.matrix.shape[0]

    @classmethod
    def build(cls, docs: Iterable[list[str]], k1: float = 1.5, b: float = 0.75) -> "BM25Index":
        vocab: dict[str, int] = {}
        # array() instead of lists: ~4 bytes per entry, matters at millions of passages
        rows, cols, tfs, lengths = array("i"), array("i"), array("f"), array("f")
        for d, tokens in enumerate(docs):
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                rows.append(d)
                cols.append(vocab.setdefault(term, len(vocab)))
                tfs.append(tf)
        n_docs, n_terms = len(lengths), len(vocab)
        r = np.frombuffer(rows, dtype=np.int32)
        c = np.frombuffer(cols, dtype=np.int32)
        tf = np.frombuffer(tfs, dtype=np.float32)
        dl = np.frombuffer(lengths, dtype=np.float32)
        avgdl = float(dl.mean()) if n_docs and dl.mean() > 0 else 1.0
        df = np.bincount(c, minlength=n_terms).astype(np.float32)
        idf = np.log1p((n_docs - df + 0.5) / (df + 0.5))
        weights = idf[c] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[r] / avgdl))
        matrix = sparse.csc_matrix((weights.astype(np.float32), (r, c)), shape=(n_docs, n_terms))
        return cls(matrix, vocab, k1, b)

    def query_columns(self, tokens: list[str]) -> list[int]:
        return sorted({self.vocab[t] for t in tokens if t in self.vocab})

    def score(self, tokens: list[str], rows: np.ndarray | None = None) -> np.ndarray:
        """BM25 score of every passage (or only `rows`) for the query tokens."""
        cols = self.query_columns(tokens)
        size = self.n_docs if rows is None else len(rows)
        if not cols or size == 0:
            return np.zeros(size, dtype=np.float32)
        scores = np.asarray(self.matrix[:, cols].sum(axis=1), dtype=np.float32).ravel()
        return scores if rows is None else scores[rows]

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path) -> None:
        sparse.save_npz(directory / "bm25.npz", self.matrix)
        (directory / "bm25.json").write_text(json.dumps({"k1": self.k1, "b": self.b, "vocab": self.vocab}))

    @classmethod
    def load(cls, directory: Path) -> "BM25Index":
        meta = json.loads((directory / "bm25.json").read_text())
        return cls(sparse.load_npz(directory / "bm25.npz").tocsc(), meta["vocab"], meta["k1"], meta["b"])
