"""Text -> vector. Anything with `name`, `dim` and `encode()` is an Embedder.

`HashingEmbedder` needs no model download, so unit/integration tests run fast
and offline. `SentenceTransformerEmbedder` is the real model.
"""

from __future__ import annotations

import time
import zlib
from typing import Callable, Protocol, Sequence

import numpy as np

from .config import BGE_QUERY_PREFIX, EMBEDDING_MODEL
from .tokenizer import tokenize


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: Sequence[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        """Return float32 array (len(texts), dim) with L2-normalized rows."""
        ...


def l2_normalize(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words vectors (each word hashed into a bucket).
    Not 'semantic' at all; it exists so tests don't need a 130 MB model."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: Sequence[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in tokenize(text):
                out[i, zlib.crc32(token.encode()) % self.dim] += 1.0
        return l2_normalize(out)


class SentenceTransformerEmbedder:
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        query_prefix: str | None = None,
        batch_size: int = 64,
        device: str | None = None,
    ):
        import logging

        # torch prints a harmless macOS warning on import; keep CLI output readable
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
        from sentence_transformers import SentenceTransformer  # lazy: importing torch is slow

        try:
            from transformers.utils import logging as hf_logging

            hf_logging.disable_progress_bar()
            hf_logging.set_verbosity_error()
        except Exception:  # cosmetic only
            pass

        self.name = model_name
        self.query_prefix = BGE_QUERY_PREFIX if query_prefix is None and "bge" in model_name.lower() else (query_prefix or "")
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = min(512, self.model.max_seq_length or 512)
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dim = get_dim()

    def encode(self, texts: Sequence[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        )
        return np.asarray(vectors, dtype=np.float32)


class CachedEmbedder:
    """Wraps an embedder and remembers query vectors.

    `prefill()` embeds many queries in one batch (fast on a GPU). Benchmarks and
    the evaluation use it so they measure *search* time, not model time, and
    don't re-embed the same query once per setup.
    """

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name, self.dim = inner.name, inner.dim
        self._cache: dict[str, np.ndarray] = {}

    def prefill(self, queries: Sequence[str], show_progress: bool = False) -> None:
        missing = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if missing:
            self._cache.update(zip(missing, self.inner.encode(missing, is_query=True, show_progress=show_progress)))

    def encode(self, texts: Sequence[str], *, is_query: bool = False, show_progress: bool = False) -> np.ndarray:
        if not is_query:
            return self.inner.encode(texts, show_progress=show_progress)
        self.prefill(texts)
        return np.stack([self._cache[t] for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


def encode_with_progress(
    embedder: Embedder,
    texts: Sequence[str],
    *,
    chunk_size: int = 4096,
    report: Callable[[str], None] = lambda _: None,
) -> np.ndarray:
    """Embed many passages in chunks, reporting progress after each chunk.
    (A progress bar is invisible in `docker logs`; these lines are not.)"""
    parts: list[np.ndarray] = []
    start_time = time.perf_counter()
    for start in range(0, len(texts), chunk_size):
        parts.append(embedder.encode(texts[start : start + chunk_size]))
        done = min(start + chunk_size, len(texts))
        rate = done / max(time.perf_counter() - start_time, 1e-9)
        report(f"embedded {done:,}/{len(texts):,} passages ({rate:.0f}/s, ~{(len(texts) - done) / rate:.0f}s left)")
    return np.concatenate(parts) if parts else np.zeros((0, embedder.dim), np.float32)


def make_embedder(name: str, device: str | None = None) -> Embedder:
    """'hashing-256' -> HashingEmbedder(256); anything else is a Hugging Face model name."""
    if name.startswith("hashing"):
        _, _, dim = name.partition("-")
        return HashingEmbedder(int(dim) if dim else 256)
    return SentenceTransformerEmbedder(name, device=device)
