"""Build, save and load everything the search engine needs.

On disk (artifacts/<name>/):
    meta.json        how it was built (model, chunking, counts, timings, load report)
    patents.jsonl    one cleaned patent per line (the single source of text)
    bm25.npz/.json   keyword index
    vectors.npy      one embedding per passage

Passages are NOT stored: they are re-made from patents.jsonl at load time
with the saved chunking settings (fast and deterministic), so the text lives
in exactly one place.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

from .bm25 import BM25Index
from .chunk import KINDS, ChunkConfig, Passage, chunk_patent, index_text
from .embed import Embedder, encode_with_progress
from .filters import MetadataIndex
from .load import Patent, patent_from_dict, patent_to_dict
from .tokenizer import tokenize
from .vectors import VectorIndex

log = logging.getLogger(__name__)


def build_passages(patents: Sequence[Patent], config: ChunkConfig) -> tuple[list[Passage], np.ndarray]:
    """All passages, grouped by patent, plus `starts`: patent p owns rows
    starts[p]:starts[p+1]. Keeping a patent's passages contiguous makes
    "patents -> passage rows" a cheap range lookup."""
    passages: list[Passage] = []
    starts = np.zeros(len(patents) + 1, dtype=np.int64)
    for i, patent in enumerate(patents):
        passages.extend(chunk_patent(patent, i, config))
        starts[i + 1] = len(passages)
    return passages, starts


def rows_for_patents(patent_ids: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """Passage rows of the given patents, without a Python loop.
    e.g. ranges [5,7) and [10,13) -> [5, 6, 10, 11, 12]."""
    patent_ids = np.asarray(patent_ids, dtype=np.int64)
    lo, hi = starts[patent_ids], starts[patent_ids + 1]
    lengths = hi - lo
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    offsets = np.repeat(lo - np.concatenate(([0], np.cumsum(lengths)[:-1])), lengths)
    return np.arange(total, dtype=np.int64) + offsets


class PatentIndex:
    def __init__(
        self,
        patents: list[Patent],
        passages: list[Passage],
        starts: np.ndarray,
        bm25: BM25Index,
        vectors: VectorIndex,
        meta: dict,
    ):
        if len(passages) != len(vectors) or len(passages) != bm25.n_docs:
            raise ValueError(
                f"index is inconsistent: {len(passages)} passages, {len(vectors)} vectors, {bm25.n_docs} bm25 rows"
            )
        self.patents = patents
        self.passages = passages
        self.starts = starts
        self.bm25 = bm25
        self.vectors = vectors
        self.meta = meta
        self.metadata = MetadataIndex(patents)
        self.doc_to_idx = {p.doc_number: i for i, p in enumerate(patents)}
        self.passage_patent = np.repeat(np.arange(len(patents), dtype=np.int32), np.diff(starts))
        self.passage_kind = np.array([KINDS.index(p.kind) for p in passages], dtype=np.int8)

    @property
    def chunk_config(self) -> ChunkConfig:
        return ChunkConfig(**self.meta["chunking"])

    # ------------------------------------------------------------ build

    @classmethod
    def build(
        cls,
        patents: list[Patent],
        embedder: Embedder | None,
        config: ChunkConfig = ChunkConfig(),
        *,
        extra_meta: dict | None = None,
        show_progress: bool = False,
    ) -> "PatentIndex":
        timings: dict[str, float] = {}

        t = time.perf_counter()
        passages, starts = build_passages(patents, config)
        texts = [index_text(p, patents[p.patent_idx].title, config.title_context) for p in passages]
        timings["chunk"] = time.perf_counter() - t

        t = time.perf_counter()
        bm25 = BM25Index.build(tokenize(text) for text in texts)
        timings["bm25"] = time.perf_counter() - t

        t = time.perf_counter()
        if embedder is None:  # keyword-only index (e.g. all 457k large-data patents)
            matrix = np.zeros((len(texts), 0), np.float32)
        else:
            log.info("embedding %s passages with %s", f"{len(texts):,}", embedder.name)
            matrix = encode_with_progress(embedder, texts, report=log.info if show_progress else lambda _: None)
        vectors = VectorIndex(np.asarray(matrix, dtype=np.float32))
        timings["embed"] = time.perf_counter() - t

        meta = {
            "embedder": embedder.name if embedder else None,
            "dim": embedder.dim if embedder else 0,
            "chunking": asdict(config),
            "n_patents": len(patents),
            "n_passages": len(passages),
            "passages_by_kind": {k: sum(p.kind == k for p in passages) for k in KINDS},
            "build_seconds": {k: round(v, 3) for k, v in timings.items()},
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **(extra_meta or {}),
        }
        return cls(patents, passages, starts, bm25, vectors, meta)

    # ------------------------------------------------------------ persistence

    def save(self, directory: Path | str) -> Path:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / "patents.jsonl", "w", encoding="utf-8") as f:
            for p in self.patents:
                f.write(json.dumps(patent_to_dict(p), ensure_ascii=False) + "\n")
        self.bm25.save(directory)
        self.vectors.save(directory)
        (directory / "meta.json").write_text(json.dumps(self.meta, indent=2))
        return directory

    @classmethod
    def load(cls, directory: Path | str) -> "PatentIndex":
        directory = Path(directory)
        if not (directory / "meta.json").exists():
            raise FileNotFoundError(f"No index at {directory}. Build one first: python cli.py build --dataset small")
        meta = json.loads((directory / "meta.json").read_text())
        with open(directory / "patents.jsonl", encoding="utf-8") as f:
            patents = [patent_from_dict(json.loads(line)) for line in f]
        passages, starts = build_passages(patents, ChunkConfig(**meta["chunking"]))
        return cls(patents, passages, starts, BM25Index.load(directory), VectorIndex.load(directory), meta)
