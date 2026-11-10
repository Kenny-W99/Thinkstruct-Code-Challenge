"""Build an index from data, or load one into a ready search engine.

Shared by the CLI (`build`, `search`, `serve`) and the API server, so they all
load models the same way.
"""

from __future__ import annotations

import time
from pathlib import Path

from .chunk import ChunkConfig
from .config import EMBEDDING_MODEL, PROJECT_ROOT, QUERY_DEVICE, RERANK_DEVICE
from .index import PatentIndex
from .load import LoadReport, load_patents
from .search import SearchEngine


def portable_path(path: Path | str) -> str:
    """A path as written into saved files: relative to the project when it is inside it,
    so results don't carry this machine's folder names. (Symlinks are not followed.)"""
    path = Path(path).absolute()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def build_index(
    data_dir: Path,
    out: Path,
    *,
    config: ChunkConfig = ChunkConfig(),
    embedder_name: str = EMBEDDING_MODEL,
    max_files: int | None = None,
    dataset: str | None = None,
    show_progress: bool = True,
) -> tuple[PatentIndex, LoadReport]:
    """Load the data, build the index and save it to `out`.
    embedder_name="none" builds a keyword-only index."""
    from .embed import make_embedder

    t = time.perf_counter()
    patents, report = load_patents(
        data_dir,
        max_files=max_files,
        keep_description=config.include_description,
        keep_claims=config.include_claims,
    )
    embedder = None if embedder_name == "none" else make_embedder(embedder_name)
    index = PatentIndex.build(
        patents,
        embedder,
        config,
        extra_meta={"dataset": dataset, "data_dir": portable_path(data_dir), "load_report": report.to_dict()},
        show_progress=show_progress,
    )
    index.meta["build_seconds"]["load"] = round(report.seconds, 3)
    index.meta["build_seconds"]["total"] = round(time.perf_counter() - t, 3)
    index.save(out)
    return index, report


def load_engine(
    index_dir: Path,
    embedder_name: str | None = None,
    device: str | None = QUERY_DEVICE,
    reranker_name: str | None = None,
) -> SearchEngine:
    """Load an index plus the embedder it was built with (on the query device),
    and optionally a reranker for two-phase search."""
    from .embed import make_embedder
    from .rerank import make_reranker

    index = PatentIndex.load(index_dir)
    name = embedder_name or index.meta["embedder"]
    embedder = make_embedder(name, device=device) if name else None  # None = keyword-only index
    reranker = make_reranker(reranker_name, device=RERANK_DEVICE) if reranker_name else None
    return SearchEngine(index, embedder, reranker)
