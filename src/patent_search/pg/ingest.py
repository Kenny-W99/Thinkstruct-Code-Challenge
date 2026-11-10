"""Part 2 PoC ingest pipeline: batch files -> jobs -> patents + passages.

    for each file (one job):  read -> parse/clean -> dedupe -> upsert patents
                              -> chunk + embed the ones written -> COPY passages
                              -> mark the job done (or failed, with the error)

A file is loaded in one transaction, so it is either fully in or not at all,
and a retry never leaves half a file behind. Several of these workers can run
at once against the same database (jobs are claimed with SKIP LOCKED).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Protocol, Sequence

import numpy as np

from ..chunk import ChunkConfig
from ..embed import Embedder
from ..load import BadRecord, Patent, dedupe, find_data_files, parse_record, read_json_file
from .rows import POC_CHUNK_CONFIG, patent_row, poc_passages, vector_literal
from .store import Job, JobCounts


class Store(Protocol):
    def transaction(self): ...
    def register_files(self, files: Sequence[str]) -> int: ...
    def claim_job(self) -> Job | None: ...
    def finish_job(self, job_id: int, counts: JobCounts) -> None: ...
    def fail_job(self, job_id: int, error: str) -> None: ...
    def upsert_patents(self, rows: Sequence[tuple]) -> set[str]: ...
    def replace_passages(self, doc_numbers: set[str], rows: Sequence[tuple]) -> int: ...


def read_batch(path: Path, source_file: str) -> tuple[list[Patent], int, int, int]:
    """(unique patents, records in file, bad records, duplicates within the file)."""
    raw = read_json_file(path)
    patents: list[Patent] = []
    bad = 0
    for record in raw:
        try:
            patents.append(parse_record(record, source_file=source_file, keep_description=False))
        except BadRecord:
            bad += 1
    unique, dropped = dedupe(patents)
    return unique, len(raw), bad, dropped


def ingest_file(store: Store, root: Path, source_file: str, embedder: Embedder, config: ChunkConfig) -> JobCounts:
    patents, n_in, bad, dup_in_file = read_batch(root / source_file, source_file)
    written = store.upsert_patents([patent_row(p, source_file) for p in patents])
    pairs = [(p, ps, text) for p in patents if p.doc_number in written for ps, text in poc_passages(p, config)]
    vectors = embedder.encode([text for _, _, text in pairs]) if pairs else np.zeros((0, embedder.dim), np.float32)
    rows = [(p.doc_number, ps.kind, ps.label, ps.text, vector_literal(v)) for (p, ps, _), v in zip(pairs, vectors)]
    store.replace_passages(written, rows)
    # duplicates = copies dropped in this file + copies that lost to a stored patent
    return JobCounts(n_in, bad, dup_in_file + len(patents) - len(written), len(written), len(rows))


def run_ingest(
    store: Store,
    root: Path,
    embedder: Embedder,
    *,
    max_files: int | None = None,
    config: ChunkConfig = POC_CHUNK_CONFIG,
    log: Callable[[str], None] = print,
) -> int:
    """Queue the files under root, then work through the queue. Returns jobs processed."""
    files = [str(f.relative_to(root)) for f in find_data_files(root)][:max_files]
    store.register_files(files)
    log(f"queued {len(files)} files from {root}")
    processed = 0
    while (job := store.claim_job()) is not None:
        t = time.perf_counter()
        try:
            with store.transaction():
                counts = ingest_file(store, root, job.source_file, embedder, config)
                store.finish_job(job.id, counts)
            log(
                f"done   {job.source_file}: {counts.patents_written} patents, {counts.passages} passages, "
                f"{counts.duplicates} duplicates, {counts.bad_records} bad ({time.perf_counter() - t:.1f}s)"
            )
        except Exception as err:  # noqa: BLE001 - any failure marks the job, the loop goes on
            store.fail_job(job.id, f"{type(err).__name__}: {err}")
            log(f"FAILED {job.source_file} (attempt {job.attempts}): {type(err).__name__}: {err}")
        processed += 1
    return processed
