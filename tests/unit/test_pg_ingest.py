"""Part 2 PoC pipeline with an in-memory store: jobs, retries, duplicates,
bad files, and all-or-nothing loading of a file."""

import copy
from contextlib import contextmanager

import pytest

from patent_search.embed import HashingEmbedder
from patent_search.pg.ingest import run_ingest
from patent_search.pg.store import MAX_ATTEMPTS, Job


class MemoryStore:
    """Same methods as PgStore; transaction() restores everything on error."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.patents: dict[str, tuple] = {}  # doc -> row
        self.passages: dict[str, list] = {}  # doc -> rows

    @contextmanager
    def transaction(self):
        saved = copy.deepcopy((self.jobs, self.patents, self.passages))
        try:
            yield
        except Exception:
            self.jobs, self.patents, self.passages = saved
            raise

    def register_files(self, files):
        for f in files:
            self.jobs.setdefault(f, {"id": len(self.jobs) + 1, "status": "pending", "attempts": 0})
        return len(files)

    def claim_job(self):
        for name, j in self.jobs.items():
            if j["status"] == "pending" or (j["status"] == "failed" and j["attempts"] < MAX_ATTEMPTS):
                j.update(status="running", attempts=j["attempts"] + 1)
                return Job(j["id"], name, j["attempts"])
        return None

    def _job(self, job_id):
        return next(j for j in self.jobs.values() if j["id"] == job_id)

    def finish_job(self, job_id, counts):
        self._job(job_id).update(status="done", counts=counts)

    def fail_job(self, job_id, error):
        self._job(job_id).update(status="failed", error=error)

    def upsert_patents(self, rows):
        written = set()
        for row in rows:
            old = self.patents.get(row[0])
            if old is None or row[-1] > old[-1]:  # last column = text_size
                self.patents[row[0]] = row
                written.add(row[0])
        return written

    def replace_passages(self, doc_numbers, rows):
        for d in doc_numbers:
            self.passages[d] = []
        for row in rows:
            self.passages[row[0]].append(row)
        return len(rows)


@pytest.fixture
def store():
    return MemoryStore()


def _run(store, root, embedder=None):
    logs = []
    n = run_ingest(store, root, embedder or HashingEmbedder(16), log=logs.append)
    return n, logs


def test_ingest_large_layout(store, large_dir):
    processed, logs = _run(store, large_dir)
    assert set(store.jobs) == {
        "job_000/patents_batch_0001.json",
        "job_001/patents_batch_0001.json",
        "job_001/patents_batch_0002.json",  # truncated JSON
    }  # sample.json is not a batch file
    assert processed == 2 + MAX_ATTEMPTS  # the broken file is retried, then left failed
    broken = store.jobs["job_001/patents_batch_0002.json"]
    assert broken["status"] == "failed" and broken["attempts"] == MAX_ATTEMPTS
    assert "JSONDecodeError" in broken["error"]
    assert any(line.startswith("FAILED") for line in logs)

    first = store.jobs["job_000/patents_batch_0001.json"]["counts"]
    assert (first.records_in, first.bad_records, first.patents_written) == (5, 2, 3)

    # The charger appears twice; the longer copy (second file) replaced the short one.
    second = store.jobs["job_001/patents_batch_0001.json"]["counts"]
    assert second.patents_written == 1 and second.duplicates == 0
    assert set(store.patents) == {"20150000001", "20180000002", "20100000003"}
    charger_abstract = [r[3] for r in store.passages["20180000002"] if r[1] == "abstract"][0]
    assert "induction" in charger_abstract


def test_a_shorter_duplicate_does_not_overwrite(store, large_dir, tmp_path):
    _run(store, large_dir)
    before = dict(store.patents)
    later = tmp_path / "later"
    later.mkdir()
    (later / "patents_batch_0009.json").write_text((large_dir / "job_000" / "patents_batch_0001.json").read_text())
    _run(store, later)
    counts = store.jobs["patents_batch_0009.json"]["counts"]
    assert counts.patents_written == 0 and counts.duplicates == 3
    assert store.patents == before


def test_running_again_skips_finished_files(store, small_dir):
    first, _ = _run(store, small_dir)
    assert first == 2 and all(j["status"] == "done" for j in store.jobs.values())
    again, _ = _run(store, small_dir)
    assert again == 0


class FlakyEmbedder(HashingEmbedder):
    """Fails the first call, like a worker crash or a model timeout."""

    def __init__(self):
        super().__init__(16)
        self.calls = 0

    def encode(self, texts, **kw):
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("model timed out")
        return super().encode(texts, **kw)


def test_a_failed_file_leaves_nothing_behind_and_is_retried(store, small_dir):
    processed, logs = _run(store, small_dir, FlakyEmbedder())
    assert processed == 3  # file 1 fails, file 2 ok, file 1 retried ok
    assert "TimeoutError: model timed out" in logs[1]
    retried = store.jobs["patents_ipa240215.json"]
    assert retried["status"] == "done" and retried["attempts"] == 2
    assert len(store.patents) == 4
    assert all(store.passages[d] for d in store.patents)  # every patent has its passages
