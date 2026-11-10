"""Part 2 PoC against a real Postgres + pgvector.

Skipped unless PATENT_PG_DSN is set, e.g. with the compose database running:
    PATENT_PG_DSN=postgresql://patents:patents@localhost:5432/patents pytest -m postgres
Each run uses its own throwaway schema.
"""

import os
import uuid

import pytest

from patent_search.embed import HashingEmbedder
from patent_search.filters import SearchFilters
from patent_search.pg.ingest import run_ingest
from patent_search.pg.rows import vector_literal, where_clause
from patent_search.pg.store import MAX_ATTEMPTS, PgStore

DSN = os.environ.get("PATENT_PG_DSN")
pytestmark = [pytest.mark.postgres, pytest.mark.skipif(not DSN, reason="set PATENT_PG_DSN to run")]

EMBEDDER = HashingEmbedder(64)


def _store(schema: str) -> PgStore:
    store = PgStore.connect(DSN)
    store.conn.execute(f"SET search_path TO {schema}, public")
    return store


@pytest.fixture(scope="module")
def schema():
    name = f"test_{uuid.uuid4().hex[:8]}"
    admin = PgStore.connect(DSN)
    admin.conn.execute(f"CREATE SCHEMA {name}")
    try:
        yield name
    finally:
        admin.conn.execute(f"DROP SCHEMA {name} CASCADE")
        admin.close()


@pytest.fixture(scope="module")
def loaded(schema, small_dir, large_dir):
    store = _store(schema)
    store.apply_schema(EMBEDDER.dim)
    store.apply_schema(EMBEDDER.dim)  # running it twice is safe
    run_ingest(store, small_dir, EMBEDDER, log=lambda _: None)
    run_ingest(store, large_dir, EMBEDDER, log=lambda _: None)
    yield store
    store.close()


def _search(store, query, **filters):
    where, params = where_clause(SearchFilters(**filters))
    vec = vector_literal(EMBEDDER.encode([query], is_query=True)[0])
    rows, ms, _ = store.search(vec, where, params, top_k=10)
    return [r["doc_number"] for r in rows]


def test_jobs_and_counts(loaded):
    s = loaded.job_summary()
    assert s["files"] == {"pending": 0, "running": 0, "done": 4, "failed": 1}
    assert s["failed_examples"][0]["attempts"] == MAX_ATTEMPTS
    assert s["db_patents"] == 7  # 4 small + battery, charger, wheel
    assert s["bad_records"] == 2


def test_longer_duplicate_wins(loaded):
    abstract = loaded.conn.execute("SELECT abstract FROM patents WHERE doc_number = '20180000002'").fetchone()[0]
    assert "induction" in abstract
    n = loaded.conn.execute("SELECT count(*) FROM passages WHERE doc_number = '20180000002'").fetchone()[0]
    assert n == 2  # abstract + claim 1 of the long copy only (old passages replaced)


def test_filters_match_the_in_memory_engine(loaded):
    assert set(_search(loaded, "wheel spokes", cpc=["B60B"])) == {"20240000002", "20240000004", "20100000003"}
    assert set(_search(loaded, "wheel", cpc=["B60B 3/02"])) == {"20100000003"}
    assert _search(loaded, "cooling", keywords=["plates"]) == ["20150000001"]  # plural folding / stemming
    assert set(_search(loaded, "tire", title="tire")) == {"20240000001"}
    from datetime import date

    assert set(_search(loaded, "anything", date_from=date(2015, 6, 1), date_to=date(2015, 6, 30))) == {"20150000001"}


def test_exact_scan_and_index_agree_on_tiny_data(loaded):
    where, params = where_clause(SearchFilters(cpc=["B60"]))
    vec = vector_literal(EMBEDDER.encode(["carbon fiber spokes"], is_query=True)[0])
    hnsw, _, plan = loaded.search(vec, where, params, explain=True)
    exact, _, _ = loaded.search(vec, where, params, exact=True)
    assert [r["doc_number"] for r in hnsw] == [r["doc_number"] for r in exact]
    assert plan and any("passages" in line for line in plan)


def test_two_workers_never_get_the_same_job(loaded, schema):
    a, b = _store(schema), _store(schema)
    try:
        a.register_files(["w1.json", "w2.json"])
        with a.conn.transaction(), b.conn.transaction():
            first, second = a.claim_job(), b.claim_job()
        assert first and second and first.source_file != second.source_file
    finally:
        a.close()
        b.close()
