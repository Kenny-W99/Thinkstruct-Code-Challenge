"""Everything that talks to Postgres for the Part 2 PoC.

`PgStore` wraps one connection (autocommit; multi-statement work goes in
`store.transaction()`). The ingest pipeline only uses these methods, so tests
can swap in an in-memory store.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

from .rows import PASSAGE_COLUMNS, PATENT_COLUMNS

SCHEMA = Path(__file__).with_name("schema.sql")
MAX_ATTEMPTS = 3
LEASE_MINUTES = 10  # a 'running' job not updated for this long is taken over (its worker died)


@dataclass(frozen=True)
class Job:
    id: int
    source_file: str
    attempts: int


@dataclass(frozen=True)
class JobCounts:
    records_in: int
    bad_records: int
    duplicates: int
    patents_written: int
    passages: int


class PgStore:
    def __init__(self, conn):
        self.conn = conn

    @classmethod
    def connect(cls, dsn: str) -> "PgStore":
        import psycopg  # lazy: only the PoC needs it

        return cls(psycopg.connect(dsn, autocommit=True))

    def close(self) -> None:
        self.conn.close()

    @contextmanager
    def transaction(self) -> Iterator[None]:
        with self.conn.transaction():
            yield

    def apply_schema(self, dim: int) -> None:
        self.conn.execute(SCHEMA.read_text().replace("{dim}", str(int(dim))))

    # ------------------------------------------------------------ jobs

    def register_files(self, files: Sequence[str]) -> int:
        """Add files to the queue (files already there keep their status)."""
        with self.conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO ingest_jobs (source_file) VALUES (%s) ON CONFLICT (source_file) DO NOTHING",
                [(f,) for f in files],
            )
        return len(files)

    def claim_job(self, max_attempts: int = MAX_ATTEMPTS) -> Job | None:
        """Take the next file: pending, failed with retries left, or abandoned."""
        row = self.conn.execute(
            """
            UPDATE ingest_jobs SET status = 'running', attempts = attempts + 1,
                   started_at = now(), updated_at = now(), error = NULL
            WHERE id = (
                SELECT id FROM ingest_jobs
                WHERE status = 'pending'
                   OR (status = 'failed' AND attempts < %(max)s)
                   OR (status = 'running' AND updated_at < now() - make_interval(mins => %(lease)s))
                ORDER BY id
                FOR UPDATE SKIP LOCKED
                LIMIT 1
            )
            RETURNING id, source_file, attempts
            """,
            {"max": max_attempts, "lease": LEASE_MINUTES},
        ).fetchone()
        return Job(*row) if row else None

    def finish_job(self, job_id: int, counts: JobCounts) -> None:
        self.conn.execute(
            """
            UPDATE ingest_jobs SET status = 'done', finished_at = now(), updated_at = now(),
                   records_in = %s, bad_records = %s, duplicates = %s, patents_written = %s, passages = %s
            WHERE id = %s
            """,
            (counts.records_in, counts.bad_records, counts.duplicates, counts.patents_written, counts.passages, job_id),
        )

    def fail_job(self, job_id: int, error: str) -> None:
        self.conn.execute(
            "UPDATE ingest_jobs SET status = 'failed', error = %s, finished_at = now(), updated_at = now() WHERE id = %s",
            (error[:2000], job_id),
        )

    def job_summary(self) -> dict:
        rows = self.conn.execute(
            """
            SELECT status, count(*), coalesce(sum(records_in), 0), coalesce(sum(patents_written), 0),
                   coalesce(sum(passages), 0), coalesce(sum(duplicates), 0), coalesce(sum(bad_records), 0)
            FROM ingest_jobs GROUP BY status
            """
        ).fetchall()
        summary = {"files": {s: 0 for s in ("pending", "running", "done", "failed")}}
        totals = dict.fromkeys(("records_in", "patents_written", "passages", "duplicates", "bad_records"), 0)
        for status, n, *sums in rows:
            summary["files"][status] = n
            for key, value in zip(totals, sums):
                totals[key] += int(value)
        failed = self.conn.execute(
            "SELECT source_file, attempts, error FROM ingest_jobs WHERE status = 'failed' ORDER BY id LIMIT 5"
        ).fetchall()
        n_patents, n_passages = self.conn.execute(
            "SELECT (SELECT count(*) FROM patents), (SELECT count(*) FROM passages)"
        ).fetchone()
        return {
            **summary,
            **totals,
            "failed_examples": [{"file": f, "attempts": a, "error": e} for f, a, e in failed],
            "db_patents": n_patents,
            "db_passages": n_passages,
        }

    # ------------------------------------------------------------ data

    def upsert_patents(self, rows: Sequence[tuple]) -> set[str]:
        """Insert patents; a patent already stored is replaced only if this copy
        has more text. Returns the doc numbers actually written."""
        if not rows:
            return set()
        cols = ", ".join(PATENT_COLUMNS)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in PATENT_COLUMNS[1:])
        sql = (
            f"INSERT INTO patents ({cols}) VALUES ({', '.join(['%s'] * len(PATENT_COLUMNS))}) "
            f"ON CONFLICT (doc_number) DO UPDATE SET {updates}, ingested_at = now() "
            "WHERE EXCLUDED.text_size > patents.text_size RETURNING doc_number"
        )
        written: set[str] = set()
        with self.conn.cursor() as cur:
            cur.executemany(sql, rows, returning=True)
            while True:
                row = cur.fetchone()
                if row:
                    written.add(row[0])
                if not cur.nextset():
                    break
        return written

    def replace_passages(self, doc_numbers: set[str], rows: Sequence[tuple]) -> int:
        """Drop old passages of these patents, then bulk-load the new ones with COPY."""
        with self.conn.cursor() as cur:
            if doc_numbers:
                cur.execute("DELETE FROM passages WHERE doc_number = ANY(%s)", (list(doc_numbers),))
            with cur.copy(f"COPY passages ({', '.join(PASSAGE_COLUMNS)}) FROM STDIN") as copy:
                for row in rows:
                    copy.write_row(row)
        return len(rows)

    # ------------------------------------------------------------ search

    def search(
        self,
        query_vector: str,
        where: str,
        params: dict,
        *,
        top_k: int = 10,
        candidates: int = 200,
        exact: bool = False,
        explain: bool = False,
    ) -> tuple[list[dict], float, list[str]]:
        """Filtered vector search in ONE query: filters on `patents`, nearest
        passages by cosine distance, best passage per patent.

        HNSW + a WHERE clause can run out of matching neighbours; pgvector's
        iterative scan keeps walking the graph until enough rows pass the filter.
        exact=True turns the index off (an exact scan) for comparison."""
        sql = f"""
            WITH hits AS (
                SELECT s.doc_number, s.label, s.embedding <=> %(q)s::vector AS distance
                FROM passages s JOIN patents p ON p.doc_number = s.doc_number
                WHERE {where}
                ORDER BY s.embedding <=> %(q)s::vector
                LIMIT %(n)s
            )
            SELECT h.doc_number, p.title, p.cpc_raw, coalesce(p.pub_date::text, p.pub_year::text) AS published,
                   min(h.distance) AS distance, (array_agg(h.label ORDER BY h.distance))[1] AS best_passage
            FROM hits h JOIN patents p ON p.doc_number = h.doc_number
            GROUP BY h.doc_number, p.title, p.cpc_raw, p.pub_date, p.pub_year
            ORDER BY distance
            LIMIT %(k)s
        """
        args = {**params, "q": query_vector, "n": candidates, "k": top_k}
        plan: list[str] = []
        with self.conn.transaction():
            self.conn.execute("SET LOCAL hnsw.ef_search = 100")
            self.conn.execute("SET LOCAL hnsw.iterative_scan = relaxed_order")
            if exact:
                self.conn.execute("SET LOCAL enable_indexscan = off")
            t = time.perf_counter()
            cur = self.conn.execute(sql, args)
            names = [d.name for d in cur.description]
            rows = [dict(zip(names, r)) for r in cur.fetchall()]
            ms = (time.perf_counter() - t) * 1000
            if explain:
                plan = [r[0] for r in self.conn.execute("EXPLAIN (ANALYZE, COSTS OFF) " + sql, args).fetchall()]
        return rows, ms, plan
