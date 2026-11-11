# Part 2 proof of concept: Postgres + pgvector

A small, working version of the loading and storage side of [the design](../DESIGN.md):

- **real batch files** from the large data become **jobs** in an `ingest_jobs` table;
- a worker loads each file **in one transaction**: parse → clean → dedupe → upsert patents → chunk → embed → `COPY` passages;
- **one SQL query** then does filters + vector search together.

The code is in [`src/patent_search/pg/`](../../src/patent_search/pg/):

| File | What it does |
|---|---|
| [`schema.sql`](../../src/patent_search/pg/schema.sql) | The 3 tables, with an HNSW vector index, a full-text index and a trigram index for titles |
| [`rows.py`](../../src/patent_search/pg/rows.py) | Pure functions: patent → SQL row, and filters → `WHERE` clause (same meaning as Part 1's filters) |
| [`store.py`](../../src/patent_search/pg/store.py) | Postgres calls: the job queue (`FOR UPDATE SKIP LOCKED`), upserts, and the search query |
| [`ingest.py`](../../src/patent_search/pg/ingest.py) | The pipeline loop: claim a job, load the file, mark it done or failed |

It reuses the Part 1 code for parsing, cleaning, chunking and embedding, so the database holds exactly the patents the search engine sees.

## Run it

With Docker (nothing else needed):

```bash
docker compose up -d db
docker compose --profile poc run --rm poc                         # loads the bundled small data
LARGE_DATA=~/Downloads/coding_challenge_data \
  docker compose --profile poc run --rm poc python cli.py pg-load --dataset large --max-files 10
docker compose --profile poc run --rm poc python cli.py pg-search "battery pack with liquid cooling" --cpc H01M
docker compose --profile poc run --rm poc python cli.py pg-status
```

Or from your own Python, with only the database in Docker (faster on a Mac, because embedding can use the GPU):

```bash
docker compose up -d db
python cli.py pg-load --dataset large --max-files 10
python cli.py pg-search "battery pack with liquid cooling between cells" --cpc H01M --explain
python cli.py pg-status
```

## What a run looks like

Loading the first 10 batch files of the large data (10,000 records) on an M3 MacBook Pro:

```
queued 10 files from .../coding_challenge_data
done   job_000/patents_batch_0001.json: 1000 patents, 4047 passages, 0 duplicates, 0 bad (24.4s)
done   job_000/patents_batch_0003.json: 999 patents, 4202 passages, 1 duplicates, 0 bad (31.9s)
...
done   job_000/patents_batch_0010.json: 998 patents, 3953 passages, 2 duplicates, 0 bad (36.7s)
processed 10 jobs in 340.5s
{ "files": {"pending": 0, "running": 0, "done": 10, "failed": 0},
  "records_in": 10000, "patents_written": 9991, "passages": 40355, "duplicates": 9, "bad_records": 0, ... }
```

- **9 duplicates** were caught across the 10 files, and the copy with the most text was kept.
- **3,828 of 9,991 patents (38%)** have no classification and **965** have no title (both written as `"N/A"` in the data). They are loaded anyway, as missing values.
- **Embedding is the slow part:** about 120 passages/s, on a laptop GPU that was also running the Docker stack. The time per file grew from 24 s to 42 s, partly because the HNSW index is updated on every insert. A real backfill would build the vector index once, after loading.
- **Size on disk:** 40k passages take 168 MB, of which 78 MB is the HNSW index (pgvector 0.8.7).

## Filters + vector search in one query

`pg-search` runs this query (filters are added to the `WHERE`):

```sql
WITH hits AS (
    SELECT s.doc_number, s.label, s.embedding <=> :query_vector AS distance
    FROM passages s JOIN patents p ON p.doc_number = s.doc_number
    WHERE p.cpc_subclass LIKE 'H01M%'                -- the filters
    ORDER BY s.embedding <=> :query_vector           -- nearest passages first
    LIMIT 200
)
SELECT doc_number, min(distance) AS distance, (array_agg(label ORDER BY distance))[1] AS best_passage
FROM hits GROUP BY doc_number ORDER BY distance LIMIT 10;   -- best passage per patent (MaxP, as in Part 1)
```

Median of 15 runs, top 10 patents, query "battery pack with liquid cooling between cells". "Index" is the HNSW vector index with `hnsw.iterative_scan` on; "exact" turns index scans off.

| Filter | Patents that match | With the vector index | Exact scan | Same top 10? |
|---|---:|---:|---:|---|
| none | 9,991 | **4.1 ms** (uses HNSW) | 34.0 ms | 10/10 |
| year ≥ 2010 (broad) | 7,138 | **5.7 ms** (HNSW + filter) | 53.2 ms | 10/10 |
| keyword "battery" | 184 | 4.5 ms (full-text index first) | 4.2 ms | 10/10 |
| CPC `G06F` + year ≥ 2020 | 207 | 10.2 ms (filter first) | 8.3 ms | 10/10 |
| CPC `H01M` | 95 | 8.3 ms (filter first) | 6.4 ms | 10/10 |
| CPC `H01M 10/6` (narrow) | 6 | 1.2 ms (code index first) | 0.7 ms | 6/6 |

**Postgres makes the same choice as the Part 1 planner.**
- **Broad or no filter:** it walks the vector index and skips rows that fail the filter (about 8–9× faster than exact).
- **Narrow filter:** it finds the few matching patents first, using the classification or full-text index, and computes their distances exactly.

Either way the filter never throws results away. The narrow `H01M 10/6` filter returns all 6 matching patents, not the fraction a post-filter would keep.

## What it shows, and what it doesn't

Shows:
- one job per file, with status and counts you can query;
- retries: a failed file is retried up to 3 times, then left `failed` with its error;
- a lease, so a job whose worker died is taken over after 10 minutes;
- safe parallel workers (`SKIP LOCKED`);
- duplicates resolved by document number, and a file loaded all-or-nothing;
- filters + vector search in the database.

Doesn't (yet):
- **heartbeat the lease.** The lease is set once, when a job is claimed. A file that takes longer than 10 minutes is claimed again by a second worker while the first is still on it. No data is lost (the second copy loses the "more text wins" upsert), but the job's counts end up as the second worker's zeros. A real worker would refresh `updated_at` from a separate connection while it works.
- **back off between retries.** A failed file is retried right away, in the same loop. DESIGN.md §8 describes retrying later with a growing delay.

The tests cover each of these:
- [`tests/unit/test_pg_ingest.py`](../../tests/unit/test_pg_ingest.py) runs the pipeline against an in-memory store, with a broken file and a flaky embedder;
- [`tests/integration/test_pg_store.py`](../../tests/integration/test_pg_store.py) runs against real Postgres, including two workers claiming jobs at once.

Run the real-Postgres tests with:
```bash
PATENT_PG_DSN=postgresql://patents:patents@localhost:5432/patents pytest -m postgres
```

Doesn't show (on purpose, to keep it small):
- **Keyword search.** There's no BM25 here; the full-text index is only used as a filter.
- **Separate stages.** Parsing and embedding are not split into their own stages and workers.
- **Descriptions.** It embeds title + abstract and independent claims only (~4 passages per patent, not 140).
- **Scale.** 10k patents is far below where pgvector gets hard; the design moves search to OpenSearch for that reason.
