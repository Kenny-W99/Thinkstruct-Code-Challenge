-- Part 2 proof of concept: patents + passages (with embeddings) + ingest jobs
-- in one Postgres database. {dim} is filled in by store.py (384 for bge-small).
-- Safe to run again: everything is "IF NOT EXISTS".

CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: vector type + HNSW index
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- trigram index for "title contains"

-- One row per input batch file. Workers claim pending rows with
-- FOR UPDATE SKIP LOCKED, so many workers can run at once without two of them
-- taking the same file. A job is the unit of retry and of status reporting.
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id              bigserial PRIMARY KEY,
    source_file     text NOT NULL UNIQUE,
    status          text NOT NULL DEFAULT 'pending'
                    CHECK (status IN ('pending', 'running', 'done', 'failed')),
    attempts        int NOT NULL DEFAULT 0,
    records_in      int,
    bad_records     int,
    duplicates      int,   -- copies of a patent already seen (in this file or earlier)
    patents_written int,
    passages        int,
    error           text,
    started_at      timestamptz,
    finished_at     timestamptz,
    updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ingest_jobs_status ON ingest_jobs (status);

CREATE TABLE IF NOT EXISTS patents (
    doc_number      text PRIMARY KEY,
    title           text,
    abstract        text,
    cpc_raw         text,
    cpc_subclass    text,   -- "B60C"
    cpc_group       text,   -- "11"  (NULL when the source code had no "/")
    cpc_subgroup    text,   -- "03"
    cpc_digits      text,   -- every digit after the subclass
    pub_date        date,   -- exact date when known (small data)
    pub_year        int,    -- always set when known (large data has only the year)
    has_description boolean NOT NULL DEFAULT false,
    n_claims        int NOT NULL DEFAULT 0,
    source_file     text NOT NULL,
    text_size       int NOT NULL,  -- duplicates: the copy with the most text wins
    tsv             tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(title, '') || ' ' || coalesce(abstract, ''))
                    ) STORED,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS patents_subclass ON patents (cpc_subclass, cpc_group);
CREATE INDEX IF NOT EXISTS patents_year ON patents (pub_year);
CREATE INDEX IF NOT EXISTS patents_tsv ON patents USING gin (tsv);
CREATE INDEX IF NOT EXISTS patents_title_trgm ON patents USING gin (lower(title) gin_trgm_ops);

CREATE TABLE IF NOT EXISTS passages (
    id          bigserial PRIMARY KEY,
    doc_number  text NOT NULL REFERENCES patents ON DELETE CASCADE,
    kind        text NOT NULL,  -- abstract | claim | description
    label       text NOT NULL,  -- "Claim 7"
    text        text NOT NULL,
    embedding   vector({dim}) NOT NULL
);
CREATE INDEX IF NOT EXISTS passages_doc ON passages (doc_number);
-- Approximate nearest-neighbour index (cosine distance).
CREATE INDEX IF NOT EXISTS passages_hnsw ON passages USING hnsw (embedding vector_cosine_ops);
