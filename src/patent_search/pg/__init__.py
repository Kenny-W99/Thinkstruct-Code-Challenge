"""Part 2 proof of concept: the same patents in Postgres + pgvector.

    rows.py    pure: patents -> SQL rows, filters -> WHERE clause
    store.py   Postgres: job queue, upserts, filtered vector search
    ingest.py  the pipeline: batch files -> jobs -> patents + passages
"""
