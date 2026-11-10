"""Paths and tunable constants in one place.

Every path can be overridden with an environment variable, so tests and other
machines never need to edit code.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(var: str, default: Path) -> Path:
    value = os.environ.get(var)
    return Path(value).expanduser() if value else default


# Folder names match what's inside the challenge zips, so unzipping into data/ just works.
DATA_DIRS: dict[str, Path] = {
    "small": _path_from_env("PATENT_DATA_SMALL", PROJECT_ROOT / "data" / "patent_data_small"),
    "large": _path_from_env("PATENT_DATA_LARGE", PROJECT_ROOT / "data" / "coding_challenge_data"),
}
# The small data's zip is committed to Git (3 MB), so a fresh clone can run everything.
SMALL_DATA_ZIP = _path_from_env("PATENT_DATA_SMALL_ZIP", PROJECT_ROOT / "data" / "patent_data_small.zip")
ARTIFACTS_DIR = _path_from_env("PATENT_ARTIFACTS", PROJECT_ROOT / "artifacts")
# Part 2 PoC database (docker compose's `db` service). Local demo credentials only.
PG_DSN = os.environ.get("PATENT_PG_DSN", "postgresql://patents:patents@localhost:5432/patents")

# Models (all free, run locally, no API key)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this instruction in front of *queries* (not passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
# Phase-2 reranker. Chosen by measurement (benchmarks/eval_plain.md): the
# classic ms-marco MiniLM cross-encoders did not beat plain hybrid search on
# patent text; mxbai-rerank-base did (+8 points Recall@1, +12 Recall@10).
RERANK_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"
# Device for embedding ONE query at search time. CPU is steady (~15 ms); the
# Apple GPU is faster on average (~9 ms) but spikes to ~90-200 ms whenever it
# sees a new query length. Building the index uses the default (GPU if present).
QUERY_DEVICE = os.environ.get("PATENT_QUERY_DEVICE", "cpu")
# Measured: in float32 this reranker is as fast on the CPU as on the Apple GPU
# (~2 s for 30 pairs), and CPU is what Docker gets, so CPU is the default.
RERANK_DEVICE = os.environ.get("PATENT_RERANK_DEVICE", "cpu")

# Chunking: long paragraphs/claims are cut into overlapping word windows,
# because the embedding model only reads ~512 tokens.
MAX_PASSAGE_WORDS = 250
PASSAGE_OVERLAP_WORDS = 50

# Search
RRF_K = 60  # standard constant for Reciprocal Rank Fusion
RETRIEVE_N = 200  # passages taken from each retriever before fusion
POST_FILTER_OVERFETCH = 1000  # naive post-filter: fetch this many, then filter
PASSAGES_PER_PATENT = 3

# Two-phase search (Part 3), all three settled by measurement on the plain-English set:
# - reranking the top 30 gave the same Recall@1/MRR as the top 50 or 100, at 1/3 the cost
# - 1 passage per patent beat 2 (0.625 vs 0.604 Recall@1) at half the cost
# - 256 tokens per pair gave the same quality as 512 and is ~1.7x faster
# rerank="auto" only reranks queries up to this many words. Measured: reranking
# helps short plain-English queries but hurts long claim-style ones (eval_claims.md).
RERANK_AUTO_MAX_WORDS = 40
RERANK_CANDIDATES = 30
RERANK_PASSAGES_PER_PATENT = 1
RERANK_MAX_LENGTH = int(os.environ.get("PATENT_RERANK_MAX_LENGTH", "256"))  # query + passage tokens
