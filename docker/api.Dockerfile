# The search API (FastAPI) with both models baked in.
#
#   docker compose up api        -> http://localhost:8000/docs
#
# First start: unzips the bundled small dataset and builds the index into the
# `artifacts` volume. Embedding 33k passages on CPU takes ~15 min on an M3 Pro
# (Docker on a Mac has no GPU); progress is logged. Later starts take seconds.
# To reuse an index built on the host instead: ARTIFACTS=./artifacts docker compose up

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/models \
    PATENT_ARTIFACTS=/artifacts \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# 1. CPU-only torch: the default wheel bundles CUDA (~2 GB more), and a
#    container on a Mac has no GPU anyway.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 2. Everything else (torch is already satisfied by the CPU build).
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Download the embedding model and the reranker at build time, so the
#    container never needs the internet at run time. (Before copying the code,
#    so editing code doesn't re-download ~500 MB.) Must match config.py.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
ARG RERANK_MODEL=mixedbread-ai/mxbai-rerank-base-v1
RUN python -c "import sys; from huggingface_hub import snapshot_download; \
[snapshot_download(m, allow_patterns=['*.json', '*.txt', '*.safetensors', '*.model']) for m in sys.argv[1:]]" \
    "$EMBEDDING_MODEL" "$RERANK_MODEL"
ENV HF_HUB_OFFLINE=1

COPY cli.py pyproject.toml ./
COPY src ./src
COPY data/patent_data_small.zip ./data/
COPY benchmarks ./benchmarks
COPY tests ./tests

EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=90 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"
CMD ["python", "cli.py", "serve", "--host", "0.0.0.0", "--port", "8000", "--build-if-missing"]
