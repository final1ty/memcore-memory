FROM python:3.11-slim

# MNEM_* is the prefix config.py actually reads (see CLAUDE.md "Known issues" - it
# used to be MEMCORE_, so every one of these was silently ignored).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MNEM_DATA_DIR=/data

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Real sentence-transformers embeddings are opt-in: they add ~2GB of torch to the
# image and download the model at first use. The previous version of this file tried
# to install them with `pip install -e ".[all]" || pip install -e .`, so when the
# extras failed the build carried on and produced an image that advertised
# MNEM_EMBEDDING_PROVIDER=bge-small while actually running the hash fallback. That is
# exactly what the deployed image had been doing. Either branch below is now explicit
# and fails the build if it cannot deliver what it claims.
#
# One knob, so the label and the install can never disagree: anything other than
# "local" pulls the extras in and the build fails if they don't import.
#   docker compose build --build-arg EMBEDDING_PROVIDER=bge-small
#
# "local" is the deterministic hash embedder the bge-small path was already silently
# falling back to - same vectors, honest name - so labelling it that way does not
# invalidate anything already stored.
ARG EMBEDDING_PROVIDER=local
# Same rule for the storage backend. MNEM_BACKEND=postgres on an image without
# asyncpg/pgvector used to crash at the first import of storage/postgres.py, so the
# k8s manifest, which asks for postgres, could never start. Build that image with
#   docker build --build-arg BACKEND=postgres -t mnemosyne-memory:1.0.0-pg .
# and keep the tag distinct from the SQLite one compose builds for SkyNAS.
ARG BACKEND=sqlite
RUN pip install --upgrade pip && pip install -e . && \
    if [ "$EMBEDDING_PROVIDER" != "local" ]; then \
        pip install -e ".[embeddings]" hnswlib && \
        python -c "import sentence_transformers, torch"; \
    fi && \
    case "$BACKEND" in \
        sqlite) ;; \
        postgres) pip install -e ".[postgres]" && \
                  python -c "import asyncpg, pgvector.sqlalchemy" ;; \
        *) echo "unknown BACKEND=$BACKEND (expected sqlite or postgres)" >&2; exit 1 ;; \
    esac
ENV MNEM_EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER} \
    MNEM_BACKEND=${BACKEND}

VOLUME ["/data"]
# REST only. 7742 used to be exposed for P2P gossip, but nothing has ever listened on
# it: P2P is not implemented.
EXPOSE 8000

# /livez answers without touching the store. /health decrypts every row to count
# tiers, which is a full-table AES-GCM pass every 30s for a probe that only needs to
# know the process is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/livez', timeout=3).raise_for_status()" || exit 1

CMD ["python", "-m", "memcore_memory.cli.main", "server", "start", "--host", "0.0.0.0", "--port", "8000"]
