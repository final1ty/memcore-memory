FROM python:3.11-slim

# MNEM_* is the prefix config.py actually reads (see CLAUDE.md "Known issues" - it
# used to be MEMCORE_, so every one of these was silently ignored).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MNEM_DATA_DIR=/data \
    MNEM_BACKEND=sqlite

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
RUN pip install --upgrade pip && pip install -e . && \
    if [ "$EMBEDDING_PROVIDER" != "local" ]; then \
        pip install -e ".[embeddings]" hnswlib && \
        python -c "import sentence_transformers, torch"; \
    fi
ENV MNEM_EMBEDDING_PROVIDER=${EMBEDDING_PROVIDER}

VOLUME ["/data"]
EXPOSE 8000 7742

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()" || exit 1

CMD ["python", "-m", "memcore_memory.cli.main", "server", "start", "--host", "0.0.0.0", "--port", "8000"]
