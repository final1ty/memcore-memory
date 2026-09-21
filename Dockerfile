
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MNEM_DATA_DIR=/data \
    MNEM_BACKEND=sqlite \
    MNEM_EMBEDDING_PROVIDER=bge-small

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sqlite3 \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/

# Install with all extras for production: postgres + pgvector + BGE/E5 + hnswlib
RUN pip install --upgrade pip && \
    pip install -e ".[all]" || pip install -e . && \
    pip install sentence-transformers==2.6.1 torch --extra-index-url https://download.pytorch.org/whl/cpu || true

VOLUME ["/data"]
EXPOSE 8000 7742

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=3).raise_for_status()" || exit 1

CMD ["python", "-m", "mnemosyne.cli.main", "server", "start", "--host", "0.0.0.0", "--port", "8000"]
