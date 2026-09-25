# Tools come from the project venv, not whatever is first on PATH: `mnem`, ruff and
# bandit only exist in .venv/bin. Override with `make PY=python BIN=` inside an
# activated environment.
PY ?= .venv/bin/python
BIN ?= .venv/bin/

.PHONY: install dev test coverage docker run cli mcp lint encrypt-check

install:
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -e ".[dev]"

# No --cov here: pytest-cov is a dev extra, and without it the flag is an error.
test:
	$(PY) -m pytest tests/ -q

# Needs `make dev`. memcore_memory is the real package; `mnemosyne` is only an alias shim.
coverage:
	$(PY) -m pytest tests/ -q --cov=memcore_memory --cov-report=term-missing

docker:
	docker build -t mnemosyne-memory:1.0.0 .

# Starts a container from this checkout. For the live SkyNAS deployment use
# scripts/deploy-skynas.sh instead: it rescues the live store and checks record
# counts first, and this does neither.
run:
	docker compose up -d --build
	@echo "REST: http://localhost:8000/docs"
	@echo "Health: http://localhost:8000/health"

cli:
	$(BIN)memcore --help
	$(BIN)memcore memory --help
	$(BIN)memcore kg --help
	$(BIN)memcore sync --help

mcp:
	$(BIN)memcore server mcp

# Needs `make dev`. Existing ruff findings are not fixed by this target.
lint:
	$(BIN)ruff check src/ tests/
	$(BIN)bandit -q -r src/

encrypt-check:
	$(PY) -c "from memcore_memory.crypto.aes_gcm import AES256GCM; k=AES256GCM.generate_key(); c=AES256GCM(k); n,ct=c.encrypt(b'test'); assert c.decrypt(n,ct)==b'test'; print('AES-256-GCM OK')"
