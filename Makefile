
.PHONY: install dev test docker run cli mcp docs

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --cov=mnemosyne

docker:
	docker build -t mnemosyne-memory:1.0.0 .

run:
	docker-compose up -d
	@echo "REST: http://localhost:8000/docs"
	@echo "Health: http://localhost:8000/health"

cli:
	mnem --help
	mnem memory --help
	mnem kg --help
	mnem sync --help

mcp:
	mnem server mcp

docs:
	python -m mnemosyne.cli.main system stats

lint:
	rufflehog --fail || true
	bandit -r src/

encrypt-check:
	python -c "from mnemosyne.crypto.aes_gcm import AES256GCM; k=AES256GCM.generate_key(); c=AES256GCM(k); n,ct=c.encrypt(b'test'); assert c.decrypt(n,ct)==b'test'; print('AES-256-GCM OK')"
