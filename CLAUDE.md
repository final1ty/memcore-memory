# memcore-memory — project guide

Production-grade local-first memory system for AI agents. PyPI: `memcore-memory` (imports as `memcore_memory`, legacy shim package `mnemosyne`). Docker Hub: `memcorehq`. Author: Kovács-Dobos Ádám <kovacsdobosadam@gmail.com>. GitHub: https://github.com/final1ty/memcore-memory (Apache-2.0).

## Architecture (verified against `src/memcore_memory/`)

- **Tiers** (`core/tiers.py:9-12`, enum `Tier`): `sensory` (30s) → `working` (7±2 items, 20min) → `episodic` (weeks) → `semantic` (years). Promotion to `semantic` happens when `importance > 0.8` and rehearsals pass `semantic_consolidation_threshold`; demotion/eviction driven by TTL + retention.
- **Ebbinghaus forgetting** (`core/ebbinghaus.py`): `R = exp(-t / S)`, `S = S0 * (1 + log(1+rehearsals)) * (1+importance)`, each rehearsal does `S = S*1.6 + 0.5`.
- **6-way hybrid retrieval** (`retrieval/hybrid.py` + `retrieval/retrievers/*`), fused via RRF (k=60) + learned weights: vector 0.35, bm25 0.25, graph 0.15, temporal 0.10, importance 0.10, metadata 0.05 → reported **MRR@10 = 0.85** (per README/pyproject; this is the project's own benchmark claim on LoCoMo + LongMemEval, not something re-verified in this session).
- **Embeddings**: BGE-small + matryoshka + reranker (`embeddings/`), falls back to a hash embedding if `sentence-transformers` isn't installed.
- **Storage**: encrypted SQLite (`storage/encrypted_sqlite.py`) + HNSW vector store, WAL for durability, optional Postgres/pgvector backend.
- **Security**: AES-256-GCM (random 96-bit nonce/record), master key wrapped with Argon2id KEK (memory_cost=64MB, iterations=3), blind-index for encrypted search, audit log, PII filter, rate limiting.
- **Interfaces**: MCP server (29 tools, stdio, `memcore server mcp`), REST API (FastAPI, `server start`), CLI (`memcore`/`mnem`/`mnemosyne`, 60 commands), Python SDK (sync/async), P2P gossip+CRDT sync (OR-Set + LWW-Register).
  - Counts corrected 2026-09-22. The manifest previously claimed 33 MCP tools (22 of them were unimplemented no-ops) and "200+" CLI commands (actual: 60, of which 28 are generated stubs that print a placeholder). Don't restore the old numbers.
- **REST routes** (`api/rest.py`): `POST /memory`, `GET /memory/{id}`, `POST /recall`, `GET /memories`, `DELETE /memory/{id}`, `POST /consolidate`, `POST /forget`, `GET /health`, `POST /sync/merge`, `GET /kg/traverse/{entity}`.

## Topology — how the pieces actually connect

```
Windows PC (C:\Users\A\)
  Claude Desktop  ──stdio (JSON-RPC over MCP)──▶  memcore_remote_mcp.py
                                                    (python.exe -u, MCP name "memcore-skynas", 4 tools)
                                                              │
                                                              │ HTTP, LAN
                                                              ▼
SkyNAS — HP EliteDesk 840G5, Ubuntu, 192.168.1.183, Docker enabled
  container `mnemosyne` (mnemosyne-memory:1.0.0)
    - REST API   :8000  ◀── used by the Windows bridge and by curl/CLAUDE.md examples below
    - P2P gossip :7742  (CRDT sync between peers, not an HTTP endpoint)
    restart: unless-stopped

  container `skymcp` :8765   — separate, unrelated MCP gateway (/opt/skymcp), listed for
                                context only, not part of memcore-memory
```

Claude Code sessions on the SkyNAS host reach the same container through the local stdio bridge configured in [.mcp.json](.mcp.json) (`server mcp --remote http://192.168.1.183:8000`), which forwards to `POST /mcp/call` — same 29 tools, same store, no password. Plain `curl` to `:8000` still works for scripts (examples below).

```
SkyNAS host
  Claude Code ──stdio──▶ server mcp --remote ──HTTP──▶ container :8000 ──▶ /data (one store)
```

## SkyNAS deployment (verified live, 2026-09-22)

- Host: SkyNAS, HP EliteDesk 840 G5, Ubuntu, LAN IP `192.168.1.183` (confirmed via `ip addr` on host).
- Container `mnemosyne` (image `mnemosyne-memory:1.0.0`) — **running**, ports `8000` (REST) and `7742` (P2P) both bound, `restart: unless-stopped`. Confirmed via `docker ps`.
- Volume: `memcore-memory-100_mnem_data` (confirmed via `docker volume ls`), mounted at `/data` in the container (confirmed via `docker inspect`). A stale `memcore-memory-100_mnem_data_peer2` volume also exists from an earlier two-peer compose setup that has since been removed from `docker-compose.yml` (uncommitted working-tree change).
- `GET http://192.168.1.183:8000/health` → `{"status":"ok","version":"1.0.0","tier_counts":{"working":1}}` — one memory currently in the `working` tier. `POST /recall {"query":"Docker"}` correctly returns that memory.
- **There are three separate, non-synced SQLite stores on this host** — see "Known issues" below for why, and don't assume they contain the same data:
  1. `/root/.memcore/` **inside the `mnemosyne` container's writable layer** — this is what the REST API on :8000 serves right now. `memory.db` = 28672 bytes, `master.key` = 32 bytes (unprotected, no password). ⚠️ **Its on-disk key no longer decrypts its own database** — see the master-key bug. The running process still holds the correct key in memory; nothing else can read this store. Use the recovered copy under `backups/20260922-recovered-store/` instead.
  2. `/var/lib/docker/volumes/memcore-memory-100_mnem_data/_data` — the named volume that's *supposed* to hold the data. Currently **empty**. Root-owned, so a host process can't read it even once it's populated; that's why MCP goes through the REST bridge.
  3. `/home/skynas/.memcore/` — a third copy created when the CLI (`.venv/bin/memcore ...`) is run directly on the host outside the container. `memory.db` = 45056 bytes, `master.key` = 187 bytes (password-protected, Argon2id-wrapped). Last written 2026-09-21 04:48, ~18h before the container store. Not touched by the REST API at all, and **not** what any MCP client should be pointed at.
- Windows bridge (`C:\Users\A\memcore_remote_mcp.py` → Claude Desktop MCP entry `memcore-skynas`, 4 tools) is **not verifiable from this host** — this session has no access to the Windows filesystem. Take its config on faith until confirmed from the Windows side.
- Config snapshot: [.claude/memcore.memory.json](.claude/memcore.memory.json).

## CLI cheat sheet (from `memcore --help`, actual flags — verify before trusting any older note)

```bash
memcore system init                                   # only if DB not already initialized
memcore system health                                 # static config, no password needed
memcore --password X system stats                     # global --password flag; also reads $MNEM_MASTER_PASSWORD
memcore system stats                                  # prompts interactively when stdin is a TTY
memcore memory add "text" --importance 0.9             # add to memory
memcore memory list --limit 10 [--tier working]        # list memories
memcore memory get <id>                                 # fetch by id
memcore memory recall "query" --k 2 [--tier working]     # NOTE: --k, not -k, and not positional
memcore server start --host 0.0.0.0 --port 8000          # REST API
memcore server mcp                                       # stdio MCP server for Claude Desktop
```

Any command that touches the encrypted DB (`system stats`, `memory list/get/recall/add`, …) needs the master password. Supply it with the global `--password` flag or `$MNEM_MASTER_PASSWORD`; with a TTY and neither set, it prompts. The REST API remains the easiest option for scripts, since the container unlocked its key at startup and never re-prompts:

```bash
curl -s http://192.168.1.183:8000/health
curl -s http://192.168.1.183:8000/memories
curl -s -X POST http://192.168.1.183:8000/recall -H "Content-Type: application/json" \
  -d '{"query":"SkyNas","k":2,"tier_filter":["working"]}'
```

## Known issues (critical)

**Bug: opening an unprotected store silently destroyed its master key.** (Found, fired for real, and fixed 2026-09-22. This is the worst bug the project has had.)

`KeyManager.load_or_create` branched on `key_path.exists() and key_path.stat().st_size > 100`. A password-wrapped key is JSON (>100 bytes) and matched. A **raw, unprotected key is exactly 32 bytes**, so it never matched — and fell through to the *create* branch, which generated a fresh key with `AES256GCM.generate_key()` and wrote it straight over the old one. Every open of an unprotected store replaced the key that store was encrypted under.

It hid perfectly. The process doing the overwrite kept the new key in memory and carried on encrypting and decrypting without complaint, so nothing looked wrong until the *next* process opened the store and got `cryptography.exceptions.InvalidTag` on every row — by which point the original key existed nowhere.

It fired on the SkyNAS container on 2026-09-22: a `docker exec mnemosyne python -c "... create_memory_system() ..."` run during an audit rotated `/root/.memcore/master.key` at 23:40:47 UTC. The REST process kept serving normally from its in-memory copy while its own database was already unreadable on disk. A restart would have lost the data permanently.

**Recovery worked completely**, because only `content` and `metadata` are encrypted — `id`, `tier`, `timestamp`, `forgetting_json`, `entities_json` and `embedding` are plaintext columns. The plaintext of the two encrypted columns was pulled out over the REST API while the old key was still resident, then re-encrypted under a new key with every plaintext column carried across verbatim. Recovered store: `/mnt/nas7/SkyNas/memcore-memory/backups/20260922-recovered-store/`; the pre-recovery copy and the rescued plaintext are in `backups/20260922-014334-container-writable-layer/`.

Fixed in [crypto/key_manager.py](src/memcore_memory/crypto/key_manager.py): `load_or_create` now creates a key **only when no key file exists**, and tells raw from wrapped by *content* (JSON with `salt`/`nonce`/`ct` → unwrap; exactly 32 bytes → use as-is; anything else → raise `MasterKeyUnreadable` rather than overwrite). New key files are also written `0600` instead of `0644`. Covered by [tests/test_key_persistence.py](tests/test_key_persistence.py) — the load-twice tests are the point; **never** reintroduce a size-based branch.

**Bug: `MNEM_*` env vars are silently ignored — data never lands in the named volume.**

`config.py` had `env_prefix = "MEMCORE_"`, but `Dockerfile`/`docker-compose.yml` (and every doc/example) set `MNEM_DATA_DIR`, `MNEM_ENCRYPTION_ENABLED`, etc. Since pydantic-settings only maps env vars matching the configured prefix, `MNEM_DATA_DIR=/data` was never read, and `Settings.data_dir` fell back to its default `Path.home() / ".memcore"`. Inside the container that resolves to `/root/.memcore` (container runs as root) — **not** `/data`, which is where the named volume `memcore-memory-100_mnem_data` is mounted. Verified by `docker exec mnemosyne env` (shows `MNEM_DATA_DIR=/data`) vs `docker exec mnemosyne ls $HOME/.memcore` (shows the real, live `memory.db`) vs the volume's actual host directory (`/var/lib/docker/volumes/.../​_data`, empty except `.`/`..`).

**Impact**: all data the container has ever stored lives only in that one container's writable layer. It is **not covered by the named volume**, so:
- The backup command in the README (below) currently backs up nothing until this is fixed and the container is recreated with the fix live.
- `docker compose down` / `docker rm mnemosyne` / any container recreation **destroys all memory data** — `restart: unless-stopped` only protects against restarts of the *same* container, not recreation.

**There was a second bug stacked underneath it.** Fixing the prefix alone was *not* enough. `db_path`, `key_path`, `vector_path`, `audit_log_path` and `working_buffer_path` were declared at class scope as `data_dir / "..."`, which pydantic evaluates **once, at class-creation time, against the default `data_dir`**. So even with `MNEM_DATA_DIR=/data` correctly parsed, only `data_dir` moved — every derived path still pointed into `~/.memcore`. Found on 2026-09-22 when a test run against an isolated data dir unexpectedly hit the real `/home/skynas/.memcore/master.key`. Both are now fixed: the paths default to `None` and a `model_validator(mode="after")` resolves them against the effective `data_dir`, with per-path overrides (`MNEM_DB_PATH`, …) still winning. Covered by [tests/test_config_paths.py](tests/test_config_paths.py) — **do not** re-inline those defaults.

**Fixed**: [src/memcore_memory/config.py](src/memcore_memory/config.py) now uses `env_prefix = "MNEM_"` (via `SettingsConfigDict`) and resolves derived paths at runtime. **Not yet deployed** — the running container was built before these fixes, and both remaining steps are "shared resource" writes that this session's auto-mode permissions correctly gate for a human to approve:

```bash
docker run --rm -v memcore-memory-100_mnem_data:/dst -v /mnt/nas7/SkyNas/memcore-memory/backups/20260922-recovered-store:/src:ro alpine cp -a /src/. /dst/
```

```bash
docker compose up -d --build
```

Stage first, then rebuild — the other order loses the data. Note the source is the **recovered** store, not `/root/.memcore`: the container's on-disk key no longer matches its own database (see the master-key bug above), so copying the writable layer across would move an undecryptable store into the volume. The recovered directory has been verified end-to-end against a container built from this tree — same id, content, tier, timestamp, metadata, embedding and forgetting curve, and `master.key` unchanged across a restart.

Beware that the prefix fix activates **every** `MNEM_*` var at once, including ones that were previously inert. `docker-compose.yml` had `MNEM_EMBEDDING_DIM=768`, which would have re-dimensioned the embedder away from the 384-dim vectors already stored; it is now 384, matching `config.py`. Check any new var against the defaults in `config.py` before adding it.

Until both steps are done, treat the container's data as ephemeral and **do not restart or recreate the container** — its in-memory key is the only thing still able to read its own database.

**Bug: rehearsals were persisted and then thrown away on every read.** (Found and fixed 2026-09-22.)

`MemoryItem.__post_init__` set `forgetting.strength` from the tier unconditionally — including in the construction inside `EncryptedStore._row_to_item`. `rehearse()` correctly did `S = S*1.6 + 0.5` and `store.put` wrote it to SQLite, but the very next `store.get` overwrote `strength` with the tier baseline again. Net effect: **the Ebbinghaus curve never strengthened with use.** A memory recalled a hundred times decayed on exactly the same schedule as one never touched again, which nullifies the central feature of the system. This is the likely explanation for the live SkyNAS memory dropping from R=0.869 to R=0.562 over a few hours despite being recalled repeatedly.

Fixed in [core/tiers.py](src/memcore_memory/core/tiers.py): `forgetting` now defaults to `None` and is seeded from `TIER_BASE_STRENGTH` only for genuinely new items; a curve passed in (i.e. loaded from storage) is authoritative. Because the old clobber also *accidentally* raised strength on promotion, [storage/encrypted_sqlite.py](src/memcore_memory/storage/encrypted_sqlite.py) `update_tier` now raises strength to the new tier's floor explicitly — and never lowers it, so a demoted memory keeps what rehearsal earned. Covered by [tests/test_forgetting_persistence.py](tests/test_forgetting_persistence.py). The Postgres backend passes `forgetting=` explicitly too, so it gets the same fix.

**Bug: BM25 silently lost the terms that mattered most.** (Found and fixed 2026-09-22.)

`BM25Okapi`'s IDF is `log((N-df+0.5)/(df+0.5))`, which hits **zero** once a term appears in half the corpus and goes negative beyond that. The retriever drops non-positive scores, so on a small personal store the most characteristic terms returned nothing: with 4 memories, `"SkyNAS"` (present in 2) scored 0 hits while `"Docker"` (present in 1) worked fine. The more central a term is to your own memory, the less findable it was.

Switched to `BM25L` in [retrieval/retrievers/bm25.py](src/memcore_memory/retrieval/retrievers/bm25.py), which keeps non-matching documents at zero while scoring every genuine match positive. `BM25Plus` was rejected: it scores *every* document positive, which would have destroyed precision in the RRF fusion. The index is now cached and rebuilt only when the corpus changes, instead of re-tokenising the whole store on every query. Covered by [tests/test_bm25_recall.py](tests/test_bm25_recall.py).

## Critical fixes / gotchas

1. **Recall syntax**: `memcore memory recall "query" --k 2` — `--k` is a named flag (default 10), not `-k` and not positional. Same for `--tier`.
2. **Missing embeddings**: `sentence-transformers`/`torch` are **not** installed anywhere on SkyNAS — not in the container, not in `.venv` (an older note here claimed they were; `pip list` says otherwise). So `BGEEmbedder` has always been falling back to `LocalHashEmbedder`, a deterministic hash, while `/health` and `config_get` reported `bge-small`. The old `Dockerfile` caused this: `pip install -e ".[all]" || pip install -e .` swallowed the failure and shipped an image that advertised a model it didn't have. The Dockerfile now takes one `EMBEDDING_PROVIDER` build arg, labels the image with what actually got installed, and fails the build if the extras don't import:
   ```bash
   docker compose build --build-arg EMBEDDING_PROVIDER=bge-small   # ~2GB of torch, model downloaded at first use
   ```
   Default is `local` (hash embeddings), which is what the deployment has been running all along — so switching the label changes the reported name, not the vectors. Real BGE embeddings would invalidate anything already stored under the hash embedder.
3. **Zip/git bloat**: never zip with `.venv/` or `.git` included (`.venv` pulled objects into `.git/objects` once and produced a 9.0GB archive). Fix:
   ```bash
   zip -r out.zip . -x "*/.venv/*" "*/__pycache__/*" "*.pyc" "*/dist/*" "*/.git/*" "*/.pytest_cache/*"
   ```
   For GitHub pushes: `git config user.email "kovacsdobosadam@gmail.com"`, `git config user.name "Kovács-Dobos Ádám"`, and keep `.venv/` in `.gitignore`.
4. **File transfer**: `transfer.sh` is blocked by the SkyNAS firewall — use `curl -F "file=@zip" https://0x0.st` instead.
5. **Do not re-init the DB** (`system init`) on the SkyNAS instance — it's already initialized and holds live data; re-init would create/overwrite the key material.
6. **Password handling** (rewritten 2026-09-22 — this used to say there was no non-interactive path; there is one now). `KeyManager.load_or_create` resolves the password in this order: explicit argument → `$MNEM_MASTER_PASSWORD` → interactive `getpass()`, **and only prompts when stdin is a TTY**. When it isn't, it raises `MasterPasswordRequired` with a diagnostic instead of consuming the protocol stream or aborting cryptically. The CLI takes a global `--password` flag (`memcore --password X memory list`), which also reads that env var.
   - The interactive prompt is still the default for humans; nothing is stored on disk. Supplying the password by env var does expose it to the process environment — the usual `PGPASSWORD` tradeoff — so prefer the prompt when you have a terminal.
   - `load_or_create` creates a key **only when the file doesn't exist**, and refuses to overwrite one it can't parse. Any change to that branch risks the key-destruction bug above; run [tests/test_key_persistence.py](tests/test_key_persistence.py) if you touch it.
7. **stdout is sacred under stdio MCP.** Library diagnostics in `embeddings/`, `retrieval/` and `storage/` used to `print()` to stdout, which corrupts the JSON-RPC stream the moment a model loads. They now go to stderr, and `server mcp` additionally wraps startup in `contextlib.redirect_stdout(sys.stderr)`. If you add a `print()` anywhere in the library import path, send it to stderr.

## MCP access for this project

- **From this SkyNAS host / Claude Code sessions here**: [.mcp.json](.mcp.json) runs the stdio MCP server in **bridge mode**, proxying to the container's REST API:
  ```
  server mcp --remote http://192.168.1.183:8000
  ```
  It talks the real MCP protocol (JSON-RPC 2.0 via the official `mcp` SDK, v2.2.0) and exposes all 29 tools. No master password is needed on the client side — the container unlocked its key at startup.

  **Do not point it at a local data dir instead.** That was the first attempt and it was wrong: `MNEM_DATA_DIR=/home/skynas/.memcore` opens a *different database* from the one the REST API serves (see the three-stores note above). Both open cleanly, both answer, and they disagree — so the client looks healthy while serving memories nobody wrote. There is no host path that fixes this: the live store is inside the container and its volume is root-owned. One store, reached one way. Covered by [tests/test_mcp_remote.py](tests/test_mcp_remote.py).

  The bridge needs `GET /mcp/tools` and `POST /mcp/call` ([api/rest.py](src/memcore_memory/api/rest.py)), which dispatch through the same `HANDLERS` table as the stdio server — the tool surface can't drift between the two. Against an instance built before those routes existed it exits with "predates the REST-backed MCP bridge", not an obscure failure.

  One semantic difference in bridge mode: `memory_export`/`memory_import` take a path that the *server* resolves, so they read and write inside the container.

  Before the 2026-09-22 rewrite, `mcp/server.py` implemented a homemade `{"tool": ..., "args": ...}` line protocol that no MCP client could speak, and fell through to `{"status": "executed"}` for any tool it didn't implement — so 22 of the 33 advertised tools reported success while doing nothing at all. `mcp/server.py` now asserts at import time that every advertised tool has a handler.

  Verify the bridge standalone at any time:
  ```bash
  .venv/bin/python -m memcore_memory.cli.main server mcp --remote http://192.168.1.183:8000
  ```
  (Local-store mode still exists for a machine that really does own its store: drop `--remote` and supply `--password` or `$MNEM_MASTER_PASSWORD` if the key is protected. Check `~/.cache/claude-cli-nodejs/*/mcp-logs-memcore/` if the server shows as failed.)

- **REST API** (still the simplest thing for scripts, and what the container serves):
  ```bash
  curl -s http://192.168.1.183:8000/health
  curl -s http://192.168.1.183:8000/memories
  curl -s -X POST http://192.168.1.183:8000/recall -H "Content-Type: application/json" -d '{"query":"...","k":5}'
  curl -s -X POST http://192.168.1.183:8000/memory -H "Content-Type: application/json" -d '{"content":"...","tier":"working","importance":0.5}'
  ```
- **From Windows (Claude Desktop)**: the bridge script and both config locations are on `C:\` and out of reach from this session. Confirmed by direct filesystem search on 2026-09-22 — no copy of `memcore_remote_mcp.py` exists anywhere on the SkyNAS host or its NAS mounts, so it genuinely only lives on the Windows machine. If it stops showing up ("announcing memcore-skynas: 4 tool(s)" missing from `mcp-server-memcore-skynas.log`), the config block to check/repair in **both**
  `C:\Users\A\AppData\Local\Claude\claude_desktop_config.json` and
  `C:\Users\A\AppData\Roaming\Claude\claude_desktop_config.json` is:
  ```json
  {
    "mcpServers": {
      "memcore-skynas": {
        "command": "C:\\Users\\A\\AppData\\Local\\Programs\\Python\\Python314\\python.exe",
        "args": ["-u", "C:\\Users\\A\\memcore_remote_mcp.py"]
      }
    }
  }
  ```
  Things to check by hand on the Windows side (can't be verified remotely): both paths above actually exist, the bridge script points at `http://192.168.1.183:8000`, and the log directory has write access.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q        # pytest + pytest-asyncio installed into .venv 2026-09-22
```

Current state: **53 passed, 4 failed, 2 skipped.** The 4 failures are all in `tests/test_excellent.py` and **pre-date this work** (verified by stashing every change and re-running). They assert features that were never implemented, not regressions:

- `test_ebbinghaus_power_law` — passes `ForgettingCurve(decay_model=...)`; no such parameter exists.
- `test_encrypted_store_blind_index` — calls `EncryptedStore.search_by_blind_index()`; no such method.
- `test_blind_index` — expects the blind index to return matches; it returns empty.
- `test_key_manager_enforce_password` — expects `MEMCORE_ENV=prod` to force a password on key *creation*; no such enforcement exists. Worth implementing, but it would refuse to create the unprotected keys the running container currently relies on, so it needs a deliberate decision rather than a drive-by fix.

`tests/conftest.py` has an autouse `isolate_data_dir` fixture. It exists because the suite previously ran against the real `~/.memcore` — `test_memory.py` called `create_memory_system()` with no isolation and was writing test memories into the live encrypted store. **Never remove it.**

## Known discrepancies found during 2026-09-22 verification pass

A prior task brief for this project contained some claims that did not match the live system when checked against `docker ps`, `curl /health`, `curl /recall`, and the actual source (`core/tiers.py`). Recorded here so future sessions don't re-propagate them:

- **Tier names**: brief said `working/short/long/archival`. Actual enum in `core/tiers.py` is `sensory/working/episodic/semantic`. Fixed in `.claude/memcore.memory.json`.
- **"SkyNas first memory" record**: the brief described a memory added with content `"SkyNas first memory"`, id `8d235d7f-ab86-45a6-899b-3a5cf9edb619`, retention R=0.88. The database currently holds exactly one memory, and it is a *different* record: id `166657ab-1e3f-4d4d-b668-d5d6d0656eff`, content `"SkyNAS Docker-only telepítés sikeres"`, tier `working`, importance 0.5. Either the DB was reset/replaced since that note was written, or the note was never actually run against this DB. Don't assume the old id/content still exists without checking `GET /memories` first.
- **DB size**: 28672 bytes was stale; current size is 45056 bytes (see above).

Everything else in the original brief (author, repo URL, PyPI/Docker Hub names, retrieval weights, MRR@10=0.85, encryption scheme, container/volume names, endpoint IP, critical fixes 1-4) checked out against the live host and source and is reflected above as fact.

## Second verification pass, 2026-09-22 (same day, follow-up task)

A follow-up brief proposed tiers `["working","midterm","longterm"]` for `.claude/memcore.memory.json`. Still doesn't match the code (`core/tiers.py`: `sensory/working/episodic/semantic`, also confirmed by `tier_counts` keys in the live `/health` response) — kept the verified names, didn't overwrite them with the new unverified list. This is the second different tier-name list handed to this project in one day (`working/short/long/archival`, then `working/midterm/longterm`); neither matches reality, so verify against `core/tiers.py` again before trusting a future one.

This pass also found and fixed the env-prefix persistence bug described above under "Known issues" — that's new information, not a correction of anything stated earlier today.
