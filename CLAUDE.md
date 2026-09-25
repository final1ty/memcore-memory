# memcore-memory — project guide

Production-grade local-first memory system for AI agents. PyPI: `memcore-memory` (imports as `memcore_memory`, legacy shim package `mnemosyne`, including `python -m mnemosyne.cli.main`). Docker Hub: `memcorehq`. Author: Kovács-Dobos Ádám <kovacsdobosadam@gmail.com>. GitHub: https://github.com/final1ty/memcore-memory (Apache-2.0).

> **2026-09-24/25 audit-fix pass.** A full audit (133 confirmed findings, two fix rounds) changed a lot of behaviour described below. Statements that changed are marked **[2026-09-24/25]**. The finding records (ids `F*`/`R*`) are in `memcore-full-audit-2026-09-24_wf_951355b1-d57/workflow-teljes-eredmeny.json` (gitignored). **Deployed to the live SkyNAS container on 2026-09-25 04:56 (+0200)** — see the verification under "SkyNAS deployment". See [Deploying the 2026-09-24/25 changes](#deploying-the-2026-0924-25-changes) before you deploy.

## Architecture (verified against `src/memcore_memory/`)

- **Tiers** (`core/tiers.py`, enum `Tier`): `sensory` (30s) → `working` (7-item cap, 20min) → `episodic` (weeks) → `semantic` (years). **[2026-09-24/25]** The lifecycle as implemented in `TierManager`'s docstring:
  - `sensory`: deleted after `sensory_ttl_seconds` unless marked `attended` and still retained (R > 0.3), which moves it to working.
  - `working`: where new memories land when no tier is given; importance never chooses the starting tier. Capped at `working_capacity` (7) against the store, oldest first. A working memory is **demoted, never deleted**, when rehearsed, older than `working_ttl_seconds` (20 min), or pushed out by overflow; demotion raises strength to the episodic floor (7 days). Every add applies TTL and cap; nothing schedules the lifecycle otherwise, so on an idle store an expired working memory keeps its short curve and shows low retention until the next add or `forget`.
  - `episodic`: deleted at R < 0.05. Promoted to `semantic` only with `rehearsals >= semantic_consolidation_threshold` (3), R > 0.6 **and** `importance > semantic_min_importance` (0.8).
  - `semantic`: never deleted automatically.
- **Ebbinghaus forgetting** (`core/ebbinghaus.py`): `R = exp(-t / S)` (optional `decay_model="power_law"`), `S = S0 * (1 + log(1+rehearsals)) * (1+importance)`, each rehearsal does `S = S*(1+0.6f) + 0.5f` with feedback `f` (1.0 → `S*1.6 + 0.5`). `rehearse(count=False)` strengthens without adding to the promotion count. Importance must be a number in [0, 1]; booleans are rejected.
- **6-way hybrid retrieval** (`retrieval/hybrid.py` + `retrieval/retrievers/*`), fused via weighted RRF (k=60): vector 0.35, bm25 0.25, graph 0.15, temporal 0.10, importance 0.10, metadata 0.05. Caveats that matter in practice: `temporal` and `importance` are query-independent priors, ranked **[2026-09-24/25]** only among the memories the query-dependent arms (vector, bm25, graph, metadata) matched, never store-wide; on a deployment without `sentence-transformers` the vector weight is 0 and redistributed. **[2026-09-24/25]** Zero-weight arms are skipped, a query no query-dependent arm matches returns `[]`, tier filters apply before truncation, a failing arm is logged to stderr (all query-dependent arms failing raises, and `POST /recall` answers 503), and every result has a boolean `matched`. Metadata ignores provenance keys (`importance`, `imported_from`, `origin_id`, `id`, `uuid`, `hash`, `checksum`, any `*_id`), and on a store of 10+ memories a token found in the metadata of more than half of them nominates nothing (on the live store: `claude`, `session`). **MRR@10 = 0.85** is an unverified project claim (LoCoMo + LongMemEval with real BGE embeddings): never reproduced here, not reproducible on the hash fallback. Don't cite it as a result.
- **Embeddings**: BGE-small via sentence-transformers (`embeddings/bge.py`), falling back to a hash embedding (`embeddings/local.py`) if `sentence-transformers` isn't installed. **[2026-09-24/25]** `embeddings/matryoshka.py` and `retrieval/reranker.py` are experimental and not wired in: no recall reranks, `matryoshka_enabled`/`reranker_enabled` default to false and are inert. `BGEEmbedder` always uses the model's own output dimension and refuses to start if `MNEM_EMBEDDING_DIM` disagrees (bge-base/e5-base need 768, bge-large/e5-large 1024); the hash fallback still uses `MNEM_EMBEDDING_DIM`, so the 384-dim SkyNAS store is unaffected.
- **Storage**: encrypted SQLite (`storage/encrypted_sqlite.py`) in SQLite WAL journal mode (persisted in the file, 10 s busy timeout) + a vector store whose sidecar is a cache, optional Postgres/pgvector backend. `storage/wal.py` is used by nothing. **[2026-09-24/25]** New tables/columns: `store_meta` (holds `key_id`, the master-key fingerprint), `aad_v` (row's ciphertexts are AAD-bound), `entities_enc`/`entities_nonce` (entities are encrypted; `entities_json` stays NULL for rows written by current code). New `memory.db` files are created 0600 (existing files keep their mode). List/`--tier` queries return newest first everywhere.
- **Knowledge graph** **[2026-09-24/25]**: `memory.kg.db`, always a per-instance local SQLite file next to `db_path`, even with `MNEM_BACKEND=postgres`. Schema v3: node ids are HMACs of the NFC-lowercased name, labels/props AES-GCM encrypted and AAD-bound to their row; the graph's shape is visible. WAL mode, new files 0600. Entities come from the caller; text extraction (capitalised-word heuristic, not NER) runs only when `MNEM_AUTO_EXTRACT_ENTITIES=true` and no entities were passed. No NetworkX.
- **Security**: AES-256-GCM (random 96-bit nonce per field), `content`/`metadata`/`entities` bound to row id and column via AAD (pre-AAD rows until next write or `memcore system migrate-aad`); `id`, `tier`, `timestamp`, `forgetting_json`, `embedding` and `blind_index_json` are plaintext and unauthenticated. Master key raw (32 bytes) or wrapped with an Argon2id KEK (memory_cost=64MB, iterations=3, lanes=4). Blind index for encrypted keyword search (SDK and `memcore memory search-blind` — not exposed over REST or MCP). Per-IP rate limiting on the REST API, **[2026-09-24/25]** including `/consolidate`, `/forget` and search/write tools over `/mcp/call`. Optional REST auth via `MNEM_API_KEY`. **Not wired into any request path:** the PII filter (`security/pii_filter.py`), the audit log (`storage/audit_log.py`), the reranker and the mTLS settings — `config_get` lists them under `not_implemented`.
- **Config** (`config.py`, prefix `MNEM_`): `MNEM_DATA_DIR` accepts `~`. Unknown `MNEM_*` variables warn on stderr. The only `MNEM_*` variables read outside `Settings` are `MNEM_MASTER_PASSWORD`, `MNEM_ENV` and `MNEM_REMOTE_URL`. An invalid value (e.g. `MNEM_BACKEND=mysql`, `MNEM_FORGETTING_MODEL=typo`, bad JSON in `MNEM_P2P_PEERS`) makes every entry point — `--help` and `server mcp --remote` included — fail at import with a one-line `ConfigError` naming the variable. Fail-closed on purpose. New: `MNEM_API_KEY`, `MNEM_AUTO_EXTRACT_ENTITIES` (default false), `MNEM_SEMANTIC_MIN_IMPORTANCE` (0.8), `MNEM_FORGETTING_MODEL`.
- **Interfaces**: MCP server (29 tools, stdio, `memcore server mcp`, local or `--remote` bridge mode), REST API (FastAPI, `server start`), CLI (`memcore`/`mnem`/`mnemosyne`/`memcore-memory`, **67 commands**), Python SDK (sync/async). **P2P sync is not implemented**: the CRDT primitives (LWW-Register, LWW-element set) are real and tested, but nothing transports or applies them; `P2PNode.start`, `broadcast_memory` and `GossipProtocol.gossip_loop` raise `NotImplementedError`, `/sync/merge` returns 501, and nothing has ever listened on 7742.
  - Counts corrected 2026-09-22 and again 2026-09-25. The manifest once claimed 33 MCP tools (22 of them were unimplemented no-ops) and "200+" CLI commands; later docs said 60. Actual: 29 tools and **67 CLI commands** (the help text computes it: system 9, memory 52, server 2, kg 2, sync 2), of which 28 are generated tier stubs that print a placeholder. Don't restore the old numbers.
- **REST routes** (`api/rest.py`): `POST /memory`, `GET /memory/{id}`, `DELETE /memory/{id}`, `POST /recall`, `GET /memories`, `POST /consolidate`, `POST /forget`, `GET /health`, `GET /livez`, `POST /sync/merge` (501), `GET /kg/traverse/{entity}`, `GET /mcp/tools`, `POST /mcp/call`. Behaviour details under [REST API behaviour](#rest-api-behaviour-2026-0924-25).

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
    - :7742  published by the RUNNING container only; docker-compose.yml no longer publishes
             it, so it disappears at the next redeploy. Nothing listens - P2P is not implemented
    restart: unless-stopped

  container `skymcp` :8765   — separate, unrelated MCP gateway (/opt/skymcp), listed for
                                context only, not part of memcore-memory
```

Claude Code sessions on the SkyNAS host use [.mcp.json](.mcp.json), which selects **bridge mode** (`server mcp --remote http://192.168.1.183:8000`, forwarding to `POST /mcp/call`) since 2026-09-25: the container's store is the authoritative one, and no password is needed. Local mode (from commit `32906cd` until then) never connected from the desktop app, because `$MNEM_MASTER_PASSWORD` is exported only by `~/.profile`/`~/.bashrc`, which the app does not source; see [MCP access](#mcp-access-for-this-project). `server mcp` prints which one it is serving to stderr at startup. Plain `curl` to `:8000` still works for scripts (examples below).

```
SkyNAS host
  Claude Code ──stdio──▶ server mcp (local)    ──▶ /home/skynas/.memcore   (retired 2026-09-25)
  Claude Code ──stdio──▶ server mcp --remote ──HTTP──▶ container :8000 ──▶ /data   (.mcp.json today)
```

## SkyNAS deployment (verified live 2026-09-25 after deploying the 2026-09-24/25 changes)

- **Deployed 2026-09-25 04:56 (+0200)** with `scripts/deploy-skynas.sh` (commit `18c22fb`). Verified right after: 120/120 memories, `master.key` unchanged across startup, `/livez` ok, `/health` `unreadable_count: 0`, only port 8000 published (7742 gone), `/data/vectors.vectors.json` 0600 with meta keys `{'tier'}` only (the plaintext excerpts are gone from the volume), 120/120 rows with encrypted entities and none in `entities_json`, `POST /recall "WireGuard"` finds the WireGuard memory. Rollback point: `backups/20260925-045648-volume-before-deploy.tgz` (the pre-deploy volume - it still contains the old plaintext sidecar, so treat it as sensitive) and `backups/20260925-045648-recovered-store`. The first deploy attempt that morning aborted safely in the rescue verifier (forgetting-curve importance is now synced from metadata on load); fixed in `18c22fb`.
- The rescue rebuilds the sidecar with current code, so the `reindex-vectors` step below was not needed for this deploy; it still is when deploying a `SOURCE` that carries an old-format sidecar.
- Everything below this point up to the store list is the 2026-09-22 state, kept for history.

- Host: SkyNAS, HP EliteDesk 840 G5, Ubuntu, LAN IP `192.168.1.183` (confirmed via `ip addr` on host).
- Container `mnemosyne` (image `mnemosyne-memory:1.0.0`) — **running**, ports `8000` (REST) and `7742` bound as of 2026-09-22, `restart: unless-stopped`. Confirmed via `docker ps` then. It runs the code of the 2026-09-22 deploy (first deploy 16:39 +0200; `backups/` shows further rescue/deploy runs up to 17:24 that day, and which one was the last successful redeploy was not re-verified). **None of the 2026-09-24/25 fixes are live in it.**
- Volume: `memcore-memory-100_mnem_data` (confirmed via `docker volume ls`), mounted at `/data` in the container (confirmed via `docker inspect`). A stale `memcore-memory-100_mnem_data_peer2` volume also exists from an earlier two-peer compose setup that has since been removed from `docker-compose.yml`.
- `GET http://192.168.1.183:8000/health` → `{"status":"ok","version":"1.0.0","tier_counts":{"working":7,"episodic":100,"semantic":13}}` — 120 memories as of 2026-09-22 16:40. **[2026-09-24/25]** After the redeploy the response gains `reasons`, `unreadable_count` and `unreadable`, and `status` can be `degraded` (HTTP 200) or `unhealthy` (HTTP 503). `GET /memories` defaults to `limit=50` and cuts content at 500 characters; use `GET /memory/{id}` whenever full content matters.
- **There are two separate, non-synced SQLite stores on this host.** Don't assume they hold the same data — as of 2026-09-22 16:40 they do not:
  1. **The volume** `memcore-memory-100_mnem_data`, mounted at `/data` — what the REST API on :8000 serves, 120 memories, `master.key` 32 bytes (unprotected, no password). Since the 2026-09-22 deploy this is the real store: the container no longer has a `/root/.memcore` at all, so recreation and `tar`-based backups cover the data. The host directory is root-owned, so a host process still can't open it directly — that's why MCP has a REST bridge.
  2. `/home/skynas/.memcore/` — a separate store used when the CLI (`.venv/bin/memcore ...`) runs directly on the host. 124 memories, `master.key` 187 bytes (password-protected, Argon2id-wrapped). [scripts/sync-rest-to-local.py](scripts/sync-rest-to-local.py) copies REST records into it, matching on exact full content, but the counts still differed. **Reconciled 2026-09-25**: of 125 host memories 117 matched the container by content; the 5 real host-only ones (Nostro lead automation, Nexi templates, PDF AcroForm pitfalls, commit convention) were added to the container with `metadata.origin_id`, and 3 smoke-test lines ("SkyNas first memory", "SkyNas production memory with BGE embeddings", "B terv REST szerver megy SkyNas-on") were not carried over. The container is authoritative since; this store is retired and no longer used by `.mcp.json`.

  A third store used to exist — `/root/.memcore` in the container's writable layer — and was where everything actually lived while `MNEM_DATA_DIR` was being ignored. The deploy moved its contents into the volume and it is gone.
- Windows bridge (`C:\Users\A\memcore_remote_mcp.py` → Claude Desktop MCP entry `memcore-skynas`, 4 tools; source since 2026-09-25 in `scripts/windows/`) is **not verifiable from this host** — this session has no access to the Windows filesystem. Take its config on faith until confirmed from the Windows side. If it checks `/health`, it must not treat a non-`ok` status as "down" after the redeploy: `degraded` is HTTP 200 and still serves.
- Config snapshot: [.claude/memcore.memory.json](.claude/memcore.memory.json).

### Deploying the 2026-09-24/25 changes

`bash scripts/deploy-skynas.sh [--keep-plaintext] [--drop-kg-hand-added] [SOURCE]`. What it does now (F16 and follow-ups):

1. Builds the new image while the old container still serves.
2. Rescues the live contents through the API on **every** run, also when `SOURCE` is given ([scripts/rescue-container-store.py](scripts/rescue-container-store.py)). Before stopping anything it checks that `SOURCE`'s `master.key` is a raw 32-byte key that decrypts `SOURCE/memory.db` (`store_meta` key_id match plus every row), and checks the rescue copy the same way.
3. Refuses unless **every live id** is in the store to stage (ids, not counts).
4. Stops the container, tars the volume to `BACKUP_ROOT/<stamp>-volume-before-deploy.tgz` (default `BACKUP_ROOT=/mnt/nas7/SkyNas/memcore-memory/backups`, override `MEMCORE_BACKUP_ROOT`), and compares the stopped volume with the rescue snapshot row by row ([scripts/store_fingerprint.py](scripts/store_fingerprint.py)). Any write between snapshot and stop — new memory, delete, edit, rehearsal, tier move, graph change — aborts before staging and restarts the old container; re-run the deploy.
5. Stages only `master.key`, `memory.db`, `memory.kg.db`, `vectors.vectors.json`, verifies them, and recreates with `--force-recreate`.

On abort it deletes the rescue's `RESCUE.plaintext.json` (even if the rescue failed after writing it) and prints the path; `--keep-plaintext` keeps it. The rescue refuses a non-empty `OUTPUT_DIR`, rebuilds the KG and the vector sidecar, writes `OUTPUT_DIR/source-fingerprint.json` (not staged), no longer uses `assert` (safe under `python -O`), and reads entities via `GET /memory/{id}` because they are no longer a plaintext column. It carries hand-added graph entities and relations across by reading the graph with the container's `master.key` (copied into its private 0700 temp dir); if that key cannot read the graph it refuses unless `--drop-kg-hand-added` (alias `--drop-kg-relations`) is given.

**Operational consequences — read before deploying:**

- **The first open by the new code migrates the store one-way.** `EncryptedStore.init()` encrypts every existing plaintext `entities_json` in place (after verifying the key), then runs `secure_delete`, `VACUUM` and `wal_checkpoint(TRUNCATE)` so the names also leave free pages and the `-wal`. Rows whose `entities_json` doesn't parse are logged to stderr and left untouched. `KnowledgeGraph.init()` upgrades `memory.kg.db` from v1/v2 to v3 in place, also one-way. On the first start after deploy, `create_memory_system` also backfills memory-entity links for all 120 memories once.
- **Rollback = restore the pre-deploy volume tarball** (`<BACKUP_ROOT>/<stamp>-volume-before-deploy.tgz`). **Never start the old image on the migrated volume**: it reads every migrated row's entities as `[]`, and after any write by the new code (or `migrate-aad`) it fails on list/health with `InvalidTag`. For the graph alone, an older image needs `memory.kg.db` moved aside (rebuilt from memories at next start; hand-added entities and relations lost).
- **After the deploy, run `memcore system reindex-vectors` in the container** — `docker exec mnemosyne python -m memcore_memory.cli.main system reindex-vectors` — so the live sidecar is rewritten in format 2. The old sidecar format held the first 500 **plaintext** characters of every memory next to the encrypted DB (F1). The first vector write of each process also deletes a pre-upgrade `vectors.vectors.tmp` that is 60+ s old (it can hold plaintext excerpts at 0644); if that fails it asks on stderr for manual removal. The pre-deploy tarball and older backups still contain the plaintext-bearing sidecar — treat them as plaintext.
- `memcore server start` warns on stderr that the REST API is **UNAUTHENTICATED** whenever `MNEM_API_KEY` is unset and the host isn't loopback. The container binds 0.0.0.0, so it will print this until a key is set; setting one requires the Windows bridge and any `--remote` bridge to send it first.
- Existing files keep their modes. To tighten by hand: `chmod 600 memory.db memory.db-wal memory.db-shm memory.kg.db vectors.vectors.json`. `memory.kg.db` now runs in WAL mode, so backups must include `memory.kg.db-wal` if it exists (the README's volume tar does).
- AAD rebinding of pre-AAD rows is **not** automatic: rows are upgraded when next written, or all at once by `memcore system migrate-aad` (one-way, back up first; needs `--yes` without a TTY). Until then legacy rows' ciphertexts can be swapped between rows undetected.
- `GET /health` still decrypts every row (F101 `count_by_tier()` not implemented), which is why the Dockerfile `HEALTHCHECK` and k8s probes now use `GET /livez`.

## CLI cheat sheet (from `memcore --help`, actual flags — verify before trusting any older note)

```bash
memcore system init                                   # safe on an existing store: loads the key, never replaces it
memcore system health                                 # passive file check as JSON (key format, row count, sidecar incl. embedder, kg); no password; exits 1 when degraded
memcore --password X system stats                     # global --password flag; also reads $MNEM_MASTER_PASSWORD
memcore system stats                                  # prompts interactively when stdin is a TTY
memcore system verify                                 # decrypts every row: "N of M memories readable", UNREADABLE <id> lines; read-only; exits 1 on any
memcore system protect [--password X]                 # wraps an unprotected key in place (key bytes unchanged)
memcore system migrate-aad [--yes]                    # binds pre-AAD rows to their id; one-way, back up first
memcore system reindex-vectors [--reembed [--force]]  # rebuild sidecar: one vector per memory; re-embeds rows of wrong width
memcore system reindex-blind                          # recompute the blind index for every row
memcore system reindex-kg                             # rebuild memory-entity links; exits 1 if some rows are unreadable
memcore memory add "text" --importance 0.9 [--tier T] [--entities "A,B"]   # importance in [0,1]
memcore memory list --limit 10 [--tier working]       # newest first
memcore memory get <id>                               # fetch by id
memcore memory recall "query" --k 2 [--tier working]  # NOTE: --k, not -k, and not positional
memcore memory search-blind "query" [--k N]           # exact-keyword search over the blind index; no rehearsal
memcore memory forget                                 # full lifecycle pass: forgotten/demoted/promoted; exits 1 if any row failed
memcore memory consolidate                            # "Promoted N episodic->semantic"
memcore kg add-entity NAME                            # leaves an existing entity unchanged ("already exists")
memcore server start --host 0.0.0.0 --port 8000       # REST API
memcore server mcp [--remote URL] [--api-key KEY]     # stdio MCP server; prints its mode to stderr
```

Any command that touches the encrypted DB (`system stats`, `memory list/get/recall/add`, …) needs the master password when the key is wrapped. Supply it with the global `--password` flag or `$MNEM_MASTER_PASSWORD`; with a TTY and neither set, it prompts. The REST API remains the easiest option for scripts, since the container unlocked its key at startup and never re-prompts:

```bash
curl -s http://192.168.1.183:8000/health
curl -s http://192.168.1.183:8000/memories
curl -s -X POST http://192.168.1.183:8000/recall -H "Content-Type: application/json" \
  -d '{"query":"SkyNas","k":2,"tier_filter":["working"],"rehearse":false}'
```

(`"rehearse": false` is accepted only after the redeploy; the running container rehearses every recall.)

## Known issues (critical)

**Bug: opening an unprotected store silently destroyed its master key.** (Found, fired for real, and fixed 2026-09-22. This is the worst bug the project has had.)

`KeyManager.load_or_create` branched on `key_path.exists() and key_path.stat().st_size > 100`. A password-wrapped key is JSON (>100 bytes) and matched. A **raw, unprotected key is exactly 32 bytes**, so it never matched — and fell through to the *create* branch, which generated a fresh key with `AES256GCM.generate_key()` and wrote it straight over the old one. Every open of an unprotected store replaced the key that store was encrypted under.

It hid perfectly. The process doing the overwrite kept the new key in memory and carried on encrypting and decrypting without complaint, so nothing looked wrong until the *next* process opened the store and got `cryptography.exceptions.InvalidTag` on every row — by which point the original key existed nowhere.

It fired on the SkyNAS container on 2026-09-22: a `docker exec mnemosyne python -c "... create_memory_system() ..."` run during an audit rotated `/root/.memcore/master.key` at 23:40:47 UTC. The REST process kept serving normally from its in-memory copy while its own database was already unreadable on disk. A restart would have lost the data permanently.

**Recovery worked completely**, because at the time only `content` and `metadata` were encrypted — `id`, `tier`, `timestamp`, `forgetting_json`, `entities_json` and `embedding` were plaintext columns. The plaintext of the two encrypted columns was pulled out over the REST API while the old key was still resident, then re-encrypted under a new key with every plaintext column carried across verbatim. **[2026-09-24/25]** This is no longer true of entities: they are encrypted (`entities_enc`), so a `docker cp`'d `memory.db` written by current code no longer yields them, and the rescue reads them through `GET /memory/{id}` (which now returns `entities`).

The exposure grew a great deal before it was closed. At 01:45 the container held 1 memory; by 16:30 it held 120 — a full CLAUDE-memory export plus the NOSTRO knowledge base — every one of them written under a key that existed nowhere but in that process's RAM. All 120 were rescued and verified byte-for-byte, then deployed into the volume. [scripts/rescue-container-store.py](scripts/rescue-container-store.py) is that procedure, kept because it is the only way to get data out of a store in this state: `docker cp` gives you the plaintext columns, the running API gives you the encrypted ones, and the script re-encrypts them under a fresh key.

Fixed in [crypto/key_manager.py](src/memcore_memory/crypto/key_manager.py): `load_or_create` now creates a key **only when no key file exists**, and tells raw from wrapped by *content* (JSON with `salt`/`nonce`/`ct` → unwrap; exactly 32 bytes → use as-is; anything else → raise `MasterKeyUnreadable` rather than overwrite). New key files are also written `0600` instead of `0644`. Covered by [tests/test_key_persistence.py](tests/test_key_persistence.py) — the load-twice tests are the point; **never** reintroduce a size-based branch.

**[2026-09-24/25]** That "never overwrite" change covers only the overwrite path. Key loss is now also guarded by:
- `MasterKeyMissing` — no new key is created next to a populated DB (an initialised but empty knowledge graph, `kg_meta` only, doesn't count; `store_meta` and `kg_meta` are bookkeeping, every other table counts as data).
- `MasterKeyMismatch` — `store_meta.key_id`, a fingerprint of the key, is checked in `EncryptedStore.init()`; a key that doesn't match the store is refused rather than used.
- Stale `.master.key.*.tmp` files older than 60 s (a process killed mid-write, each holding a full key copy) are removed by `load_or_create`. If the no-hard-link fallback (`O_EXCL` on CIFS/SMB) crashed mid-write, `master.key` is truncated: it is reported as `MasterKeyUnreadable` and never replaced. Delete it by hand only if no store was ever written under it; otherwise restore it from backup.

**Bug: retrieval barely retrieved.** (Found and fixed 2026-09-22 during a full read of the codebase.) Three independent faults, all silent:

1. **BM25 tokenized with `content.lower().split()`** — whitespace only. On real prose that makes `WireGuard:` and `master.key` tokens in their own right, which no plain query term can match. Searching `WireGuard` against a store with four documents containing the word returned **zero** hits. Now `\w+`, unicode-aware so accented words survive, applied identically to corpus and query.
2. **`TemporalRetriever` and `ImportanceRetriever` never look at the query.** They rank the whole store by recency and importance and return the same order for every search. Fused as equals they carried ~31% of the weight in favour of the same few documents regardless of the question — which is why the newest high-importance memory came back top of everything. They are priors, so they now reorder what the query-dependent retrievers found rather than nominating candidates. See `QUERY_DEPENDENT` in [retrieval/hybrid.py](src/memcore_memory/retrieval/hybrid.py). **[2026-09-24/25]** They are now ranked only among those matches (previously still store-wide, so a tier filter could change the relative order); `query` is optional in their `retrieve()` and ignored either way.
3. **The fusion weights could not affect ranking.** It was `rrf * 0.6 + score * 0.4 * w`: the rank term carried no weight, and the score term compares cosine similarity, unbounded BM25 and a recency weight as if they were the same quantity — the exact thing RRF exists to avoid. Now plain weighted RRF, `w / (k + rank)`.

Making the weights real exposed a fourth problem: they were grid-searched with BGE embeddings, but this deployment falls back to a hash embedder whose similarity is noise, and `vector` is the largest weight. `is_semantic()` now detects that and redistributes the vector share to the arms that work.

Measured on the live store, top-3 precision over eight queries whose terms are present: **22/22 after, against roughly 4/24 before.** Re-measure with `memcore memory recall` rather than trusting the numbers here.

**Bug: the blind index never indexed anything — one missing `r` prefix.** (Found and fixed 2026-09-22.)

`BlindIndex._tokenize` matched with `'\b[a-z0-9]{3,}\b'` written as a normal string, so Python turned each `\b` into a **backspace character (0x08)** and the regex went looking for a literal control byte. No real text contains one, so the tokenizer returned an empty set for every input ever passed to it. `compute_index()` returned `[]`, `search_query_hmacs()` returned `[]`, and encrypted search matched nothing — silently, with no error anywhere. `cat` renders 0x08 invisibly, which is why the line reads as correct in a terminal; `grep -P '\x08'` or a hex dump is what shows it.

Fixed with a raw string, and the `\b` was dropped on purpose: `[a-z0-9]{3,}` already breaks on every non-alphanumeric, while word boundaries would stop `user_name` from yielding `user` and `name`. A scan of the whole source found no other control bytes and no other non-raw regex literal.

The storage half was missing too: there was no column to put an index in and no way to search one. [storage/encrypted_sqlite.py](src/memcore_memory/storage/encrypted_sqlite.py) now has `blind_index_json` (added by `ALTER TABLE` for existing stores), computes it on `put`, and offers `search_by_blind_index()` ranked by term overlap plus `rebuild_blind_index()` for rows written while the tokenizer was broken — `memcore system reindex-blind` runs it. The index key is derived from the content key via `AES256GCM.derive_subkey`, never reused, per OWASP. Live store reindexed 2026-09-22: 120/120 rows carry a non-empty index and keyword search over ciphertext works. **[2026-09-24/25]** It is now reachable from the CLI as `memcore memory search-blind QUERY [--k N]` (exact keywords, no rehearsal); still not over REST or MCP. The Postgres backend indexes rows lacking a blind index automatically at init.

Note the tradeoff this feature makes: the HMACs live in a plaintext column, so anyone holding the database learns which rows share keywords, though not what they are. Covered by [tests/test_blind_index_and_decay.py](tests/test_blind_index_and_decay.py).

**Also found in the full-codebase read, all fixed 2026-09-22:**

- **The vector store never persisted anything.** `_persist()` ran only `if self.index`, and the index exists only with hnswlib, which is installed nowhere here. So vectors were written to memory and thrown away at every restart; the volume confirms no vector file has ever existed. `memcore system reindex-vectors` rebuilds the sidecar from the embeddings in SQLite, which were never lost — 120/120 restored on the live store. **[2026-09-24/25, R1/F1]** SQLite's `embedding` column is the source of truth. `<vector_path>.vectors.json` is a cache shared by every process: written 0600 under an flock (`.vectors.json.lock`) with merge-on-write and unique temp files, format 2 (base64 float32, meta limited to `tier`, plus an `embedder` field next to `dim`), and format 1 still reads. Before this fix the sidecar held the **first 500 plaintext characters of every memory**. An unreadable sidecar is moved aside to `vectors.vectors.json.corrupt-<epoch>-<n>` (kept byte for byte as evidence — may still contain plaintext excerpts, delete once inspected); another dimension's to `vectors.vectors.dim<N>.bak.json`; another embedder's of the same dimension to `vectors.vectors.emb-<label>.bak.json` (e.g. after rebuilding with `EMBEDDING_PROVIDER=bge-small`, which is 384-dim like the hash fallback — restore with `reindex-vectors --reembed`; a plain reindex would copy the hash embeddings across unchanged). Set-aside files are never overwritten (`.bak.<n>.json`) and are 0600. A sidecar without `embedder` counts as `local-hash-<dim>`. `reindex-vectors` now reports "re-embedded N (W of another width than D)" and drops deleted memories' vectors in one write.
- **HNSW labels were list positions**, while `delete()` removes from the middle and shifts every later position, so a single delete made the index map labels to the wrong memories. The index is rebuilt from the lists now rather than mutated. Mismatched embedding dimensions are refused rather than stored.
- **Rate limiting had never run** — the middleware was never attached to an app. Its condition also parsed as `"/add" in path or ("/memory" in path and POST)`, because `and` binds tighter than `or`, so a GET to an `/add` path counted as a write and `DELETE /memory/{id}` counted as nothing. Now attached (behind `settings.rate_limit_enabled`), fixed, and returning a real 429 — raising `HTTPException` from middleware never produced one. **[2026-09-24/25]** Writes are decided by method, so `/consolidate` and `/forget` are limited too; over `POST /mcp/call`, search tools (`memory_recall`, `memory_search_*`) count against the recall bucket (100/min) and write tools against the write bucket (20/min); other reads are not limited.
- **`POST /sync/merge` answered `{"status": "merged"}` for any payload and merged nothing.** It returns **501** now. Wiring it up would make it an unauthenticated write endpoint for anything on the LAN, which needs an auth model rather than a quick fix. `MemoryCRDT.merge` also ignored tombstones — every deleted memory came back on the next sync — and had no `from_dict`, so a received payload could not be merged even in principle. Both fixed and tested, so the primitives are ready when the transport is.
- **`server start` claimed to start an MCP server and a P2P node.** It starts neither; nothing has ever listened on 7742, confirmed against the running container. Message corrected. **[2026-09-24/25]** `docker-compose.yml` no longer publishes 7742 (takes effect at the next redeploy).
- Two library `print()`s still went to **stdout** (`sync/p2p.py`, `sync/gossip.py`), which corrupts the JSON-RPC stream under stdio MCP; a bare `except:` in the WAL swallowed `KeyboardInterrupt` and hid corruption; the PII email pattern had a literal `|` inside a character class.

**Also fixed 2026-09-22, same pass:**
- `ForgettingCurve` gained `decay_model="power_law"` with `power_d` (Wixted & Ebbesen's fit, which keeps a long tail where the exponential collapses) and `rehearse(feedback=)` to scale how much a recall counts. `feedback=1.0` reproduces the original `S = S*1.6 + 0.5` exactly. **Both new fields are persisted** — `to_dict`/`from_dict` on the curve are now the single serialization point for both the SQLite and Postgres backends, because leaving them out would silently turn a power-law memory exponential on the next read, which is the rehearsal bug all over again.
- `KeyManager` refuses to *create* an unprotected key when `MNEM_ENV=prod` (the legacy `MEMCORE_ENV` is still honoured). Creation only: gating loads too would strand a running deployment from its own data, and the SkyNAS container relies on exactly such a key. **[2026-09-24/25]** `MNEM_ENV` is a recognised variable and no longer triggers the "unknown environment variable" warning. k8s sets `MNEM_ENV=prod`; the Dockerfile and the SkyNAS `docker-compose.yml` deliberately don't.

**Bug: `MNEM_*` env vars are silently ignored — data never lands in the named volume.** (Fixed and deployed 2026-09-22.)

`config.py` had `env_prefix = "MEMCORE_"`, but `Dockerfile`/`docker-compose.yml` (and every doc/example) set `MNEM_DATA_DIR`, `MNEM_ENCRYPTION_ENABLED`, etc. Since pydantic-settings only maps env vars matching the configured prefix, `MNEM_DATA_DIR=/data` was never read, and `Settings.data_dir` fell back to its default `Path.home() / ".memcore"`. Inside the container that resolves to `/root/.memcore` (container runs as root) — **not** `/data`, which is where the named volume `memcore-memory-100_mnem_data` is mounted. Verified by `docker exec mnemosyne env` (shows `MNEM_DATA_DIR=/data`) vs `docker exec mnemosyne ls $HOME/.memcore` (shows the real, live `memory.db`) vs the volume's actual host directory (`/var/lib/docker/volumes/.../​_data`, empty except `.`/`..`).

**Impact at the time**: all data the container had ever stored lived only in that one container's writable layer, not covered by the named volume, so:
- The backup command in the README backed up nothing until this was fixed and the container recreated with the fix live.
- `docker compose down` / `docker rm mnemosyne` / any container recreation would have **destroyed all memory data** — `restart: unless-stopped` only protects against restarts of the *same* container, not recreation.

**There was a second bug stacked underneath it.** Fixing the prefix alone was *not* enough. `db_path`, `key_path`, `vector_path`, `audit_log_path` and `working_buffer_path` were declared at class scope as `data_dir / "..."`, which pydantic evaluates **once, at class-creation time, against the default `data_dir`**. So even with `MNEM_DATA_DIR=/data` correctly parsed, only `data_dir` moved — every derived path still pointed into `~/.memcore`. Found on 2026-09-22 when a test run against an isolated data dir unexpectedly hit the real `/home/skynas/.memcore/master.key`. Both are now fixed: the paths default to `None` and a `model_validator(mode="after")` resolves them against the effective `data_dir`, with per-path overrides (`MNEM_DB_PATH`, …) still winning. Covered by [tests/test_config_paths.py](tests/test_config_paths.py) — **do not** re-inline those defaults.

**Fixed and deployed 2026-09-22** (first deploy 16:39 +0200). [config.py](src/memcore_memory/config.py) uses `env_prefix = "MNEM_"` (via `SettingsConfigDict`) and resolves derived paths at runtime. The container was rebuilt and recreated with [scripts/deploy-skynas.sh](scripts/deploy-skynas.sh). Verified after the fact:

- `data_dir: /data`, `db_path: /data/memory.db` — the volume, not `~/.memcore`
- `/root/.memcore` no longer exists in the container
- the volume holds `memory.db` (1.2MB), `master.key`, `memory.kg.db` — it was empty for the whole prior life of the deployment
- **survives a restart**: 120 memories intact, `master.key` unchanged (this was fatal before)
- the README backup command now produces a 619KB archive instead of an empty one

Re-run the deploy script after any change that has to reach the container; see [Deploying the 2026-09-24/25 changes](#deploying-the-2026-0924-25-changes) for what it checks now. It rescues on every run rather than trusting a directory prepared earlier — the first version of it did trust a fixed path, and would have destroyed 119 of 120 memories.

Beware that the prefix fix activates **every** `MNEM_*` var at once, including ones that were previously inert. `docker-compose.yml` had `MNEM_EMBEDDING_DIM=768`, which would have re-dimensioned the embedder away from the 384-dim vectors already stored; it is now 384, matching `config.py`. Check any new var against the defaults in `config.py` before adding it — and since 2026-09-24/25 an invalid value stops every entry point with a `ConfigError`.

**Bug: rehearsals were persisted and then thrown away on every read.** (Found and fixed 2026-09-22.)

`MemoryItem.__post_init__` set `forgetting.strength` from the tier unconditionally — including in the construction inside `EncryptedStore._row_to_item`. `rehearse()` correctly did `S = S*1.6 + 0.5` and `store.put` wrote it to SQLite, but the very next `store.get` overwrote `strength` with the tier baseline again. Net effect: **the Ebbinghaus curve never strengthened with use.** A memory recalled a hundred times decayed on exactly the same schedule as one never touched again, which nullifies the central feature of the system. This is the likely explanation for the live SkyNAS memory dropping from R=0.869 to R=0.562 over a few hours despite being recalled repeatedly.

Fixed in [core/tiers.py](src/memcore_memory/core/tiers.py): `forgetting` now defaults to `None` and is seeded from `TIER_BASE_STRENGTH` only for genuinely new items; a curve passed in (i.e. loaded from storage) is authoritative. Because the old clobber also *accidentally* raised strength on promotion, [storage/encrypted_sqlite.py](src/memcore_memory/storage/encrypted_sqlite.py) `update_tier` now raises strength to the new tier's floor explicitly — and never lowers it, so a demoted memory keeps what rehearsal earned. Covered by [tests/test_forgetting_persistence.py](tests/test_forgetting_persistence.py). The Postgres backend passes `forgetting=` explicitly too, so it gets the same fix. **[2026-09-24/25]** Rehearsal now updates only `forgetting_json`, never the whole row; `MnemosyneMemory.get()` returns `None` if the memory is deleted between read and rehearsal; recall rehearses only the top 3 matched results (by 1, 1/2, 1/3) and only the top hit adds to the rehearsal count.

**Bug: BM25 silently lost the terms that mattered most.** (Found and fixed 2026-09-22.)

`BM25Okapi`'s IDF is `log((N-df+0.5)/(df+0.5))`, which hits **zero** once a term appears in half the corpus and goes negative beyond that. The retriever drops non-positive scores, so on a small personal store the most characteristic terms returned nothing: with 4 memories, `"SkyNAS"` (present in 2) scored 0 hits while `"Docker"` (present in 1) worked fine. The more central a term is to your own memory, the less findable it was.

Switched to `BM25L` in [retrieval/retrievers/bm25.py](src/memcore_memory/retrieval/retrievers/bm25.py), which keeps non-matching documents at zero while scoring every genuine match positive. `BM25Plus` was rejected: it scores *every* document positive, which would have destroyed precision in the RRF fusion. The index is cached instead of re-tokenising the whole store on every query; **[2026-09-24/25, F24]** the cache now rebuilds whenever any content changes, not only when ids or the count change. Covered by [tests/test_bm25_recall.py](tests/test_bm25_recall.py).

## The 2026-09-24/25 audit-fix pass — other behaviour changes

Items not already folded into the sections above.

### REST API behaviour (2026-09-24/25)

- **API key on SkyNAS (2026-09-25)**: the key lives in `/home/skynas/.config/memcore/api-key` (0600, off the share) and, as `MNEM_API_KEY=...`, in `container.env` next to it, which `docker-compose.yml` loads via `env_file` (`required: false`: without the file the API stays open). `.mcp.json` passes `--api-key-file` with that path, since the desktop app doesn't source the shell profile. The Windows bridge source is now [scripts/windows/memcore_remote_mcp.py](scripts/windows/memcore_remote_mcp.py); it reads `%MEMCORE_API_KEY%` or `C:\Users\A\.memcore-api-key`.
- **Auth**: with `MNEM_API_KEY` set, every route except `GET /health` and `GET /livez` needs `Authorization: Bearer <key>` or `X-API-Key: <key>`. Unset (the SkyNAS default today) keeps the API open. The MCP bridge (`server mcp --remote`) sends `MNEM_API_KEY` / `--api-key` from its own environment.
- `GET /livez` returns `{"status":"ok"}` without touching the store.
- `GET /health`: `ok`, `degraded` (200, `reasons`) or `unhealthy` (503 when no row is readable or more than 10% cannot be decrypted); fields `reasons`, `unreadable_count`, `unreadable` (up to 50 ids). `degraded` also when the knowledge graph couldn't be opened, or `memory.kg.db` was created empty at startup for a store that already held memories.
- `GET /memory/{id}` does **not** rehearse by default (`?touch=true` does); MCP `memory_get` rehearses by default (`touch: false` doesn't). It now returns `entities`, `timestamp` and `forgetting`, and answers 422 `{detail, id}` for a row that exists but can't be decrypted (over MCP: a tool error naming row and column; never plaintext).
- `POST /recall` and MCP `memory_recall` accept `rehearse: false`; `POST /recall` returns 503 when every query-dependent retriever failed. `POST /memory` returns 422 for input the core rejects (importance outside [0,1], embedding dimension mismatch).
- `POST /forget` and MCP `memory_forget` return `{forgotten, demoted, promoted, errors}` (undecodable rows as `{"id", "error": "unreadable: ..."}`). `POST /consolidate` returns `{status, promoted}`; MCP `memory_consolidate` returns `{promoted, before, after}`.
- A `MasterKeyError` mid-request becomes a 503; at startup `memcore server start` still exits with a one-line message.

### MCP changes (2026-09-24/25)

- `memory_export`/`memory_import` take a **file name inside `<data_dir>/exports`** (`/data/exports` in the container), not an arbitrary path. Exports are plaintext JSON, written 0600, never overwrite without `overwrite: true`, v2 lossless format (ids, timestamps, forgetting curves); legacy exports still import. Export skips rows it can't decrypt and reports them (`skipped`, `partial: true`, `skipped_reasons`). Import validates the whole file first and is all-or-nothing (a failed write removes every row that call wrote); it restores id, timestamp and curve when any of id/tier/timestamp/forgetting is present, and assigns a missing tier by the `memory_add` rule applied to the entry's own timestamp.
- NaN/Infinity anywhere in tool arguments or import files (and boolean importance) is refused, on every transport.
- `memory_search_temporal` needs no query (ignored). `memory_search_graph` returns exact-entity matches first, then entities named inside the text, then graph neighbours within depth, and reports `exact_matches`. `memory_stats` and `health_check` report `unreadable_count` and ids.
- Bridge: a pool timeout is "not delivered", like a connect failure; `--timeout N` expiry is reported as "did not answer '<tool>' within Ns ... may still complete on the server" — check with `health_check`/`memory_list` before retrying. Other HTTP 4xx become tool errors with status and body.

### Knowledge graph (2026-09-24/25)

- Startup no longer refuses when `memory.kg.db` can't be opened or migrated: it logs `[kg] ... continuing WITHOUT the knowledge graph`, serves memories without the graph (graph arm and entity tools fail with `KnowledgeGraphUnreadable`, graph writes skipped, `MnemosyneMemory.kg_error` holds the reason, file never modified) and `/health` is `degraded`. Recover by repairing or moving the file aside; the next start creates a new graph and rebuilds memory links. If `kg.db` was missing while the store held memories, a loud stderr WARNING says hand-added entities and relations are lost.
- A `kg.db` with a newer schema, or a v1/v2 file whose labels don't decrypt under the loaded key, is refused and left byte-for-byte unchanged. If an older release wrote into an upgraded file, the next start logs `[kg] ... upgraded rows written by an older version` and rebuilds links.
- Deleting a memory only garbage-collects entity nodes that memory-linking created; entities from `kg_add_entity`, `add_relation` endpoints and all pre-existing nodes are never deleted automatically.
- Known gap (R12): memories added while the graph was unavailable stay unlinked if an old graph file already marked complete is restored; run `memcore system reindex-kg`.

### Postgres backend (2026-09-24/25)

`content`/`metadata` bound to their row with AAD (`aad_v` added automatically; old rows still read; `PostgresStore.reencrypt_legacy_rows()` is one-way). `get()` raises `CorruptRow`; `list_all`/`list_by_tier` skip and log unreadable rows (or `strict=True`), newest first; `verify()`/`scan()` match SQLite. The old untrained ivfflat index `idx_memories_embedding` is dropped at init.

### SDK, scripts, ops (2026-09-24/25)

- SDK `MnemosyneClient`/`AsyncMnemosyneClient` have **no default `base_url`** (it used to be `http://localhost:8000`); pass it or set `MNEM_REMOTE_URL`, else `ValueError`. `api_key` falls back to `MNEM_API_KEY`, sent as Bearer. Ids and entity names are URL-escaped as one path segment.
- `scripts/sync-rest-to-local.py`: `--data-dir`, `--update-changed`, `--create` (a `--data-dir` without `memory.db` + `master.key` is refused without it), `--dry-run` no longer starts the local MCP server (no init/migration/stamp against the local store). Ignores inherited `MNEM_*` except the password, no longer rehearses local memories, carries entities (warns and counts `without_entities=N` against an older REST server). Copies still get `timestamp=now` and a fresh curve.
- `scripts/store_fingerprint.py DATA_DIR` prints a row-level fingerprint as JSON; `--compare OLD.json NEW.json` prints differences and exits 1 if any. Stdlib only.
- `docker-compose.yml` no longer publishes 7742. Dockerfile `HEALTHCHECK` and k8s probes use `/livez`. k8s: `kubectl apply -k k8s/` after creating Secret `mnemosyne-secrets` by hand as in `k8s/secrets.example.yaml` (`postgres-password`, `master.key` raw base64-32-bytes or a wrapped file, optional `master-password` and `api-key`); single replica; image built with `docker build --build-arg BACKEND=postgres -t mnemosyne-memory:1.0.0-pg .`.
- `make test` no longer uses `--cov` (separate `make coverage`). `pyproject.toml` requires `cryptography>=44`, `mcp>=2.2`, `jsonschema>=4.20`; `networkx`, `websockets`, `sqlalchemy` (now in the `postgres` extra) and `argon2-cffi` are no longer core dependencies.
- `.gitignore` ignores `*.tgz` (the deploy's volume tarball contains the raw `master.key`). `.dockerignore` excludes store, key and backup files at any depth.
- Audit log (`storage/audit_log.py`, not wired in): hash-chained with a `.head` file; `verify_chain()` detects truncation and head deletion/forgery, `log()` appends an `audit-tamper-detected` entry instead of repairing silently; an unparseable last line raises `AuditLogCorrupt`, `repair_torn_tail()` removes a torn final line. Not detected: deleting both log and head, or replaying an older head with a matching truncation.
- PII filter (not wired in): card numbers found as 13-19-digit Luhn-valid windows inside digit-group runs; only the card window is redacted.

### Deliberately not done (owner decisions)

- P2P transport and `/sync/merge` (F32/F48): need an auth model and wire contract first.
- No lifecycle scheduler (F22): idle stores don't age working memories until the next add or `forget`.
- `/health` still decrypts every row (F101 `count_by_tier()` not implemented).
- `--help` and `server mcp --remote` still fail on an invalid `MNEM_*` value (R14), by design.
- A shared Postgres knowledge graph (F49) would be a new feature.
- `deploy-skynas.sh` allow-rules in `.claude/settings.local.json` (R5) must be removed by the owner by hand.

## Critical fixes / gotchas

1. **Recall syntax**: `memcore memory recall "query" --k 2` — `--k` is a named flag (default 10), not `-k` and not positional. Same for `--tier`.
2. **Missing embeddings**: `sentence-transformers`/`torch` are **not** installed anywhere on SkyNAS — not in the container, not in `.venv` (an older note here claimed they were; `pip list` says otherwise). So `BGEEmbedder` has always been falling back to `LocalHashEmbedder`, a deterministic hash, while `/health` and `config_get` reported `bge-small`. The old `Dockerfile` caused this: `pip install -e ".[all]" || pip install -e .` swallowed the failure and shipped an image that advertised a model it didn't have. The Dockerfile now takes one `EMBEDDING_PROVIDER` build arg, labels the image with what actually got installed, and fails the build if the extras don't import:
   ```bash
   docker compose build --build-arg EMBEDDING_PROVIDER=bge-small   # ~2GB of torch, model downloaded at first use
   ```
   Default is `local` (hash embeddings), which is what the deployment has been running all along. **[2026-09-24/25]** Switching to a real model is now detected: the sidecar records its `embedder`, the old one is set aside on the first write, and `memcore system reindex-vectors --reembed` must be run to restore vector search.
3. **Zip/git bloat**: never zip with `.venv/` or `.git` included (`.venv` pulled objects into `.git/objects` once and produced a 9.0GB archive). Fix:
   ```bash
   zip -r out.zip . -x "*/.venv/*" "*/__pycache__/*" "*.pyc" "*/dist/*" "*/.git/*" "*/.pytest_cache/*"
   ```
   For GitHub pushes: `git config user.email "kovacsdobosadam@gmail.com"`, `git config user.name "Kovács-Dobos Ádám"`, and keep `.venv/` in `.gitignore`.
4. **File transfer**: `transfer.sh` is blocked by the SkyNAS firewall — use `curl -F "file=@zip" https://0x0.st` instead.
5. **[2026-09-24/25]** **`system init` on an existing store is harmless**: it prints `Data dir: <abs path>`, loads the existing key (asking for the password if the key is wrapped), never overwrites a key file, and says `Loaded existing key ...; nothing was replaced` or `Created new key ...`, plus `password-protected` or `UNPROTECTED`. It also creates the store and KG tables. The real risk is running it against the wrong or empty data dir, where it creates a *new* key and store there — check `MNEM_DATA_DIR` first. `init --password X` on an existing raw key does **not** protect it; use `memcore system protect`. (This gotcha used to say "do not re-init — it would create/overwrite the key material", which is no longer true.)
6. **Password handling** (rewritten 2026-09-22 — this used to say there was no non-interactive path; there is one now). `KeyManager.load_or_create` resolves the password in this order: explicit argument → `$MNEM_MASTER_PASSWORD` → interactive `getpass()`, **and only prompts when stdin is a TTY**. When it isn't, it raises `MasterPasswordRequired` with a diagnostic instead of consuming the protocol stream or aborting cryptically. The CLI takes a global `--password` flag (`memcore --password X memory list`), which also reads that env var.
   - The interactive prompt is still the default for humans; nothing is stored on disk. Supplying the password by env var does expose it to the process environment — the usual `PGPASSWORD` tradeoff — so prefer the prompt when you have a terminal.
   - `load_or_create` creates a key **only when the file doesn't exist** (and not next to a populated DB), and refuses to overwrite one it can't parse. Any change to that branch risks the key-destruction bug above; run [tests/test_key_persistence.py](tests/test_key_persistence.py) if you touch it.
   - **[2026-09-24/25]** `memcore system protect [--password X]` wraps an unprotected key in place (the key bytes are unchanged). Running REST/MCP servers keep working, but every next start needs `--password` or `MNEM_MASTER_PASSWORD` — on SkyNAS the container's environment must be updated **before** any restart.
7. **stdout is sacred under stdio MCP.** Library diagnostics in `embeddings/`, `retrieval/` and `storage/` used to `print()` to stdout, which corrupts the JSON-RPC stream the moment a model loads. They now go to stderr, and `server mcp` additionally wraps startup in `contextlib.redirect_stdout(sys.stderr)`. If you add a `print()` anywhere in the library import path, send it to stderr.

## MCP access for this project

- **From this SkyNAS host / Claude Code sessions here**: two modes exist and the choice is a real one, because the two stores hold different data.

  **Bridge mode** — `server mcp --remote http://192.168.1.183:8000` (or an exported `MNEM_REMOTE_URL`) — proxies to the container, so MCP sees exactly what the REST API serves (120 memories) and needs no master password, since the container unlocked its key at startup. Verified end-to-end: 29 tools, `data_dir=/data`. If the container ever sets `MNEM_API_KEY`, pass it with `--api-key` / `MNEM_API_KEY`.

  **Local mode** — plain `server mcp` with `$MNEM_MASTER_PASSWORD` — opens `/home/skynas/.memcore` (retired, see the store list). [.mcp.json](.mcp.json) selected it from commit `32906cd` until 2026-09-25 and now selects bridge mode. Opening that store with current code migrates it one-way; inspect a copy (`MNEM_DATA_DIR=<copy>`).

  Whichever is configured, **know which store you are talking to**. Both open cleanly, both answer every query, and they disagree — so a client pointed at the wrong one looks perfectly healthy while serving memories nobody wrote. That is what made the first version of `.mcp.json` wrong: it silently switched stores without saying so. **[2026-09-24/25]** `server mcp` now prints `[memcore] bridge mode -> <url> (from MNEM_REMOTE_URL|--remote)` or `[memcore] local store (<backend>): <data_dir>` to stderr at startup — check that line. Bridge mode is covered by [tests/test_mcp_remote.py](tests/test_mcp_remote.py), and `tests/test_audit_J_round2.py` runs the same call sequence through local and bridge mode against a real uvicorn server and requires identical results.

  The bridge needs `GET /mcp/tools` and `POST /mcp/call` ([api/rest.py](src/memcore_memory/api/rest.py)), which dispatch through the same `HANDLERS` table as the stdio server — the tool surface can't drift between the two. Against an instance built before those routes existed it exits with "predates the REST-backed MCP bridge", not an obscure failure.

  One semantic difference in bridge mode: `memory_export`/`memory_import` resolve on the *server*. **[2026-09-24/25]** They take a file name inside `<data_dir>/exports` — `/data/exports` in the container — not an arbitrary path (see [MCP changes](#mcp-changes-2026-0924-25)).

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
  The last two write to the live store (recall rehearses the top results). For experiments use a throwaway server: `MNEM_DATA_DIR=$(mktemp -d) memcore server start --host 127.0.0.1 --port 8765`.
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
  Things to check by hand on the Windows side (can't be verified remotely): both paths above actually exist, the bridge script points at `http://192.168.1.183:8000`, the log directory has write access, and — after the redeploy — that it doesn't treat a `degraded` `/health` as down, and sends the key if `MNEM_API_KEY` is ever set on the container.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q -p no:cacheprovider     # ~3.5 min
```

Current state (2026-09-25, uncommitted working tree): **611 passed, 0 failed, 2 skipped.** The suite now includes the per-group audit tests (`tests/test_audit_*`), a manifest/tool-schema parity test (`tests/test_audit_h_ops.py::test_mcp_manifest_matches_tool_schema`), and end-to-end bridge tests that start the REST app on uvicorn at `127.0.0.1:0`. `test_excellent.py::test_reranker_fallback` uses a nonexistent model name, so the suite never downloads `bge-reranker-large`.

`tests/conftest.py` has an autouse `isolate_data_dir` fixture. It exists because the suite previously ran against the real `~/.memcore` — `test_memory.py` called `create_memory_system()` with no isolation and was writing test memories into the live encrypted store. **Never remove it**, and never call `monkeypatch.undo()` in a test: that also undoes the isolation. A draft test did exactly that on 2026-09-25 and created an empty `/home/skynas/.memcore/exports` (0700, 04:06:48); nothing else was touched, and it can be removed with `rmdir /home/skynas/.memcore/exports`.

## Known discrepancies found during 2026-09-22 verification pass

A prior task brief for this project contained some claims that did not match the live system when checked against `docker ps`, `curl /health`, `curl /recall`, and the actual source (`core/tiers.py`). Recorded here so future sessions don't re-propagate them:

- **Tier names**: brief said `working/short/long/archival`. Actual enum in `core/tiers.py` is `sensory/working/episodic/semantic`. Fixed in `.claude/memcore.memory.json`.
- **"SkyNas first memory" record**: the brief described a memory added with content `"SkyNas first memory"`, id `8d235d7f-ab86-45a6-899b-3a5cf9edb619`, retention R=0.88. At the time the database held exactly one memory, and it was a *different* record: id `166657ab-1e3f-4d4d-b668-d5d6d0656eff`, content `"SkyNAS Docker-only telepítés sikeres"`, tier `working`, importance 0.5. Either the DB was reset/replaced since that note was written, or the note was never actually run against this DB. Don't assume the old id/content still exists without checking `GET /memories` first.
- **DB size**: 28672 bytes was stale; the size then was 45056 bytes (since grown to ~1.2MB with 120 memories).

Everything else in the original brief (author, repo URL, PyPI/Docker Hub names, retrieval weights, encryption scheme, container/volume names, endpoint IP, critical fixes 1-4) checked out against the live host and source. The brief's **MRR@10=0.85** was only ever the project's own claim and was never measured here — it is recorded above as unverified, not as fact.

## Second verification pass, 2026-09-22 (same day, follow-up task)

A follow-up brief proposed tiers `["working","midterm","longterm"]` for `.claude/memcore.memory.json`. Still doesn't match the code (`core/tiers.py`: `sensory/working/episodic/semantic`, also confirmed by `tier_counts` keys in the live `/health` response) — kept the verified names, didn't overwrite them with the new unverified list. This is the second different tier-name list handed to this project in one day (`working/short/long/archival`, then `working/midterm/longterm`); neither matches reality, so verify against `core/tiers.py` again before trusting a future one.

This pass also found and fixed the env-prefix persistence bug described above under "Known issues" — that's new information, not a correction of anything stated earlier today.
