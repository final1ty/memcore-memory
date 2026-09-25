# memcore-memory (PyPI) / memcorehq (Docker Hub)

**Author: Kovács-Dobos Ádám**
> Renamed from mnemosyne-memory - old import `mnemosyne` still works via shim (including `python -m mnemosyne.cli.main ...`)


# Mnemosyne — Local-First Encrypted Memory for AI Agents

Local-first, zero-cloud memory with AES-256-GCM encryption, a 4-tier Ebbinghaus forgetting curve, 6-way hybrid retrieval, an encrypted knowledge graph, MCP Server (29 tools), CLI (67 commands), REST API and Python SDK.

## Features Checklist
- [x] **Local-first, zero-cloud** — encrypted SQLite store (WAL journal mode) plus an in-process vector index (NumPy cosine search; hnswlib is used if installed via the `all` extra). No external calls. AES-256-GCM authenticated encryption, Argon2id-wrapped master key
- [x] **4-tier memory** — Sensory (30s), Working (7-item cap, 20 min), Episodic (weeks), Semantic (years) with Ebbinghaus R = exp(-t/S) (or an optional power-law curve); S grows with rehearsals
- [x] **6-way hybrid retrieval** — Vector (cosine), BM25L lexical, Graph, Metadata, plus Temporal/Recency and Importance as priors that only re-rank what the first four matched. Weighted RRF fusion (k=60). See [Hybrid Retrieval](#hybrid-retrieval) for what that means on a store without a real embedding model
- [x] **Knowledge Graph** — entity names encrypted (keyed-HMAC node ids, AES-GCM encrypted labels and props bound to their row; the graph's shape is visible), BFS traversal, plain SQLite (`memory.kg.db`). Entities come from the caller, or from a capitalised-word heuristic when `MNEM_AUTO_EXTRACT_ENTITIES=true`
- [x] **MCP Server** — 29 tools over the official MCP SDK (JSON-RPC 2.0, stdio): memory_add/get/update/delete/recall, bm25/vector/graph/temporal search, list/list_all/promote/demote/touch/forget/consolidate/stats/export/import, kg_add_entity/add_relation/traverse/get_related/list_entities/delete_entity, sync_status/peers, config_get, health_check
- [x] **CLI** — `memcore` (also `mnem`, `mnemosyne`, `memcore-memory`) with 67 commands: memory add/get/recall/search-blind/list/delete/forget/consolidate, system init/protect/stats/verify/health/migrate-aad/reindex-blind/reindex-kg/reindex-vectors, server start/mcp, kg traverse/add-entity, sync status/add-peer, plus a generated tier x operation matrix. 28 of the generated matrix commands (`<tier>-export/clear/touch-all/decay-report/importance-boost/pin/unpin`) are still stubs that print a placeholder rather than acting
- [x] **REST API** — FastAPI: POST /memory, GET /memory/{id}, DELETE /memory/{id}, POST /recall, GET /memories, POST /consolidate, POST /forget, GET /health, GET /livez, GET /kg/traverse/{entity}, GET /mcp/tools, POST /mcp/call; POST /sync/merge answers 501. Optional API-key auth via `MNEM_API_KEY`
- [x] **Python SDK** — Sync `MnemosyneClient` and Async `AsyncMnemosyneClient`
- [ ] **Federated P2P sync** — CRDT primitives only (LWW-Register + LWW-element set); no transport, not implemented. `P2PNode.start`, `broadcast_memory` and `GossipProtocol.gossip_loop` raise `NotImplementedError`, and nothing listens on port 7742
- [ ] **Not wired in** — the PII filter, the audit log, the cross-encoder reranker, the Matryoshka embedder and the mTLS settings exist as modules or config fields, but no request path uses them. `config_get` lists them under `not_implemented`

## Architecture

```
TierManager (Ebbinghaus lifecycle)
  ↓
MemoryItem → EncryptedStore (AES-256-GCM, SQLite) → VectorStore (JSON sidecar cache)
  ↓                ↓
KnowledgeGraph ←→ HybridRetriever (4 query arms + 2 priors, weighted RRF)
  ↓
MCP (29 tools) + REST + CLI (67) + SDK

sync/ (CRDT primitives only - no transport, not implemented)
```

## Install

```bash
pip install -e .
# or
pip install memcore-memory
# optional extras: [embeddings] (sentence-transformers + torch), [postgres], [all]
```

Without the `embeddings` extra the embedder falls back to a deterministic hash embedding. Everything works, but vector search is then noise and its fusion weight is set to 0.

## Quickstart

```python
import asyncio
from mnemosyne import create_memory_system

async def main():
    mem = await create_memory_system(password="optional")
    await mem.add("User likes concise answers", tier="semantic", importance=0.9)
    results = await mem.recall("user preferences", k=5)
    print(results)

asyncio.run(main())
```

This opens the store in `MNEM_DATA_DIR` (default `~/.memcore`). `examples/quickstart.py` and `examples/mcp_demo.py` run against a temporary store instead, and `examples/rest_api_demo.py` takes the URL of a throwaway server as its argument (e.g. `python examples/rest_api_demo.py http://127.0.0.1:8765`).

## CLI

```bash
memcore system init                      # creates the key and tables, or loads existing ones; never replaces a key
memcore memory add "Important fact" --tier semantic --importance 0.9 --entities "User,Fact"
memcore memory recall "what fact?" --k 10
memcore memory search-blind "fact" --k 10   # exact-keyword search over the encrypted rows' blind index
memcore memory list --tier episodic --limit 20
memcore memory forget                    # full lifecycle pass; exits 1 if some rows could not be processed
memcore system stats
memcore system health                    # passive JSON check of the store files; no password; writes nothing
memcore system verify                    # decrypts every row; exits 1 if any is unreadable
memcore system protect                   # wraps an unprotected master key with a password
memcore server start --port 8000
memcore server mcp                       # stdio for Claude Desktop / MCP clients
```

Commands that open the encrypted store need the master password when the key is protected: global `--password X`, or `$MNEM_MASTER_PASSWORD`, or an interactive prompt when stdin is a terminal.

## REST

Use a throwaway data dir and port when trying this out, never a live instance:

```bash
MNEM_DATA_DIR=$(mktemp -d) memcore server start --host 127.0.0.1 --port 8765
curl -X POST http://127.0.0.1:8765/memory -H "Content-Type: application/json" -d '{"content":"hello","tier":"episodic"}'
curl -X POST http://127.0.0.1:8765/recall -H "Content-Type: application/json" -d '{"query":"hello","k":5}'
curl http://127.0.0.1:8765/livez     # liveness: {"status":"ok"}, never touches the store
curl http://127.0.0.1:8765/health    # decrypts every row: status ok | degraded (200) | unhealthy (503)
```

Behaviour worth knowing:

- **Authentication** is off unless `MNEM_API_KEY` is set. When set, every route except `GET /health` and `GET /livez` needs `Authorization: Bearer <key>` or `X-API-Key: <key>`. Without it the API is unauthenticated, and `server start` warns on stderr when bound to anything but localhost. Keep an unauthenticated instance on localhost or a trusted LAN.
- **`GET /health`** returns `status` `ok`, `degraded` (HTTP 200, with a `reasons` list, e.g. the knowledge graph could not be opened) or `unhealthy` (HTTP 503, when no row is readable or more than 10% cannot be decrypted), plus `tier_counts`, `unreadable_count` and up to 50 `unreadable` ids.
- **`GET /memory/{id}`** does not rehearse unless `?touch=true`. It returns `entities`, `timestamp` and the full `forgetting` curve, and answers 422 for a row that exists but cannot be decrypted.
- **`POST /recall`** rehearses the top 3 matched results (only the best hit counts towards promotion); send `"rehearse": false` to search without strengthening. It answers 503 when every query-dependent retriever failed.
- **`POST /forget`** returns the lifecycle report `{forgotten, demoted, promoted, errors}`; `POST /consolidate` returns `{status, promoted}`.
- **Rate limiting** (per client IP): 100/min for `/recall` and search, 20/min for writes (every POST/PUT/PATCH/DELETE). Over `POST /mcp/call`, search tools count against the recall bucket and write tools against the write bucket; other reads are not limited.
- **`GET /memories`** defaults to `limit=50` (max 500), newest first, and cuts content at 500 characters.

## MCP Config (Claude Desktop)

```json
{
  "mcpServers": {
    "memcore": {
      "command": "memcore",
      "args": ["server", "mcp"],
      "env": {
        "MNEM_DATA_DIR": "/home/you/.memcore",
        "MNEM_MASTER_PASSWORD": "..."
      }
    }
  }
}
```

`MNEM_MASTER_PASSWORD` is required whenever the master key is password-protected:
stdio transport has no terminal, so the server cannot prompt and will exit with a
diagnostic on stderr instead of hanging. Leave it out if the key is unprotected.
On startup `server mcp` prints which store it serves to stderr, either
`[memcore] local store (<backend>): <data_dir>` or `[memcore] bridge mode -> <url>`.

### Connecting to a server instead of a local store

If the store you want is being served by another process — a Docker container, a
remote host — do **not** point `MNEM_DATA_DIR` at a directory on the client machine.
That opens a second, unrelated database: it answers every query happily and none of
the answers are the memories you stored. Use bridge mode instead, which forwards
every tool to the running instance's `POST /mcp/call`:

```json
{
  "mcpServers": {
    "memcore": {
      "command": "memcore",
      "args": ["server", "mcp", "--remote", "http://HOST:8000"]
    }
  }
}
```

No master password is needed client-side — the server already unlocked its key. If
the server sets `MNEM_API_KEY`, pass the same key with `--api-key` or
`MNEM_API_KEY` in the bridge's environment. An exported `MNEM_REMOTE_URL` also
switches `server mcp` to bridge mode. Both modes expose the same 29 tools through
the same handlers. `memory_export`/`memory_import` take a file name inside
`<data_dir>/exports` **on the server** (`/data/exports` in the container); exports
are plaintext JSON written 0600, never overwrite an existing file without
`overwrite: true`, and imports are all-or-nothing.

See [mcp_manifest.json](mcp_manifest.json) for the full tool list. Every advertised tool has a real implementation; `mcp/server.py` asserts this at import time.

## Security

- AES-256-GCM with a random 96-bit nonce per field. Each encrypted field (`content`, `metadata`, `entities`) is authenticated and bound to its row id and column via AAD. Rows written before AAD existed stay unbound until they are next written or `memcore system migrate-aad` is run (one-way; back up first)
- Not encrypted and not authenticated: `id`, `tier`, `timestamp`, `forgetting_json`, `embedding` and `blind_index_json`. Anyone holding `memory.db` sees these, and the embedding vectors carry information about the content
- Master key: either a raw 32-byte key, or wrapped with an Argon2id-derived KEK (memory_cost=64MB, iterations=3, lanes=4). `memcore system protect` wraps an existing raw key. With `MNEM_ENV=prod` the process refuses to *create* an unprotected key
- The master key is never created next to a populated store, and a key that does not match the store (`store_meta.key_id` fingerprint) is refused rather than used
- Encrypted keyword search via a blind index (keyed HMACs in a plaintext column: reveals which rows share keywords, not what they are). Reachable from the SDK and `memcore memory search-blind`, not over REST or MCP
- New `memory.db` and `memory.kg.db` files are created 0600, the data dir 0700. Existing files keep their mode; tighten by hand with `chmod 600 memory.db* memory.kg.db* vectors.vectors.json`
- Durability comes from SQLite's own WAL journal mode (persisted in the file, 10 s busy timeout). `storage/wal.py` is not used by anything
- Zero-cloud: no telemetry, all data in `~/.memcore/` (override with `MNEM_DATA_DIR`)

## Ebbinghaus Formula

```
Retention R = exp(-t / S)                      (optional power law: R = (1 + t/S)^-d)
S = S0 * (1 + log(1+rehearsals)) * (1+importance)
rehearse: S = S*1.6 + 0.5                      (scaled by feedback in [0,1])
```

Lifecycle (`TierManager`):
- **sensory** — deleted after 30 s unless marked attended and still retained (R > 0.3), which moves it to working
- **working** — where new memories land by default; capped at 7, the oldest overflow and anything older than 20 min or already rehearsed is moved to episodic, never deleted
- **episodic** — deleted when R < 0.05; promoted to semantic only with at least 3 rehearsals, R > 0.6 and importance > 0.8
- **semantic** — never deleted automatically

Importance never picks the starting tier. Nothing schedules the lifecycle: it runs on every add and on `memory forget` / `POST /forget`.

## Hybrid Retrieval

Configured weights: vector 0.35 + bm25 0.25 + graph 0.15 + temporal 0.10 + importance 0.10 + metadata 0.05, fused with weighted RRF (k=60).

- Temporal and importance are query-independent priors: they only re-rank memories that vector, BM25, graph or metadata matched. A query that none of those match returns nothing.
- With the hash-fallback embedder (no `sentence-transformers`), the vector weight is 0 and its share is redistributed.
- Every result carries `matched`; recall only rehearses matched results.
- The project previously advertised MRR@10 = 0.85 on LoCoMo + LongMemEval with real BGE embeddings. That figure has not been reproduced and is not reproducible on the hash fallback; treat it as unverified.

## P2P Sync (not implemented)

`sync/crdt.py` holds tested CRDT primitives: an LWW-Register for values and an LWW-element set for tombstones (remove wins ties; clock skew decides concurrent add/remove). Nothing transports or applies them: the P2P node and gossip loop raise `NotImplementedError`, `POST /sync/merge` returns 501, and there is no WebSocket or gossip transport. `examples/p2p_demo.py` merges two in-memory replicas directly.

## Deployment

### Docker

`docker-compose.yml` publishes only port 8000 and mounts the named volume at `/data`. The image's `HEALTHCHECK` uses `/livez`. Build with real embeddings via `--build-arg EMBEDDING_PROVIDER=bge-small` (default `local`, the hash fallback).

### Kubernetes

```bash
docker build --build-arg BACKEND=postgres -t mnemosyne-memory:1.0.0-pg .
# create the Secret first, by hand, as described in k8s/secrets.example.yaml:
#   postgres-password, master.key (raw 32-byte key or a wrapped key file),
#   optional master-password (for a wrapped key) and api-key
kubectl apply -k k8s/
```

Single replica, Postgres backend, `MNEM_ENV=prod`, probes on `/livez`. The knowledge graph stays a per-instance SQLite file even with the Postgres backend.

## Backup

Back up the named Docker volume with a throwaway `alpine` container so the tar runs with the volume mounted read-only from Docker's perspective:

```bash
docker run --rm \
  -v memcore-memory-100_mnem_data:/volume \
  -v /mnt/nas7/SkyNas/backup:/backup \
  alpine tar czf /backup/mnemosyne-$(date +%F).tar.gz -C /volume .
```

Restore by extracting the tarball back into a fresh volume the same way, with `tar xzf` in place of `tar czf` and the source/destination swapped.

The tarball contains `master.key` alongside `memory.db`, so treat it as key material — and never restore a database without the key it was encrypted under. They are only useful as a pair. Both `memory.db` and `memory.kg.db` run in WAL mode, so their `-wal` files belong in the backup too; the volume-level tar above includes them.

Upgrading an existing store is one-way: on first open this version encrypts the plaintext entity column in place and upgrades `memory.kg.db` to schema v3. An older image cannot read the result correctly, so roll back by restoring a pre-upgrade backup, never by starting the old image on an upgraded store.

> **Before relying on this**: confirm the volume actually contains data first — `docker run --rm -v memcore-memory-100_mnem_data:/volume alpine ls -la /volume`. A past bug (`MNEM_*` env vars not matching the app's configured prefix, see [CLAUDE.md](CLAUDE.md#known-issues-critical)) meant the container silently wrote all memory data into its own writable layer instead of this volume, which made the command above back up an empty directory. That was fixed and deployed on 2026-09-22, but check first, every time — don't assume the volume is current just because the command exits 0.

## License

Apache-2.0
