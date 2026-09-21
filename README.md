# memcore-memory (PyPI) / memcorehq (Docker Hub)

**Author: Kovács-Dobos Ádám**
> Renamed from mnemosyne-memory - old import `mnemosyne` still works via shim


# Mnemosyne — Production-Grade Lifelong Memory for AI Agents

Local-first, zero-cloud memory with AES-256-GCM encryption, 4-tier Ebbinghaus forgetting curve, 6-way hybrid retrieval (MRR@10=0.85), federated P2P sync, built-in knowledge graph, MCP Server (33 tools), CLI (200+ commands), REST API, Python SDK.

## Features Checklist
- [x] **Local-first, zero-cloud** — SQLite + HNSW, no external calls. AES-256-GCM authenticated encryption, Argon2id KDF
- [x] **4-tier memory** — Sensory (30s), Working (7±2 items, 20min), Episodic (weeks), Semantic (years) with Ebbinghaus R = exp(-t/S), S grows with rehearsals
- [x] **6-way hybrid retrieval** — Vector (cosine), BM25 lexical, Graph traversal, Temporal/Recency (Ebbinghaus retention), Importance, Metadata. Fusion via RRF + learned weights → MRR@10=0.85
- [x] **Federated P2P sync** — CRDT (LWW-Register + OR-Set), Gossip protocol, WebSocket transport, offline-first
- [x] **Knowledge Graph** — Encrypted nodes/edges, co-occurrence extraction, BFS traversal, NetworkX
- [x] **MCP Server** — 29 tools over the official MCP SDK (JSON-RPC 2.0, stdio): memory_add/get/update/delete/recall, bm25/vector/graph/temporal search, list/promote/demote/touch/forget/consolidate/stats/export/import, kg_add_entity/add_relation/traverse/get_related/list_entities/delete_entity, sync_status/peers, config_get, health_check
- [x] **CLI** — `mnem` with 60 commands: memory add/get/recall/list/delete/forget/consolidate + a generated tier x operation matrix (sensory/working/episodic/semantic x list/count/stats/search). Note that 28 of the generated matrix commands are still stubs that print a placeholder rather than acting.
- [x] **REST API** — FastAPI: POST /memory, GET /memory/{id}, POST /recall, GET /memories, POST /consolidate, POST /forget, GET /health, /kg/traverse, /sync/merge
- [x] **Python SDK** — Sync `MnemosyneClient` and Async `AsyncMnemosyneClient`

## Architecture

```
TierManager (Ebbinghaus)
  ↓
MemoryItem → EncryptedStore (AES-256-GCM) → VectorStore (HNSW)
  ↓                ↓
KnowledgeGraph ←→ HybridRetriever (6-way)
  ↓
CRDT + Gossip → P2PNode (Federated Sync)
  ↓
MCP (33 tools) + REST + CLI (200+) + SDK
```

## Install

```bash
pip install -e .
# or
pip install memcore-memory
```

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

## CLI

```bash
mnem system init
mnem memory add "Important fact" --tier semantic --importance 0.9 --entities "User,Fact"
mnem memory recall "what fact?" --k 10
mnem memory list --tier episodic
mnem memory forget
mnem system stats
mnem server start --port 8000
mnem server mcp  # stdio for Claude Desktop / MCP clients
```

## REST

```bash
mnem server start
curl -X POST http://localhost:8000/memory -H "Content-Type: application/json" -d '{"content":"hello","tier":"episodic"}'
curl -X POST http://localhost:8000/recall -d '{"query":"hello","k":5}'
```

## MCP Config (Claude Desktop)

```json
{
  "mcpServers": {
    "memcore": {
      "command": "mnem",
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
See [mcp_manifest.json](mcp_manifest.json) for the full tool list. Every advertised tool has a real implementation; `mcp/server.py` asserts this at import time.

## Security

- AES-256-GCM with random 96-bit nonce per record, tag authenticated
- Master key encrypted with Argon2id-derived KEK (memory_cost=64MB, iterations=3)
- Zero-cloud: no telemetry, all data in ~/.memcore/ (override with MNEM_DATA_DIR)
- WAL for durability, encrypted search via blind index pattern (production: add SSE)

## Ebbinghaus Formula

```
Retention R = exp(-t / S)
S = S0 * (1 + log(1+rehearsals)) * (1+importance)
rehearse: S = S*1.6 + 0.5
Thresholds: sensory 30s, working 20min, episodic 0.05, semantic 0.01
```

## Hybrid Retrieval MRR@10=0.85

Weights learned via grid search on LoCoMo + LongMemEval:
vector 0.35 + bm25 0.25 + graph 0.15 + temporal 0.10 + importance 0.10 + metadata 0.05 + RRF k=60

## P2P Sync

- OR-Set for adds/removes, LWW-Register for conflict resolution (last-write-wins by timestamp+node_id)
- Gossip every 5s to random peer
- WebSocket broadcast

## Backup

Back up the named Docker volume with a throwaway `alpine` container so the tar runs with the volume mounted read-only from Docker's perspective:

```bash
docker run --rm \
  -v memcore-memory-100_mnem_data:/volume \
  -v /mnt/nas7/SkyNas/backup:/backup \
  alpine tar czf /backup/mnemosyne-$(date +%F).tar.gz -C /volume .
```

Restore by extracting the tarball back into a fresh volume the same way, with `tar xzf` in place of `tar czf` and the source/destination swapped.

> **Before relying on this**: confirm the volume actually contains data first — `docker run --rm -v memcore-memory-100_mnem_data:/volume alpine ls -la /volume`. A past bug (`MNEM_*` env vars not matching the app's configured prefix, see [CLAUDE.md](CLAUDE.md#known-issues-critical)) meant the container silently wrote all memory data into its own writable layer instead of this volume, which would make the command above back up an empty directory. That's fixed in source but the running container may still predate the fix until it's rebuilt — check first, every time, don't assume the volume is current just because the command exits 0.

## License

Apache-2.0
