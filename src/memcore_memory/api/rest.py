
import hmac
import sys
import traceback
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from .schemas import MemoryAddRequest, RecallRequest, RecallResponse, MemoryAddResponse, HealthResponse, MCPCallRequest
from ..core.tiers import Tier
from ..crypto.key_manager import MasterKeyError
from ..mcp.server import SERVER_NAME, SERVER_VERSION, ToolError, _non_finite, call_tool
from ..mcp.tools import SEARCH_TOOLS, WRITE_TOOLS, get_tools_schema
from ..storage.encrypted_sqlite import CorruptRow
from collections import Counter

# Reachable without a key even when one is configured: container healthchecks and
# probes have no way to send one, and neither route reveals any memory.
OPEN_PATHS = {"/health", "/livez"}

# /health answers 503 above this share of unreadable rows (or when none are
# readable), and lists at most this many of their ids.
HEALTH_UNREADABLE_LIMIT = 0.10
HEALTH_MAX_IDS = 50


def _supplied_key(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
    return request.headers.get("x-api-key")

async def _scan(memory_system):
    scan = getattr(memory_system.store, "scan", None)
    if scan is None:
        return await memory_system.store.list_all(), {}
    return await scan()


def create_app(memory_system):
    # Whether the knowledge graph was created empty for a store that already held
    # memories. Judged once, before the first request: created_fresh alone is also
    # true for a brand-new install, which then fills up normally.
    startup = {"kg_recreated": False}

    @asynccontextmanager
    async def lifespan(app):
        if getattr(getattr(memory_system, "kg", None), "created_fresh", False):
            try:
                items, unreadable = await _scan(memory_system)
                startup["kg_recreated"] = bool(items or unreadable)
            except Exception as e:  # noqa: BLE001 - /health reports the store itself
                print(f"[api] could not count memories at startup: {e!r}", file=sys.stderr)
        yield

    app = FastAPI(title="Mnemosyne Memory API", version="1.0.0",
                  description="Local-first encrypted lifelong memory REST API", lifespan=lifespan)

    # A row that exists but cannot be decrypted is a fact about that row, not a
    # server crash: the message names the id and column and carries no plaintext.
    @app.exception_handler(CorruptRow)
    async def corrupt_row(request: Request, e: CorruptRow):
        return JSONResponse(status_code=422, content={"detail": str(e), "id": e.memory_id})

    # A key problem surfacing mid-request (the key is otherwise loaded at startup,
    # where `server start` reports it and exits) means the store cannot be served.
    @app.exception_handler(MasterKeyError)
    async def master_key_error(request: Request, e: MasterKeyError):
        return JSONResponse(status_code=503, content={"detail": f"{type(e).__name__}: {e}"})

    # The limiter existed but was never attached to an app, so the documented
    # per-IP limits had never applied to a single request.
    from ..config import settings
    rate_limited = getattr(settings, "rate_limit_enabled", True)
    if rate_limited:
        from .rate_limit import rate_limit_middleware
        app.middleware("http")(rate_limit_middleware)

    # Opt-in authentication. The SDK sent a Bearer key that nothing ever checked,
    # so every route - /mcp/call included - was open to anything on the LAN. Unset
    # keeps that behaviour, because the deployed container and the Windows bridge
    # send no key today. Registered after the limiter so it runs first: a request
    # without the key is refused before it can use up anyone's bucket.
    api_key = getattr(settings, "api_key", None)
    if api_key:
        expected = api_key.encode()

        @app.middleware("http")
        async def require_api_key(request: Request, call_next):
            if request.method in ("GET", "HEAD") and request.url.path in OPEN_PATHS:
                return await call_next(request)
            supplied = _supplied_key(request)
            if supplied is None or not hmac.compare_digest(supplied.encode(), expected):
                return JSONResponse(status_code=401,
                                    content={"detail": "missing or invalid API key"},
                                    headers={"WWW-Authenticate": "Bearer"})
            return await call_next(request)

    @app.post("/memory", response_model=MemoryAddResponse)
    async def add_memory(req: MemoryAddRequest):
        # Python's JSON parser accepts NaN and Infinity; stored in metadata they make
        # every later GET of that memory a 500, since the response cannot encode them.
        bad = _non_finite(req.metadata, "metadata")
        if bad:
            raise HTTPException(422, f"{bad}: NaN and Infinity are not valid JSON numbers")
        try:
            item = await memory_system.add(content=req.content, tier=req.tier.value if req.tier else None,
                                           importance=req.importance, entities=req.entities,
                                           metadata=req.metadata)
        except ValueError as e:
            # add() refuses what the schema cannot express (a NaN importance, an
            # embedding of the wrong width): the caller's input, not a server fault.
            raise HTTPException(422, str(e))
        return MemoryAddResponse(id=item.id, tier=item.tier.value)

    @app.get("/memory/{mem_id}")
    async def get_memory(mem_id: str, touch: bool = Query(False)):
        # A plain read by default, unlike MCP memory_get, which rehearses unless told
        # otherwise. Backups and scripts/sync-rest-to-local.py read every memory
        # through here, and must not strengthen all of them as a side effect.
        item = await (memory_system.get(mem_id) if touch else memory_system.store.get(mem_id))
        if not item:
            raise HTTPException(404, "Not found")
        return {"id": item.id, "content": item.content, "tier": item.tier.value, "metadata": item.metadata,
                "retention": item.forgetting.retention(), "entities": item.entities,
                "timestamp": item.timestamp, "forgetting": item.forgetting.to_dict()}

    @app.post("/recall", response_model=RecallResponse)
    async def recall(req: RecallRequest):
        """Hybrid search. The top 3 matched results are rehearsed unless rehearse is false."""
        tier_filter = [t.value for t in req.tier_filter] if req.tier_filter else None
        try:
            results = await memory_system.recall(req.query, k=req.k, tier_filter=tier_filter,
                                                 rehearse=req.rehearse)
        except RuntimeError as e:
            # Every query-dependent retriever failed: nothing was searched, which
            # an empty 200 would pass off as "no matches".
            raise HTTPException(503, str(e))
        return RecallResponse(results=results)

    @app.get("/memories")
    async def list_memories(tier: Optional[Tier] = Query(None), limit: int = Query(50, ge=1, le=500)):
        if tier:
            items = await memory_system.store.list_by_tier(tier)
        else:
            items = await memory_system.store.list_all()
        return [{"id": i.id, "content": i.content[:500], "tier": i.tier.value, "timestamp": i.timestamp} for i in items[:limit]]

    @app.delete("/memory/{mem_id}")
    async def delete_memory(mem_id: str):
        # Through the one shared delete, which also drops the knowledge-graph links;
        # and an unknown id is a 404, not a cheerful {"deleted": ...}.
        if not await memory_system.delete(mem_id):
            raise HTTPException(404, f"no memory with id {mem_id!r}")
        return {"deleted": mem_id}

    @app.post("/consolidate")
    async def consolidate():
        promoted = await memory_system.consolidate()
        return {"status": "consolidated", "promoted": promoted}

    @app.post("/forget")
    async def forget_expired():
        # The full lifecycle report; "forgotten" keeps its meaning for old callers.
        if hasattr(memory_system, "lifecycle_pass"):
            return await memory_system.lifecycle_pass()
        return {"forgotten": await memory_system.forget_expired()}

    @app.get("/livez")
    async def livez():
        """Liveness only. /health decrypts every row, which is too heavy for a probe."""
        return {"status": "ok"}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        # list_all() skips rows it cannot decrypt, so a store whose key no longer
        # matched any row answered "ok" with empty tier counts.
        items, unreadable = await _scan(memory_system)
        total = len(items) + len(unreadable)
        reasons = []
        if unreadable:
            reasons.append(f"{len(unreadable)} of {total} rows cannot be read")
        kg_error = getattr(memory_system, "kg_error", None)
        if kg_error:
            reasons.append(f"knowledge graph unavailable: {kg_error}")
        elif startup["kg_recreated"]:
            reasons.append("knowledge graph was created empty at startup for a store that already held "
                           "memories; memory links were rebuilt, hand-added entities and relations are gone")
        failing = bool(unreadable) and (not items or len(unreadable) > HEALTH_UNREADABLE_LIMIT * total)
        body = HealthResponse(status="unhealthy" if failing else ("degraded" if reasons else "ok"),
                              version="1.0.0", tier_counts=dict(Counter(i.tier.value for i in items)),
                              reasons=reasons, unreadable_count=len(unreadable),
                              unreadable=sorted(unreadable)[:HEALTH_MAX_IDS])
        if failing:
            return JSONResponse(status_code=503, content=body.model_dump())
        return body

    @app.post("/sync/merge", status_code=501)
    async def sync_merge(crdt_data: dict):
        """Not implemented. It used to answer {"status": "merged"} and merge nothing.

        The CRDT primitives are real and tested (sync/crdt.py), but nothing wires a
        received payload into this instance's store, and doing so would make this an
        unauthenticated write endpoint: anything on the LAN could inject memories.
        That needs an auth model first, so this reports the truth instead of a
        success a caller would believe.
        """
        return JSONResponse(status_code=501, content={
            "status": "not_implemented",
            "detail": "P2P sync is not wired up. /sync/merge accepts no data and merges "
                      "nothing; it previously reported success regardless. See CLAUDE.md.",
            "received_registers": len((crdt_data or {}).get("registers") or {}),
        })

    @app.get("/kg/traverse/{entity}")
    async def kg_traverse(entity: str, depth: int = Query(2, ge=1, le=5), limit: int = Query(20, ge=1, le=200)):
        results = await memory_system.kg.traverse(entity, depth=depth, limit=limit)
        return {"entity": entity, "results": results}

    # --- MCP bridge -------------------------------------------------------
    # These two routes let an MCP client reach this instance's store without
    # opening the database itself. See mcp/remote.py for why that matters.

    @app.get("/mcp/tools")
    async def mcp_tools():
        return {"server": SERVER_NAME, "version": SERVER_VERSION, "tools": get_tools_schema()}

    @app.post("/mcp/call")
    async def mcp_call(req: MCPCallRequest, request: Request):
        """Dispatch through the same validation and handlers the stdio server uses.

        Tool-level failures come back as ``ok: false`` rather than an HTTP error, so
        the bridge can tell "you called this wrong" apart from "the server is down".
        """
        if rate_limited and (req.name in WRITE_TOOLS or req.name in SEARCH_TOOLS):
            from .rate_limit import check_recall, check_write
            limited = check_write(request) if req.name in WRITE_TOOLS else check_recall(request)
            if limited is not None:
                return limited
        try:
            result = await call_tool(memory_system, req.name, req.arguments or {})
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            # The caller gets the message, and the log gets the traceback: a failing
            # store (an InvalidTag on every row, say) otherwise showed up in the
            # server logs as nothing but "200 OK".
            print(f"[{SERVER_NAME}] /mcp/call {req.name} failed: {e!r}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "result": result}

    return app
