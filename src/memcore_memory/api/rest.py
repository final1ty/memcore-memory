
from fastapi import FastAPI, HTTPException
from .schemas import MemoryAddRequest, RecallRequest, RecallResponse, MemoryAddResponse, HealthResponse, MCPCallRequest
from ..core.tiers import Tier
from ..mcp.server import HANDLERS, SERVER_NAME, SERVER_VERSION, ToolError
from ..mcp.tools import get_tools_schema
from collections import Counter

def create_app(memory_system):
    app = FastAPI(title="Mnemosyne Memory API", version="1.0.0", description="Local-first encrypted lifelong memory REST API")

    @app.post("/memory", response_model=MemoryAddResponse)
    async def add_memory(req: MemoryAddRequest):
        item = await memory_system.add(content=req.content, tier=req.tier, importance=req.importance, entities=req.entities, metadata=req.metadata)
        return MemoryAddResponse(id=item.id, tier=item.tier.value)

    @app.get("/memory/{mem_id}")
    async def get_memory(mem_id: str):
        item = await memory_system.store.get(mem_id)
        if not item:
            raise HTTPException(404, "Not found")
        return {"id": item.id, "content": item.content, "tier": item.tier.value, "metadata": item.metadata, "retention": item.forgetting.retention()}

    @app.post("/recall", response_model=RecallResponse)
    async def recall(req: RecallRequest):
        results = await memory_system.recall(req.query, k=req.k, tier_filter=req.tier_filter)
        return RecallResponse(results=results)

    @app.get("/memories")
    async def list_memories(tier: str = None, limit: int = 50):
        if tier:
            items = await memory_system.store.list_by_tier(Tier(tier))
        else:
            items = await memory_system.store.list_all()
        return [{"id": i.id, "content": i.content[:500], "tier": i.tier.value, "timestamp": i.timestamp} for i in items[:limit]]

    @app.delete("/memory/{mem_id}")
    async def delete_memory(mem_id: str):
        await memory_system.store.delete(mem_id)
        await memory_system.vectors.delete(mem_id)
        return {"deleted": mem_id}

    @app.post("/consolidate")
    async def consolidate():
        await memory_system.consolidate()
        return {"status": "consolidated"}

    @app.post("/forget")
    async def forget_expired():
        count = await memory_system.forget_expired()
        return {"forgotten": count}

    @app.get("/health", response_model=HealthResponse)
    async def health():
        all_items = await memory_system.store.list_all()
        c = Counter([i.tier.value for i in all_items])
        return HealthResponse(status="ok", version="1.0.0", tier_counts=dict(c))

    @app.post("/sync/merge")
    async def sync_merge(crdt_data: dict):
        # merge CRDT
        return {"status": "merged", "node": "local"}

    @app.get("/kg/traverse/{entity}")
    async def kg_traverse(entity: str, depth: int = 2, limit: int = 20):
        results = await memory_system.kg.traverse(entity, depth=depth, limit=limit)
        return {"entity": entity, "results": results}

    # --- MCP bridge -------------------------------------------------------
    # These two routes let an MCP client reach this instance's store without
    # opening the database itself. See mcp/remote.py for why that matters.

    @app.get("/mcp/tools")
    async def mcp_tools():
        return {"server": SERVER_NAME, "version": SERVER_VERSION, "tools": get_tools_schema()}

    @app.post("/mcp/call")
    async def mcp_call(req: MCPCallRequest):
        """Dispatch through the same handlers the stdio server uses.

        Tool-level failures come back as ``ok: false`` rather than an HTTP error, so
        the bridge can tell "you called this wrong" apart from "the server is down".
        """
        handler = HANDLERS.get(req.name)
        if handler is None:
            return {"ok": False, "error": f"unknown tool {req.name!r}"}
        try:
            result = await handler(memory_system, req.arguments or {})
        except ToolError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "result": result}

    return app
