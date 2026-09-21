"""MCP server over the official SDK (JSON-RPC 2.0 on stdio).

Every tool advertised in ``tools.py`` has a handler in ``HANDLERS`` below; the module
asserts the two agree at import time, so a tool can never be advertised without an
implementation behind it.

Note for anyone adding to this file: under stdio transport, stdout *is* the protocol
stream. Never print to it - use stderr.
"""

import json
import sys
import time
from collections import Counter
from typing import Any, Dict

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..core.tiers import Tier
from .tools import get_tools_schema

SERVER_NAME = "memcore-memory"
SERVER_VERSION = "1.0.0"


class ToolError(Exception):
    """Raised by a handler to return an error result to the client."""


def _item_dict(item, content_limit: int = None) -> Dict[str, Any]:
    content = item.content
    if content_limit and len(content) > content_limit:
        content = content[:content_limit] + "..."
    return {
        "id": item.id,
        "content": content,
        "tier": item.tier.value,
        "importance": item.metadata.get("importance", 0.5),
        "retention": round(item.forgetting.retention(), 4),
        "rehearsals": item.forgetting.rehearsals,
        "timestamp": item.timestamp,
        "entities": item.entities,
    }


def _tier(value: str) -> Tier:
    try:
        return Tier(value)
    except ValueError:
        raise ToolError(f"unknown tier {value!r}; valid tiers are {[t.value for t in Tier]}")


async def _require(memory, memory_id: str):
    item = await memory.store.get(memory_id)
    if not item:
        raise ToolError(f"no memory with id {memory_id!r}")
    return item


def _embed(memory, text: str):
    embedder = memory.embedder
    if hasattr(embedder, "embed_query"):
        return embedder.embed_query(text)
    return embedder.embed([text])[0]


# --- handlers -------------------------------------------------------------

async def _memory_add(memory, a):
    item = await memory.add(
        a["content"],
        metadata=a.get("metadata"),
        tier=a.get("tier"),
        importance=a.get("importance", 0.5),
        entities=a.get("entities"),
    )
    return _item_dict(item)


async def _memory_get(memory, a):
    item = await _require(memory, a["id"])
    item.touch()
    await memory.store.put(item)
    return _item_dict(item)


async def _memory_delete(memory, a):
    await _require(memory, a["id"])
    await memory.store.delete(a["id"])
    await memory.vectors.delete(a["id"])
    return {"deleted": a["id"]}


async def _memory_update(memory, a):
    item = await _require(memory, a["id"])
    if "content" not in a and "metadata" not in a:
        raise ToolError("supply content and/or metadata to update")
    if "content" in a:
        item.content = a["content"]
        item.embedding = _embed(memory, a["content"])
        await memory.vectors.delete(item.id)
        await memory.vectors.add(item.id, item.embedding,
                                 {"tier": item.tier.value, "content": item.content[:500]})
    if "metadata" in a:
        item.metadata.update(a["metadata"])
    await memory.store.put(item)
    return _item_dict(item)


async def _memory_recall(memory, a):
    results = await memory.recall(a["query"], k=a.get("k", 10), tier_filter=a.get("tier_filter"))
    return {"count": len(results), "results": results}


async def _memory_search_bm25(memory, a):
    results = await memory.retriever.bm25.retrieve(a["query"], a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_vector(memory, a):
    results = await memory.retriever.vector.retrieve(a["query"], a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_temporal(memory, a):
    results = await memory.retriever.temporal.retrieve(a["query"], a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_graph(memory, a):
    edges = await memory.kg.traverse(a["entity"], depth=a.get("depth", 2), limit=a.get("k", 10))
    return {"count": len(edges), "edges": edges}


async def _memory_list(memory, a):
    tier = a.get("tier")
    items = await (memory.store.list_by_tier(_tier(tier)) if tier else memory.store.list_all())
    limit = a.get("limit", 20)
    return {"total": len(items), "returned": min(limit, len(items)),
            "items": [_item_dict(i, 200) for i in items[:limit]]}


async def _memory_list_all(memory, a):
    items = await memory.store.list_all()
    limit = a.get("limit", 100)
    return {"total": len(items), "returned": min(limit, len(items)),
            "items": [_item_dict(i, 200) for i in items[:limit]]}


async def _retier(memory, a, direction):
    order = [Tier.SENSORY, Tier.WORKING, Tier.EPISODIC, Tier.SEMANTIC]
    item = await _require(memory, a["id"])
    target = _tier(a["target_tier"])
    current_i, target_i = order.index(item.tier), order.index(target)
    if direction == "promote" and target_i <= current_i:
        raise ToolError(f"{target.value} is not above {item.tier.value}; use memory_demote")
    if direction == "demote" and target_i >= current_i:
        raise ToolError(f"{target.value} is not below {item.tier.value}; use memory_promote")
    await memory.store.update_tier(item.id, target)
    return {"id": item.id, "from": item.tier.value, "to": target.value}


async def _memory_promote(memory, a):
    return await _retier(memory, a, "promote")


async def _memory_demote(memory, a):
    return await _retier(memory, a, "demote")


async def _memory_touch(memory, a):
    item = await _require(memory, a["id"])
    before = item.forgetting.retention()
    item.touch()
    await memory.store.put(item)
    return {"id": item.id, "rehearsals": item.forgetting.rehearsals,
            "strength": round(item.forgetting.strength, 4),
            "retention_before": round(before, 4),
            "retention_after": round(item.forgetting.retention(), 4)}


async def _memory_forget(memory, a):
    return {"forgotten": await memory.forget_expired()}


async def _memory_consolidate(memory, a):
    before = Counter(i.tier.value for i in await memory.store.list_all())
    await memory.consolidate()
    after = Counter(i.tier.value for i in await memory.store.list_all())
    return {"before": dict(before), "after": dict(after)}


async def _memory_stats(memory, a):
    items = await memory.store.list_all()
    per_tier = {}
    for tier in Tier:
        group = [i for i in items if i.tier == tier]
        per_tier[tier.value] = {
            "count": len(group),
            "avg_retention": round(sum(i.forgetting.retention() for i in group) / len(group), 4) if group else 0.0,
        }
    return {"total": len(items), "tiers": per_tier}


async def _memory_export(memory, a):
    items = await memory.store.list_all()
    payload = [{"content": i.content, "tier": i.tier.value, "metadata": i.metadata,
                "entities": i.entities, "timestamp": i.timestamp} for i in items]
    try:
        with open(a["path"], "w") as fh:
            json.dump(payload, fh, indent=2)
    except OSError as e:
        raise ToolError(f"cannot write {a['path']}: {e}")
    return {"exported": len(payload), "path": a["path"], "encrypted": False}


async def _memory_import(memory, a):
    try:
        with open(a["path"]) as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as e:
        raise ToolError(f"cannot read {a['path']}: {e}")
    if not isinstance(payload, list):
        raise ToolError("expected a JSON array of memory objects")
    imported = 0
    for entry in payload:
        if not isinstance(entry, dict) or "content" not in entry:
            continue
        await memory.add(entry["content"], metadata=entry.get("metadata"), tier=entry.get("tier"),
                         importance=(entry.get("metadata") or {}).get("importance", 0.5),
                         entities=entry.get("entities"))
        imported += 1
    return {"imported": imported, "skipped": len(payload) - imported}


async def _kg_add_entity(memory, a):
    node_id = await memory.kg.add_entity(a["entity"], type=a.get("type", "entity"), props=a.get("props"))
    return {"entity": node_id}


async def _kg_add_relation(memory, a):
    edge_id = await memory.kg.add_relation(a["src"], a["dst"],
                                           relation=a.get("relation", "related_to"),
                                           weight=a.get("weight", 1.0))
    return {"edge_id": edge_id, "src": a["src"].lower(), "dst": a["dst"].lower()}


async def _kg_traverse(memory, a):
    edges = await memory.kg.traverse(a["entity"], depth=a.get("depth", 2), limit=a.get("limit", 20))
    return {"count": len(edges), "edges": edges}


async def _kg_get_related(memory, a):
    entity = a["entity"].lower()
    edges = await memory.kg.traverse(entity, depth=1, limit=200)
    related = sorted({e["dst"] if e["src"] == entity else e["src"] for e in edges} - {entity})
    return {"entity": entity, "related": related}


async def _kg_list_entities(memory, a):
    entities = await memory.kg.list_entities(limit=a.get("limit", 100))
    return {"count": len(entities), "entities": entities}


async def _kg_delete_entity(memory, a):
    return await memory.kg.delete_entity(a["entity"])


async def _sync_status(memory, a):
    from ..config import settings
    return {"p2p_port": settings.p2p_port, "peers": list(settings.p2p_peers),
            "note": "configuration only; this stdio process does not run a P2P node. "
                    "Start one with `memcore server start`."}


async def _sync_peers(memory, a):
    from ..config import settings
    return {"count": len(settings.p2p_peers), "peers": list(settings.p2p_peers)}


_CONFIG_REDACT = ("key", "password", "secret", "token", "database_url")


async def _config_get(memory, a):
    from ..config import settings
    data = {}
    for field, value in settings.model_dump().items():
        if any(s in field.lower() for s in _CONFIG_REDACT):
            continue
        data[field] = str(value) if not isinstance(value, (int, float, bool, list, type(None))) else value
    key = a.get("key")
    if key:
        if key not in data:
            raise ToolError(f"no config key {key!r} (or it is redacted)")
        return {key: data[key]}
    return data


async def _health_check(memory, a):
    from ..config import settings
    items = await memory.store.list_all()
    return {"status": "ok", "version": SERVER_VERSION, "memories": len(items),
            "tier_counts": dict(Counter(i.tier.value for i in items)),
            "data_dir": str(settings.data_dir), "db_path": str(settings.db_path),
            "embedder": type(memory.embedder).__name__, "timestamp": time.time()}


HANDLERS = {
    "memory_add": _memory_add,
    "memory_get": _memory_get,
    "memory_delete": _memory_delete,
    "memory_update": _memory_update,
    "memory_recall": _memory_recall,
    "memory_search_bm25": _memory_search_bm25,
    "memory_search_vector": _memory_search_vector,
    "memory_search_temporal": _memory_search_temporal,
    "memory_search_graph": _memory_search_graph,
    "memory_list": _memory_list,
    "memory_list_all": _memory_list_all,
    "memory_promote": _memory_promote,
    "memory_demote": _memory_demote,
    "memory_touch": _memory_touch,
    "memory_forget": _memory_forget,
    "memory_consolidate": _memory_consolidate,
    "memory_stats": _memory_stats,
    "memory_export": _memory_export,
    "memory_import": _memory_import,
    "kg_add_entity": _kg_add_entity,
    "kg_add_relation": _kg_add_relation,
    "kg_traverse": _kg_traverse,
    "kg_get_related": _kg_get_related,
    "kg_list_entities": _kg_list_entities,
    "kg_delete_entity": _kg_delete_entity,
    "sync_status": _sync_status,
    "sync_peers": _sync_peers,
    "config_get": _config_get,
    "health_check": _health_check,
}

_advertised = {t["name"] for t in get_tools_schema()}
assert _advertised == set(HANDLERS), (
    f"tools.py and server.py disagree: "
    f"advertised-only={_advertised - set(HANDLERS)}, handler-only={set(HANDLERS) - _advertised}"
)


class MCPServer:
    """Wraps a MnemosyneMemory instance as an MCP server."""

    def __init__(self, memory_system):
        self.memory = memory_system
        self.tools = [
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in get_tools_schema()
        ]
        self.server = Server(
            SERVER_NAME,
            version=SERVER_VERSION,
            instructions="Persistent 4-tier memory with Ebbinghaus forgetting. Use memory_recall to "
                         "look things up and memory_add to store them. Memories decay unless rehearsed; "
                         "memory_touch or a higher tier keeps something durable.",
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    async def _on_list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools)

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch by tool name. Raises ToolError for anything the client got wrong."""
        handler = HANDLERS.get(name)
        if handler is None:
            raise ToolError(f"unknown tool {name!r}")
        return await handler(self.memory, args or {})

    async def _on_call_tool(self, ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            result = await self.handle_tool_call(params.name, params.arguments or {})
        except ToolError as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(e))], isError=True)
        except Exception as e:
            print(f"[{SERVER_NAME}] {params.name} failed: {e!r}", file=sys.stderr, flush=True)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")], isError=True)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            structuredContent=result if isinstance(result, dict) else {"result": result},
        )

    async def serve_stdio(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())
