"""MCP tool definitions.

Each entry carries a real JSON Schema under ``inputSchema`` as the MCP spec requires,
so clients can validate arguments before dispatch. ``server.py`` owns a handler for
every name listed here - see ``HANDLERS`` there; the two are checked against each
other at import time.
"""

TIERS = ["sensory", "working", "episodic", "semantic"]


def _obj(properties: dict, required: list = None) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


_ID = {"type": "string", "description": "Memory UUID"}
_K = {"type": "integer", "minimum": 1, "maximum": 100, "default": 10, "description": "Max results"}
_TIER = {"type": "string", "enum": TIERS}

TOOLS = [
    {
        "name": "memory_add",
        "description": "Store a new memory. Tier is inferred from importance and rehearsal count unless given explicitly.",
        "inputSchema": _obj({
            "content": {"type": "string", "description": "The text to remember"},
            "tier": dict(_TIER, description="Force a tier instead of letting the TierManager assign one"),
            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5,
                           "description": "Drives both tier assignment and Ebbinghaus decay strength"},
            "entities": {"type": "array", "items": {"type": "string"},
                         "description": "Named entities to index in the knowledge graph"},
            "metadata": {"type": "object", "description": "Arbitrary JSON stored alongside the memory"},
        }, ["content"]),
    },
    {
        "name": "memory_get",
        "description": "Fetch one memory by id. This counts as a rehearsal and boosts its retention.",
        "inputSchema": _obj({"id": _ID}, ["id"]),
    },
    {
        "name": "memory_delete",
        "description": "Permanently delete a memory and its vector. Not reversible.",
        "inputSchema": _obj({"id": _ID}, ["id"]),
    },
    {
        "name": "memory_update",
        "description": "Replace the content and/or metadata of an existing memory. Re-embeds when content changes.",
        "inputSchema": _obj({
            "id": _ID,
            "content": {"type": "string"},
            "metadata": {"type": "object", "description": "Merged into existing metadata"},
        }, ["id"]),
    },
    {
        "name": "memory_recall",
        "description": "Primary search. 6-way hybrid retrieval (vector, BM25, graph, temporal, importance, metadata) fused with RRF.",
        "inputSchema": _obj({
            "query": {"type": "string"},
            "k": _K,
            "tier_filter": {"type": "array", "items": _TIER, "description": "Restrict results to these tiers"},
        }, ["query"]),
    },
    {
        "name": "memory_search_bm25",
        "description": "Lexical BM25 search only. Use memory_recall unless you specifically need keyword matching.",
        "inputSchema": _obj({"query": {"type": "string"}, "k": _K}, ["query"]),
    },
    {
        "name": "memory_search_vector",
        "description": "Dense vector similarity only. Use memory_recall unless you specifically need semantic-only matching.",
        "inputSchema": _obj({"query": {"type": "string"}, "k": _K}, ["query"]),
    },
    {
        "name": "memory_search_temporal",
        "description": "Recency-weighted search using the Ebbinghaus retention curve.",
        "inputSchema": _obj({"query": {"type": "string"}, "k": _K}, ["query"]),
    },
    {
        "name": "memory_search_graph",
        "description": "Find memories via knowledge-graph traversal from an entity.",
        "inputSchema": _obj({
            "entity": {"type": "string"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            "k": _K,
        }, ["entity"]),
    },
    {
        "name": "memory_list",
        "description": "List stored memories, optionally filtered to one tier.",
        "inputSchema": _obj({
            "tier": _TIER,
            "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 20},
        }),
    },
    {
        "name": "memory_list_all",
        "description": "List memories across every tier.",
        "inputSchema": _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}),
    },
    {
        "name": "memory_promote",
        "description": "Move a memory to a longer-lived tier (sensory -> working -> episodic -> semantic).",
        "inputSchema": _obj({"id": _ID, "target_tier": _TIER}, ["id", "target_tier"]),
    },
    {
        "name": "memory_demote",
        "description": "Move a memory to a shorter-lived tier.",
        "inputSchema": _obj({"id": _ID, "target_tier": _TIER}, ["id", "target_tier"]),
    },
    {
        "name": "memory_touch",
        "description": "Rehearse a memory: S = S*1.6 + 0.5, resetting its decay clock. Use to keep something from being forgotten.",
        "inputSchema": _obj({"id": _ID}, ["id"]),
    },
    {
        "name": "memory_forget",
        "description": "Run one forgetting pass: promote what qualifies, delete anything below its retention threshold.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_consolidate",
        "description": "Promote well-rehearsed episodic memories into the semantic tier.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_stats",
        "description": "Per-tier counts and average retention.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_export",
        "description": "Write all memories to a plaintext JSON file. The export is NOT encrypted - treat the path as sensitive.",
        "inputSchema": _obj({"path": {"type": "string", "description": "Destination file path"}}, ["path"]),
    },
    {
        "name": "memory_import",
        "description": "Load memories from a JSON file produced by memory_export.",
        "inputSchema": _obj({"path": {"type": "string"}}, ["path"]),
    },
    {
        "name": "kg_add_entity",
        "description": "Add or replace a knowledge-graph entity node.",
        "inputSchema": _obj({
            "entity": {"type": "string"},
            "type": {"type": "string", "default": "entity"},
            "props": {"type": "object"},
        }, ["entity"]),
    },
    {
        "name": "kg_add_relation",
        "description": "Add a directed edge between two entities, creating either node if missing.",
        "inputSchema": _obj({
            "src": {"type": "string"},
            "dst": {"type": "string"},
            "relation": {"type": "string", "default": "related_to"},
            "weight": {"type": "number", "default": 1.0},
        }, ["src", "dst"]),
    },
    {
        "name": "kg_traverse",
        "description": "Breadth-first walk of the knowledge graph from an entity, returning edges.",
        "inputSchema": _obj({
            "entity": {"type": "string"},
            "depth": {"type": "integer", "minimum": 1, "maximum": 5, "default": 2},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
        }, ["entity"]),
    },
    {
        "name": "kg_get_related",
        "description": "List entities directly connected to the given one.",
        "inputSchema": _obj({"entity": {"type": "string"}}, ["entity"]),
    },
    {
        "name": "kg_list_entities",
        "description": "List knowledge-graph entity nodes.",
        "inputSchema": _obj({"limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100}}),
    },
    {
        "name": "kg_delete_entity",
        "description": "Delete an entity node and every edge touching it.",
        "inputSchema": _obj({"entity": {"type": "string"}}, ["entity"]),
    },
    {
        "name": "sync_status",
        "description": "P2P sync configuration for this process.",
        "inputSchema": _obj({}),
    },
    {
        "name": "sync_peers",
        "description": "List configured P2P peers.",
        "inputSchema": _obj({}),
    },
    {
        "name": "config_get",
        "description": "Read the effective configuration. Secrets and key material are never included.",
        "inputSchema": _obj({"key": {"type": "string", "description": "Omit to return everything"}}),
    },
    {
        "name": "health_check",
        "description": "Liveness plus store location and memory count.",
        "inputSchema": _obj({}),
    },
]


def get_tools_schema():
    return TOOLS
