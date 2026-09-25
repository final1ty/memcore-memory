"""MCP tool definitions.

Each entry carries a real JSON Schema under ``inputSchema`` as the MCP spec requires.
``server.py`` validates every call against it before dispatch, on stdio and on the
REST bridge alike - clients are free to validate too, but nothing relies on it. It
also owns a handler for every name listed here - see ``HANDLERS`` there; the two are
checked against each other at import time.
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
# A file name inside <data_dir>/exports; see _exchange_path in server.py.
_EXCHANGE_PATH = {"type": "string", "minLength": 1,
                  "description": "A file name such as 'backup.json'. Always resolved inside "
                                 "<data_dir>/exports on the machine running the store; "
                                 "directories and paths outside it are refused."}

TOOLS = [
    {
        "name": "memory_add",
        "description": "Store a new memory. Unless `tier` is given it goes to working, which holds at most "
                       "working_capacity memories (default 7); the oldest overflow moves to episodic. "
                       "Importance does not choose the starting tier.",
        "inputSchema": _obj({
            "content": {"type": "string", "description": "The text to remember"},
            "tier": dict(_TIER, description="Force a tier instead of letting the TierManager assign one"),
            "importance": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.5,
                           "description": "Scales Ebbinghaus decay strength (higher = slower forgetting) and must "
                                          "exceed semantic_min_importance for episodic->semantic promotion"},
            "entities": {"type": "array", "items": {"type": "string"},
                         "description": "Named entities to index in the knowledge graph"},
            "metadata": {"type": "object", "description": "Arbitrary JSON stored alongside the memory"},
        }, ["content"]),
    },
    {
        "name": "memory_get",
        "description": "Fetch one memory by id. By default this counts as a rehearsal and boosts its "
                       "retention; pass touch: false to read it without rehearsing.",
        "inputSchema": _obj({
            "id": _ID,
            "touch": {"type": "boolean", "default": True,
                      "description": "Count the read as a rehearsal (GET /memory/{id} defaults to false)"},
        }, ["id"]),
    },
    {
        "name": "memory_delete",
        "description": "Permanently delete a memory, its vector and its knowledge-graph links. Not reversible. "
                       "Errors if no memory has that id.",
        "inputSchema": _obj({"id": _ID}, ["id"]),
    },
    {
        "name": "memory_update",
        "description": "Replace the content and/or metadata of an existing memory. Re-embeds when content changes; "
                       "metadata.importance also updates the decay curve.",
        "inputSchema": _obj({
            "id": _ID,
            "content": {"type": "string"},
            "metadata": {"type": "object", "description": "Merged into existing metadata"},
        }, ["id"]),
    },
    {
        "name": "memory_recall",
        "description": "Primary search. 6-way hybrid retrieval (vector, BM25, graph, temporal, importance, metadata) "
                       "fused with RRF. Counts as a rehearsal for the top 3 matched results (boosts their "
                       "retention; only the best hit adds to the rehearsal count) unless rehearse is false.",
        "inputSchema": _obj({
            "query": {"type": "string"},
            "k": _K,
            "tier_filter": {"type": "array", "items": _TIER, "description": "Restrict results to these tiers"},
            "rehearse": {"type": "boolean", "default": True,
                         "description": "Set false to search without strengthening the results"},
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
        "description": "List memories ranked by recency and Ebbinghaus retention. Query-independent: query is "
                       "ignored; use memory_recall for a recency-aware search.",
        "inputSchema": _obj({"query": {"type": "string", "description": "Ignored; accepted for compatibility"},
                             "k": _K}),
    },
    {
        "name": "memory_search_graph",
        "description": "Find memories through the knowledge graph: memories tagged with exactly the entity "
                       "first, then memories tagged with an entity whose name appears inside it ('docker' for "
                       "'Docker Compose'), then memories tagged with entities reachable within `depth` hops. "
                       "Returns memories; use kg_traverse for the edges themselves.",
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
        "description": "Move a memory to a shorter-lived tier (working or episodic). Demotion shortens its "
                       "lifetime: episodic memories are deleted once retention drops below 0.05. Sensory is "
                       "not a target - anything there is deleted by the next memory_forget.",
        # Sensory is left out on purpose: it is a 30-second ingest buffer, and a stored
        # memory moved there was deleted by the next forgetting pass however well retained.
        "inputSchema": _obj({"id": _ID, "target_tier": {"type": "string", "enum": ["working", "episodic"]}},
                            ["id", "target_tier"]),
    },
    {
        "name": "memory_touch",
        "description": "Rehearse a memory: S = S*1.6 + 0.5, resetting its decay clock. Use to keep something from being forgotten.",
        "inputSchema": _obj({"id": _ID}, ["id"]),
    },
    {
        "name": "memory_forget",
        "description": "Run one lifecycle pass: promote what qualifies, move working memories past their TTL to "
                       "episodic (working memories are demoted, never deleted), and delete expired sensory "
                       "memories and episodic ones below retention 0.05. Semantic memories are never "
                       "auto-deleted. Returns forgotten/demoted/promoted counts and any rows it had to skip.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_consolidate",
        "description": "Promote episodic memories into the semantic tier when they have at least "
                       "semantic_consolidation_threshold rehearsals (default 3), retention above 0.6 and "
                       "importance above semantic_min_importance (default 0.8). Deletes nothing.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_stats",
        "description": "Per-tier counts and average retention, plus how many stored rows could not be read.",
        "inputSchema": _obj({}),
    },
    {
        "name": "memory_export",
        "description": "Write all memories (ids, timestamps and forgetting curves included) to a plaintext "
                       "JSON file in <data_dir>/exports, mode 0600. The export is NOT encrypted. An existing "
                       "file is never replaced unless overwrite is true. Rows that cannot be decrypted are "
                       "left out and listed under `skipped`, with partial: true.",
        "inputSchema": _obj({
            "path": _EXCHANGE_PATH,
            "overwrite": {"type": "boolean", "default": False,
                          "description": "Replace the file if it already exists"},
        }, ["path"]),
    },
    {
        "name": "memory_import",
        "description": "Load memories from a file in <data_dir>/exports produced by memory_export. Ids, "
                       "timestamps and rehearsal history are restored; memories already present are "
                       "skipped, so re-importing is safe. The whole file is checked first, and the import "
                       "is all or nothing: if any entry is invalid or a write fails, what this call wrote is "
                       "removed again.",
        "inputSchema": _obj({"path": _EXCHANGE_PATH}, ["path"]),
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
        "description": "Store location, memory count, unreadable rows and the embedder actually loaded. "
                       "status is 'degraded' (with reasons) when rows are unreadable or the knowledge graph "
                       "could not be opened.",
        "inputSchema": _obj({}),
    },
]


# Tools that change the store (or, for export, the filesystem). The REST bridge
# charges these to the per-IP write limit; reads and rehearsals are left alone,
# because every tool call from a bridged session arrives as POST /mcp/call.
WRITE_TOOLS = frozenset({
    "memory_add", "memory_delete", "memory_update", "memory_promote", "memory_demote",
    "memory_forget", "memory_consolidate", "memory_export", "memory_import",
    "kg_add_entity", "kg_add_relation", "kg_delete_entity",
})
assert WRITE_TOOLS <= {t["name"] for t in TOOLS}

# Searches. The bridge charges these to the per-IP recall limit, the same bucket
# POST /recall uses, so switching routes does not get around it.
SEARCH_TOOLS = frozenset({
    "memory_recall", "memory_search_bm25", "memory_search_vector",
    "memory_search_temporal", "memory_search_graph",
})
assert SEARCH_TOOLS <= {t["name"] for t in TOOLS}


def get_tools_schema():
    return TOOLS
