"""Tests for the MCP layer.

The previous implementation spoke a homemade ``{"tool": ..., "args": ...}`` line
protocol that no MCP client can talk to, and fell through to
``{"status": "executed"}`` for any tool it did not implement - so 22 of the 33
advertised tools reported success while doing nothing. These tests pin down both:
the advertised surface must match the implemented one, and failures must surface
as errors rather than cheerful no-ops.
"""

import pytest

from memcore_memory.config import settings
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.mcp.server import MCPServer, HANDLERS, ToolError
from memcore_memory.mcp.tools import get_tools_schema


@pytest.fixture
async def server(isolate_data_dir):
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    mem = MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)
    return MCPServer(mem)


def test_every_advertised_tool_has_a_handler():
    advertised = {t["name"] for t in get_tools_schema()}
    assert advertised == set(HANDLERS)


def test_tool_schemas_are_valid_json_schema():
    for tool in get_tools_schema():
        schema = tool["inputSchema"]
        assert schema["type"] == "object"
        assert isinstance(schema["properties"], dict)
        assert tool["description"], f"{tool['name']} has no description"
        for required in schema.get("required", []):
            assert required in schema["properties"], f"{tool['name']}: {required} not declared"


async def test_unknown_tool_raises_rather_than_reporting_success(server):
    """The old fallback returned {"status": "executed"} for anything unrecognised."""
    with pytest.raises(ToolError):
        await server.handle_tool_call("definitely_not_a_tool", {})


async def test_missing_memory_raises(server):
    with pytest.raises(ToolError):
        await server.handle_tool_call("memory_get", {"id": "no-such-id"})


async def test_add_recall_roundtrip(server):
    added = await server.handle_tool_call(
        "memory_add", {"content": "SkyNAS runs Docker", "tier": "semantic",
                       "importance": 0.9, "entities": ["SkyNAS", "Docker"]})
    assert added["tier"] == "semantic"
    found = await server.handle_tool_call("memory_recall", {"query": "Docker", "k": 5})
    assert added["id"] in [r["id"] for r in found["results"]]


async def test_touch_increases_retention_strength(server):
    added = await server.handle_tool_call("memory_add", {"content": "rehearse me", "tier": "working"})
    first = await server.handle_tool_call("memory_touch", {"id": added["id"]})
    second = await server.handle_tool_call("memory_touch", {"id": added["id"]})
    assert second["strength"] > first["strength"]
    assert second["rehearsals"] > first["rehearsals"]


async def test_promote_rejects_a_downward_move(server):
    added = await server.handle_tool_call("memory_add", {"content": "x", "tier": "semantic"})
    with pytest.raises(ToolError):
        await server.handle_tool_call("memory_promote", {"id": added["id"], "target_tier": "working"})


async def test_delete_actually_removes(server):
    added = await server.handle_tool_call("memory_add", {"content": "temporary"})
    await server.handle_tool_call("memory_delete", {"id": added["id"]})
    with pytest.raises(ToolError):
        await server.handle_tool_call("memory_get", {"id": added["id"]})


async def test_config_get_redacts_secrets(server):
    config = await server.handle_tool_call("config_get", {})
    for field in config:
        assert "password" not in field.lower()
        assert "key_path" not in field.lower()
        assert field != "database_url"


async def test_kg_relation_and_delete(server):
    await server.handle_tool_call("kg_add_entity", {"entity": "Ubuntu"})
    await server.handle_tool_call("kg_add_relation", {"src": "SkyNAS", "dst": "Ubuntu", "relation": "runs"})
    related = await server.handle_tool_call("kg_get_related", {"entity": "SkyNAS"})
    assert "ubuntu" in related["related"]
    removed = await server.handle_tool_call("kg_delete_entity", {"entity": "Ubuntu"})
    assert removed["edges_deleted"] >= 1
