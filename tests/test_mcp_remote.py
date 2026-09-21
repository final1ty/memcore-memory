"""Tests for the REST-backed MCP bridge.

The bug these pin down: ``.mcp.json`` originally pointed a local stdio MCP server at
``/home/skynas/.memcore`` while the live data sat inside the container's writable
layer. Both stores exist, both open cleanly, and they answer differently - so the
client looked healthy while serving memories nobody had written. The bridge exists
so there is only ever one store behind the tools.
"""

import httpx
import pytest

from memcore_memory.api.rest import create_app
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.mcp.remote import RemoteMCPServer, RemoteUnavailable
from memcore_memory.mcp.server import HANDLERS, ToolError
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore


@pytest.fixture
async def bridge(isolate_data_dir):
    """A RemoteMCPServer wired to an in-process app over ASGI - no socket, real routing."""
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    mem = MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)

    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=create_app(mem)), base_url="http://memcore.test"
    )
    server = await RemoteMCPServer("http://memcore.test", client=client).connect()
    yield server
    await client.aclose()


async def test_bridge_advertises_the_full_local_tool_surface(bridge):
    """A proxy that quietly drops tools would be the same class of bug all over again."""
    assert {t.name for t in bridge.tools} == set(HANDLERS)


async def test_add_and_recall_through_the_bridge(bridge):
    added = await bridge.handle_tool_call(
        "memory_add", {"content": "SkyNAS runs Docker", "tier": "semantic", "importance": 0.9}
    )
    found = await bridge.handle_tool_call("memory_recall", {"query": "Docker", "k": 5})
    assert added["id"] in [r["id"] for r in found["results"]]


async def test_remote_tool_errors_arrive_as_tool_errors(bridge):
    """Not as HTTP noise, and above all not as a cheerful success."""
    with pytest.raises(ToolError):
        await bridge.handle_tool_call("memory_get", {"id": "no-such-id"})


async def test_unknown_tool_raises(bridge):
    with pytest.raises(ToolError):
        await bridge.handle_tool_call("definitely_not_a_tool", {})


async def test_bridge_writes_land_in_the_store_behind_the_api(bridge):
    """The whole point: the bridge must not have a store of its own."""
    added = await bridge.handle_tool_call("memory_add", {"content": "only one store"})
    listed = await bridge.handle_tool_call("memory_list_all", {})
    assert added["id"] in [m["id"] for m in listed["items"]]


async def test_old_instance_without_the_bridge_is_diagnosed(isolate_data_dir):
    """An instance built before these routes existed must say so, not fail obscurely."""

    async def handler(request):
        return httpx.Response(404, text="Not Found")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://old.test")
    with pytest.raises(RemoteUnavailable, match="predates"):
        await RemoteMCPServer("http://old.test", client=client).connect()
    await client.aclose()


async def test_unreachable_endpoint_is_diagnosed(isolate_data_dir):
    async def handler(request):
        raise httpx.ConnectError("connection refused")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://down.test")
    with pytest.raises(RemoteUnavailable, match="cannot reach"):
        await RemoteMCPServer("http://down.test", client=client).connect()
    await client.aclose()
