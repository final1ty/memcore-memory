"""Audit group E: the MCP tools and the REST API.

Each test names the finding it pins down. All of them run against the isolated
data dir from conftest; none touches a real store or the network.
"""

import json
import os
import time

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from memcore_memory.api import rate_limit
from memcore_memory.api.rest import create_app
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.mcp.remote import RemoteMCPServer, RemoteUnavailable
from memcore_memory.mcp.server import HANDLERS, SERVER_NAME, ToolError, call_tool
from memcore_memory.mcp.tools import get_tools_schema
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore


@pytest.fixture
async def mem(isolate_data_dir):
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    return MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)


@pytest.fixture
def client(mem):
    rate_limit.write_limiter.buckets.clear()
    rate_limit.recall_limiter.buckets.clear()
    with TestClient(create_app(mem)) as c:
        yield c


def exports():
    return settings.data_dir / "exports"


def mcp(client, name, **arguments):
    return client.post("/mcp/call", json={"name": name, "arguments": arguments})


# --- F2: export/import confined to <data_dir>/exports -----------------------

async def test_export_cannot_overwrite_the_master_key(mem):
    await call_tool(mem, "memory_add", {"content": "secret", "tier": "episodic"})
    before = settings.key_path.read_bytes()
    for path in (str(settings.key_path), "../master.key", str(settings.db_path), "sub/x.json", "..", "."):
        with pytest.raises(ToolError):
            await call_tool(mem, "memory_export", {"path": path})
    assert settings.key_path.read_bytes() == before
    # And the store still opens in a new process's worth of objects.
    assert KeyManager(settings.key_path).load_or_create() is not None


async def test_export_does_not_follow_a_symlink_out_of_the_exports_dir(mem):
    exports().mkdir(parents=True, exist_ok=True)
    os.symlink(settings.key_path, exports() / "innocent.json")
    before = settings.key_path.read_bytes()
    with pytest.raises(ToolError):
        await call_tool(mem, "memory_export", {"path": "innocent.json", "overwrite": True})
    assert settings.key_path.read_bytes() == before


async def test_export_is_private_and_never_clobbers_by_default(mem):
    await call_tool(mem, "memory_add", {"content": "one", "tier": "episodic"})
    out = await call_tool(mem, "memory_export", {"path": "b.json"})
    target = exports() / "b.json"
    assert out["path"] == str(target.resolve())
    assert target.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ToolError, match="already exists"):
        await call_tool(mem, "memory_export", {"path": "b.json"})
    await call_tool(mem, "memory_export", {"path": "b.json", "overwrite": True})
    assert target.stat().st_mode & 0o777 == 0o600
    # The absolute form of a path inside the exports dir is fine too.
    await call_tool(mem, "memory_export", {"path": str(exports() / "c.json")})


async def test_import_cannot_read_outside_the_exports_dir(mem):
    for path in ("/etc/hostname", str(settings.key_path), "../memory.db"):
        with pytest.raises(ToolError, match="only use files directly inside|bare file name"):
            await call_tool(mem, "memory_import", {"path": path})


def test_export_over_rest_cannot_reach_the_key(client):
    before = settings.key_path.read_bytes()
    r = mcp(client, "memory_export", path=str(settings.key_path))
    assert r.status_code == 200 and r.json()["ok"] is False
    assert settings.key_path.read_bytes() == before
    assert client.get("/health").status_code == 200


# --- F25 / R10: lossless, atomic, idempotent import -------------------------

async def test_export_import_round_trip_keeps_ids_curves_and_timestamps(mem):
    a = await call_tool(mem, "memory_add", {"content": "alpha fact", "tier": "semantic",
                                            "importance": 0.9, "entities": ["Alpha", "Beta"]})
    b = await call_tool(mem, "memory_add", {"content": "beta fact", "tier": "episodic"})
    for _ in range(3):
        await call_tool(mem, "memory_touch", {"id": a["id"]})
    orig = {i.id: i for i in await mem.store.list_all()}
    await call_tool(mem, "memory_export", {"path": "full.json"})
    data = json.loads((exports() / "full.json").read_text())
    assert data["format_version"] == 2 and {m["id"] for m in data["memories"]} == set(orig)

    for mid in orig:
        await call_tool(mem, "memory_delete", {"id": mid})
    out = await call_tool(mem, "memory_import", {"path": "full.json"})
    assert out["imported"] == 2 and out["skipped_existing"] == 0

    back = {i.id: i for i in await mem.store.list_all()}
    assert set(back) == set(orig)
    for mid, item in orig.items():
        assert back[mid].timestamp == item.timestamp
        assert back[mid].tier == item.tier
        assert back[mid].forgetting.rehearsals == item.forgetting.rehearsals
        assert back[mid].forgetting.strength == pytest.approx(item.forgetting.strength)
        assert mid in mem.vectors.ids
    assert back[a["id"]].entities == ["Alpha", "Beta"]
    assert b["id"] in back

    again = await call_tool(mem, "memory_import", {"path": "full.json"})
    assert again == {"imported": 0, "skipped_existing": 2, "total": 2}
    assert len(await mem.store.list_all()) == 2


async def test_bad_entry_imports_nothing_and_a_fixed_retry_does_not_duplicate(mem):
    exports().mkdir(parents=True, exist_ok=True)
    rows = [{"content": "import one"}, {"content": "import two", "tier": "longterm"},
            {"content": "import three"}]
    (exports() / "bad.json").write_text(json.dumps(rows))
    with pytest.raises(ToolError, match=r"nothing imported.*\[1\]"):
        await call_tool(mem, "memory_import", {"path": "bad.json"})
    assert await mem.store.list_all() == []

    rows[1]["tier"] = "semantic"
    (exports() / "bad.json").write_text(json.dumps(rows))
    assert (await call_tool(mem, "memory_import", {"path": "bad.json"}))["imported"] == 3
    # The legacy format has no ids, so a second run is matched on content.
    assert (await call_tool(mem, "memory_import", {"path": "bad.json"}))["imported"] == 0
    assert sorted(i.content for i in await mem.store.list_all()) == ["import one", "import three", "import two"]


# --- F9: arguments are validated against inputSchema ------------------------

async def test_string_importance_is_refused_before_anything_is_stored(mem):
    await call_tool(mem, "memory_add", {"content": "ok row"})
    for bad in ({"content": "p", "tier": "episodic", "importance": "high"},
                {"content": "p", "tier": "episodic", "importance": None},
                {"content": "p", "importance": 7},
                {"content": "p", "bogus_param": 1},
                {"content": 42}):
        with pytest.raises(ToolError, match="invalid arguments"):
            await call_tool(mem, "memory_add", bad)
    assert len(await mem.store.list_all()) == 1
    assert (await call_tool(mem, "memory_recall", {"query": "ok row"}))["count"] >= 1


async def test_integral_float_is_accepted_as_an_integer(mem):
    await call_tool(mem, "memory_add", {"content": "count me"})
    assert (await call_tool(mem, "memory_list", {"limit": 5.0}))["returned"] == 1


def test_validation_applies_over_rest_too(client):
    r = mcp(client, "memory_add", content="p", tier="episodic", importance="high").json()
    assert r["ok"] is False and "invalid arguments" in r["error"]
    assert client.get("/memories").json() == []


async def test_update_importance_is_validated_and_reaches_the_curve(mem):
    added = await call_tool(mem, "memory_add", {"content": "x", "tier": "episodic"})
    with pytest.raises(ToolError):
        await call_tool(mem, "memory_update", {"id": added["id"], "metadata": {"importance": "high"}})
    await call_tool(mem, "memory_update", {"id": added["id"], "metadata": {"importance": 0.9}})
    item = await mem.store.get(added["id"])
    assert item.metadata["importance"] == 0.9 and item.forgetting.importance == 0.9


# --- R9: a failed update changes nothing ------------------------------------

async def test_failed_update_leaves_row_and_sidecar_alone(mem):
    added = await HANDLERS["memory_add"](mem, {"content": "Original text about Grafana dashboards"})
    pos = mem.vectors.ids.index(added["id"])
    vec_before = np.array(mem.vectors.vectors[pos]).copy()
    with pytest.raises(ToolError):
        await HANDLERS["memory_update"](mem, {"id": added["id"], "content": "Replaced text about Prometheus",
                                              "metadata": ["a", "b", "c"]})
    row = await mem.store.get(added["id"])
    assert row.content == "Original text about Grafana dashboards"
    fresh = VectorStore(settings.vector_path, dim=settings.embedding_dim)
    assert np.allclose(fresh.vectors[fresh.ids.index(added["id"])], vec_before)


# --- F26 / F43: deletes go through MnemosyneMemory.delete -------------------

async def test_delete_of_unknown_id_is_an_error(mem):
    with pytest.raises(ToolError, match="no memory"):
        await call_tool(mem, "memory_delete", {"id": "no-such-id"})


def test_rest_delete_of_unknown_id_is_404(client):
    assert client.delete("/memory/this-id-does-not-exist").status_code == 404
    added = client.post("/memory", json={"content": "temporary"}).json()
    assert client.delete(f"/memory/{added['id']}").status_code == 200
    assert client.get(f"/memory/{added['id']}").status_code == 404
    assert client.delete(f"/memory/{added['id']}").status_code == 404


async def test_delete_removes_vector_and_graph_links(mem):
    added = await call_tool(mem, "memory_add", {"content": "zeta and eta", "tier": "episodic",
                                                "entities": ["Zeta", "Eta"]})
    assert (await call_tool(mem, "kg_traverse", {"entity": "zeta"}))["count"] >= 1
    await call_tool(mem, "memory_delete", {"id": added["id"]})
    assert added["id"] not in mem.vectors.ids
    assert (await call_tool(mem, "kg_traverse", {"entity": "zeta"}))["count"] == 0


# --- R20: memory_search_graph returns memories ------------------------------

async def test_search_graph_returns_memories_not_edges(mem):
    a = await call_tool(mem, "memory_add", {"content": "SkyNAS runs Docker", "tier": "episodic",
                                            "entities": ["SkyNAS", "Docker"]})
    b = await call_tool(mem, "memory_add", {"content": "Docker hosts Jellyfin", "tier": "episodic",
                                            "entities": ["Docker", "Jellyfin"]})
    out = await call_tool(mem, "memory_search_graph", {"entity": "SkyNAS", "depth": 2})
    ids = [r["id"] for r in out["results"]]
    assert ids[0] == a["id"]
    assert b["id"] in ids
    assert "content" in out["results"][0]


# --- F44: demoting to sensory is refused ------------------------------------

async def test_demote_to_sensory_is_refused_and_demote_then_forget_keeps_the_memory(mem):
    added = await call_tool(mem, "memory_add", {"content": "keep me", "tier": "semantic", "importance": 0.9})
    with pytest.raises(ToolError):
        await call_tool(mem, "memory_demote", {"id": added["id"], "target_tier": "sensory"})
    with pytest.raises(ToolError, match="sensory"):
        await HANDLERS["memory_demote"](mem, {"id": added["id"], "target_tier": "sensory"})
    await call_tool(mem, "memory_demote", {"id": added["id"], "target_tier": "episodic"})
    await call_tool(mem, "memory_forget", {})
    assert await mem.store.get(added["id"]) is not None


# --- F30: opt-in API key ----------------------------------------------------

def test_api_key_is_enforced_only_when_set(mem, monkeypatch):
    with TestClient(create_app(mem)) as open_client:
        assert open_client.get("/memories").status_code == 200

    monkeypatch.setattr(settings, "api_key", "s3cret")
    rate_limit.write_limiter.buckets.clear()
    with TestClient(create_app(mem)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/livez").json() == {"status": "ok"}
        assert c.get("/memories").status_code == 401
        assert c.get("/memories", headers={"Authorization": "Bearer nonsense"}).status_code == 401
        assert c.post("/mcp/call", json={"name": "health_check"}).status_code == 401
        assert c.get("/mcp/tools").status_code == 401
        assert c.get("/memories", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/memories", headers={"X-API-Key": "s3cret"}).status_code == 200
        # Refused requests are not charged to the write bucket.
        for _ in range(25):
            assert c.post("/memory", json={"content": "x"}).status_code == 401
        assert c.post("/memory", json={"content": "x"}, headers={"X-API-Key": "s3cret"}).status_code == 200


async def test_bridge_sends_the_api_key(mem, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "s3cret")
    app = create_app(mem)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://m.test") as c:
        server = await RemoteMCPServer("http://m.test", client=c).connect()
        assert (await server.handle_tool_call("health_check", {}))["status"] == "ok"
    async with httpx.AsyncClient(transport=transport, base_url="http://m.test") as c:
        with pytest.raises(RemoteUnavailable, match="API key"):
            await RemoteMCPServer("http://m.test", client=c, api_key="wrong").connect()


# --- F63: every write is rate limited ---------------------------------------

def _fill(limiter, key, n):
    for _ in range(n):
        limiter.is_allowed(key)


def test_consolidate_forget_and_mcp_writes_count_as_writes(client):
    _fill(rate_limit.write_limiter, "testclient:write", 20)
    for path in ("/consolidate", "/forget"):
        r = client.post(path)
        assert r.status_code == 429 and r.headers.get("Retry-After")
    assert mcp(client, "memory_add", content="x").status_code == 429
    # Reads through the bridge are not writes, or every bridged session would throttle.
    r = mcp(client, "memory_list")
    assert r.status_code == 200 and r.json()["ok"] is True


def test_reads_leave_the_write_bucket_untouched(client):
    _fill(rate_limit.write_limiter, "testclient:write", 20)
    assert client.get("/memories").status_code == 200
    assert len(rate_limit.write_limiter.buckets["testclient:write"]) == 20


def test_recall_limiter_applies(client):
    _fill(rate_limit.recall_limiter, "testclient:recall", 100)
    assert client.post("/recall", json={"query": "x"}).status_code == 429


# --- F81: bounded REST inputs -----------------------------------------------

def test_rest_rejects_out_of_range_input_with_422(client):
    assert client.post("/memory", json={"content": "x", "tier": "bogus"}).status_code == 422
    assert client.post("/memory", json={"content": "x", "importance": 99}).status_code == 422
    assert client.get("/memories", params={"tier": "bogus"}).status_code == 422
    assert client.get("/memories", params={"limit": -1}).status_code == 422
    for k in (-1, 0, 1000):
        assert client.post("/recall", json={"query": "x", "k": k}).status_code == 422
    assert client.post("/recall", json={"query": "x", "tier_filter": ["bogus"]}).status_code == 422
    ok = client.post("/memory", json={"content": "fine", "tier": "episodic", "importance": 1.0})
    assert ok.status_code == 200 and ok.json()["tier"] == "episodic"
    assert client.post("/recall", json={"query": "fine", "tier_filter": ["episodic"]}).status_code == 200


# --- F66 / F82 / F27: tools say what the code does --------------------------

async def test_sync_status_does_not_promise_a_p2p_node(mem):
    out = await call_tool(mem, "sync_status", {})
    assert out["implemented"] is False and "Start one" not in out["note"]


async def test_health_reports_the_embedder_actually_loaded(mem):
    from memcore_memory.retrieval.hybrid import is_semantic
    out = await call_tool(mem, "health_check", {})
    assert out["semantic"] == is_semantic(mem.embedder)
    if not out["semantic"]:
        assert out["embedder"].startswith("LocalHashEmbedder")


async def test_config_get_keeps_unwired_features_apart(mem):
    out = await call_tool(mem, "config_get", {})
    assert "pii_filter_enabled" not in out and "reranker_enabled" not in out
    assert "pii_filter_enabled" in out["not_implemented"]
    assert (await call_tool(mem, "config_get", {"key": "audit_log_enabled"}))["implemented"] is False


async def test_importance_does_not_choose_the_tier_and_the_description_says_so(mem):
    tiers = {(await call_tool(mem, "memory_add", {"content": f"m{imp}", "importance": imp}))["tier"]
             for imp in (0.99, 0.1)}
    assert tiers == {"working"}
    desc = next(t for t in get_tools_schema() if t["name"] == "memory_add")
    assert "inferred from importance" not in desc["description"]
    assert "tier assignment" not in json.dumps(desc["inputSchema"])


# --- R19: /mcp/call failures reach the server log ---------------------------

def test_mcp_call_failure_is_logged(client, mem, monkeypatch, capfd):
    async def broken(*a, **kw):
        raise RuntimeError("store exploded")
    monkeypatch.setattr(mem.store, "list_all", broken)
    monkeypatch.setattr(mem.store, "scan", broken)  # memory_stats reads through scan() since round 2
    r = mcp(client, "memory_stats")
    assert r.status_code == 200 and r.json()["ok"] is False
    err = capfd.readouterr().err
    assert "/mcp/call memory_stats failed" in err and "Traceback" in err


# --- R8 / R21: the bridge diagnoses what actually happened ------------------

async def test_read_timeout_is_not_reported_as_unreachable(isolate_data_dir):
    def handler(request):
        if request.url.path == "/mcp/tools":
            return httpx.Response(200, json={"server": SERVER_NAME, "version": "1", "tools": get_tools_schema()})
        raise httpx.ReadTimeout("")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://slow.test") as c:
        server = await RemoteMCPServer("http://slow.test", client=c, timeout=0.3).connect()
        with pytest.raises(ToolError) as info:
            await server.handle_tool_call("memory_add", {"content": "x"})
    msg = str(info.value)
    assert "unreachable" not in msg and "may still complete" in msg and "ReadTimeout" in msg


async def test_connect_error_has_a_reason(isolate_data_dir):
    def handler(request):
        if request.url.path == "/mcp/tools":
            return httpx.Response(200, json={"server": SERVER_NAME, "version": "1", "tools": get_tools_schema()})
        raise httpx.ConnectError("")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://down.test") as c:
        server = await RemoteMCPServer("http://down.test", client=c).connect()
        with pytest.raises(ToolError, match="not delivered: ConnectError"):
            await server.handle_tool_call("memory_add", {"content": "x"})


@pytest.mark.parametrize("tools_response, health, expected", [
    (httpx.Response(404), httpx.Response(404), "does not look like a memcore"),
    (httpx.Response(404), httpx.Response(200, json={"status": "ok", "version": "1.0.0"}), "predates"),
    (httpx.Response(200, json={"hello": 1}), None, "not a memcore instance"),
    (httpx.Response(200, text="<html>"), None, "other than JSON"),
    (httpx.Response(500, text="boom"), None, "HTTP 500"),
    (httpx.Response(401), None, "API key"),
])
async def test_connect_diagnoses_each_wrong_answer(isolate_data_dir, tools_response, health, expected):
    def handler(request):
        if request.url.path == "/health" and health is not None:
            return health
        if request.url.path == "/mcp/tools":
            return tools_response
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x.test") as c:
        with pytest.raises(RemoteUnavailable, match=expected):
            await RemoteMCPServer("http://x.test", client=c).connect()
