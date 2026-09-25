"""Audit group E, round 2: the MCP tools and the REST API.

Each test names the finding it pins down. All of them run against the isolated
data dir from conftest; none touches a real store or the network.
"""

import json
import sqlite3
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from memcore_memory.api import rate_limit
from memcore_memory.api.rest import create_app
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.mcp.remote import RemoteMCPServer
from memcore_memory.mcp.server import SERVER_NAME, ToolError, call_tool
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


def mcp(client, name, **arguments):
    return client.post("/mcp/call", json={"name": name, "arguments": arguments})


def write_export(name: str, text: str):
    d = settings.data_dir / "exports"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(text)


def corrupt(memory_id: str):
    """Flip one ciphertext byte so the row fails GCM authentication."""
    con = sqlite3.connect(settings.db_path)
    (ct,) = con.execute("SELECT content_enc FROM memories WHERE id=?", (memory_id,)).fetchone()
    ct = bytearray(ct)
    ct[0] ^= 0xFF
    con.execute("UPDATE memories SET content_enc=? WHERE id=?", (bytes(ct), memory_id))
    con.commit()
    con.close()


# --- R10: an id without a tier no longer stops the import halfway ------------

async def test_id_without_tier_imports_every_entry(mem):
    write_export("two.json", json.dumps([
        {"id": "id-1", "content": "first", "tier": "working"},
        {"id": "id-2", "content": "second"},
    ]))
    result = await call_tool(mem, "memory_import", {"path": "two.json"})
    assert result["imported"] == 2
    got = await mem.store.get_many(["id-1", "id-2"])
    assert set(got) == {"id-1", "id-2"}
    # Same rule as memory.add, judged on the entry's own (here: current) timestamp.
    assert got["id-2"].tier.value == "working"


async def test_timestamp_without_id_is_kept_not_replaced_by_now(mem):
    old = time.time() - 30 * 86400
    write_export("ts.json", json.dumps([{"content": "old note", "timestamp": old}]))
    await call_tool(mem, "memory_import", {"path": "ts.json"})
    (item,) = await mem.store.list_all()
    assert item.timestamp == pytest.approx(old)
    assert item.tier.value == "episodic"  # a month old is past the working TTL


async def test_failed_import_is_rolled_back_whole(mem, monkeypatch):
    await call_tool(mem, "memory_add", {"content": "already here", "tier": "episodic"})
    real = mem.vectors.add_many

    async def broken(rows):
        raise OSError("disk full")
    monkeypatch.setattr(mem.vectors, "add_many", broken)
    write_export("three.json", json.dumps([
        {"content": "plain, via add"},
        {"id": "r-1", "content": "restored one", "tier": "episodic"},
        {"id": "r-2", "content": "restored two", "tier": "semantic"},
    ]))
    with pytest.raises(ToolError, match="nothing imported.*removed again.*disk full"):
        await call_tool(mem, "memory_import", {"path": "three.json"})
    assert [i.content for i in await mem.store.list_all()] == ["already here"]
    # Not monkeypatch.undo(): that would also undo isolate_data_dir's settings.
    monkeypatch.setattr(mem.vectors, "add_many", real)
    # And the retry starts from the same store: nothing is skipped as "present".
    result = await call_tool(mem, "memory_import", {"path": "three.json"})
    assert result == {"imported": 3, "skipped_existing": 0, "total": 3}


# --- F40: one sidecar write for the restored rows ---------------------------

async def test_import_writes_vectors_in_one_batch(mem, monkeypatch):
    calls = {"add": 0, "add_many": 0}
    real_many = mem.vectors.add_many

    async def count_add(*a, **k):
        calls["add"] += 1

    async def count_many(rows):
        calls["add_many"] += 1
        return await real_many(rows)
    monkeypatch.setattr(mem.vectors, "add", count_add)
    monkeypatch.setattr(mem.vectors, "add_many", count_many)
    write_export("many.json", json.dumps(
        [{"id": f"m-{n}", "content": f"memory {n}", "tier": "episodic"} for n in range(25)]))
    assert (await call_tool(mem, "memory_import", {"path": "many.json"}))["imported"] == 25
    assert calls == {"add": 0, "add_many": 1}
    assert len(mem.vectors.ids) == 25


# --- F9: non-finite numbers and booleans are refused ------------------------

async def test_nan_importance_is_a_tool_error_not_a_crash(mem):
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ToolError, match="importance.*NaN and Infinity"):
            await call_tool(mem, "memory_add", {"content": "x", "importance": bad})
    with pytest.raises(ToolError, match="metadata/deep/0"):
        await call_tool(mem, "memory_add", {"content": "x", "metadata": {"deep": [float("nan")]}})
    assert await mem.store.list_all() == []


def test_nan_over_rest_bridge_is_refused(client):
    # Starlette's json.loads accepts the NaN literal; the tool call must not.
    r = client.post("/mcp/call", content='{"name":"memory_add","arguments":{"content":"x","importance":NaN}}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 200 and r.json()["ok"] is False and "NaN" in r.json()["error"]


async def test_boolean_importance_is_refused_in_update(mem):
    item = await call_tool(mem, "memory_add", {"content": "x", "tier": "episodic"})
    with pytest.raises(ToolError, match="importance"):
        await call_tool(mem, "memory_update", {"id": item["id"], "metadata": {"importance": True}})
    assert (await mem.store.get(item["id"])).metadata["importance"] == 0.5


@pytest.mark.parametrize("entry", [
    '{"content": "nanTS", "tier": "episodic", "timestamp": NaN}',
    '{"content": "infTS", "tier": "episodic", "timestamp": 1e999}',
    '{"content": "curve", "tier": "episodic", "forgetting": {"strength": Infinity}}',
    '{"content": "bool", "tier": "episodic", "metadata": {"importance": true}}',
])
async def test_import_refuses_non_finite_and_boolean_values(mem, entry):
    write_export("bad.json", f"[{entry}]")
    with pytest.raises(ToolError, match="nothing imported"):
        await call_tool(mem, "memory_import", {"path": "bad.json"})
    assert await mem.store.list_all() == []


# --- F63: searches over the bridge share the recall bucket ------------------

def test_bridge_searches_are_charged_to_the_recall_bucket(client):
    rate_limit.recall_limiter.buckets["testclient:recall"].extend([time.time()] * 100)
    for tool in ("memory_recall", "memory_search_bm25", "memory_search_graph"):
        args = {"entity": "x"} if tool == "memory_search_graph" else {"query": "x"}
        assert mcp(client, tool, **args).status_code == 429
    # Plain reads stay free, and the write bucket is not touched by searches.
    assert mcp(client, "memory_list").status_code == 200
    assert not rate_limit.write_limiter.buckets.get("testclient:write")


# --- R8: the bridge names what happened --------------------------------------

async def _call_through_bridge(handler, name, args):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x.test") as c:
        server = await RemoteMCPServer("http://x.test", client=c).connect()
        return await server.handle_tool_call(name, args)


def _tools_then(fn):
    def handler(request):
        if request.url.path == "/mcp/tools":
            return httpx.Response(200, json={"server": SERVER_NAME, "version": "1", "tools": get_tools_schema()})
        return fn(request)
    return handler


async def test_pool_timeout_is_reported_as_not_delivered(isolate_data_dir):
    def fail(request):
        raise httpx.PoolTimeout("")
    with pytest.raises(ToolError, match="not delivered: PoolTimeout"):
        await _call_through_bridge(_tools_then(fail), "memory_add", {"content": "x"})


@pytest.mark.parametrize("status", [404, 422])
async def test_other_4xx_is_a_tool_error_with_status_and_body(isolate_data_dir, status):
    handler = _tools_then(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(ToolError, match=f"returned {status} for 'memory_list': nope"):
        await _call_through_bridge(handler, "memory_list", {})


async def test_bridge_still_recognises_a_degraded_instance(isolate_data_dir):
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(503, json={"status": "unhealthy", "version": "1.0.0"})
        return httpx.Response(404)
    c = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x.test")
    async with c:
        from memcore_memory.mcp.remote import RemoteUnavailable
        with pytest.raises(RemoteUnavailable, match="predates"):
            await RemoteMCPServer("http://x.test", client=c).connect()


# --- R20: exact entity first, and k results despite stale links --------------

async def test_search_graph_puts_exact_entity_first(mem):
    docker = await call_tool(mem, "memory_add", {"content": "plain docker", "entities": ["docker"],
                                                 "tier": "episodic"})
    compose = await call_tool(mem, "memory_add", {"content": "compose file", "entities": ["Docker Compose"],
                                                  "tier": "episodic"})
    r = await call_tool(mem, "memory_search_graph", {"entity": "Docker Compose"})
    ids = [x["id"] for x in r["results"]]
    assert ids[0] == compose["id"] and r["exact_matches"] == 1
    assert docker["id"] in ids[1:]


async def test_search_graph_fills_k_past_stale_links(mem):
    live = [await call_tool(mem, "memory_add", {"content": f"live {n}", "entities": ["alpha"],
                                                "tier": "episodic"}) for n in range(3)]
    stale = [await call_tool(mem, "memory_add", {"content": f"stale {n}", "entities": ["alpha"],
                                                 "tier": "episodic"}) for n in range(3)]
    # The row goes but the link stays, as with memories written by an older version.
    for s in stale:
        await mem.store.delete(s["id"])
    r = await call_tool(mem, "memory_search_graph", {"entity": "alpha", "k": 3})
    assert {x["id"] for x in r["results"]} == {x["id"] for x in live}


# --- F68: temporal search needs no query ------------------------------------

async def test_temporal_search_without_query(mem):
    await call_tool(mem, "memory_add", {"content": "x", "tier": "episodic"})
    r = await call_tool(mem, "memory_search_temporal", {"k": 5})
    assert r["count"] == 1
    schema = next(t for t in get_tools_schema() if t["name"] == "memory_search_temporal")
    assert schema["inputSchema"]["required"] == [] and "ignored" in schema["description"]


# --- R2: unreadable rows are reported, never silently dropped ----------------

async def test_unreadable_row_is_reported_by_mcp_tools(mem):
    good = await call_tool(mem, "memory_add", {"content": "fine", "tier": "episodic"})
    bad = await call_tool(mem, "memory_add", {"content": "damaged", "tier": "episodic"})
    corrupt(bad["id"])
    with pytest.raises(ToolError, match=f"row {bad['id']}: content"):
        await call_tool(mem, "memory_get", {"id": bad["id"]})
    stats = await call_tool(mem, "memory_stats", {})
    assert stats["unreadable_count"] == 1 and stats["unreadable"] == [bad["id"]]
    health = await call_tool(mem, "health_check", {})
    assert health["status"] == "degraded" and health["unreadable"] == [bad["id"]]
    out = await call_tool(mem, "memory_export", {"path": "b.json"})
    assert out["exported"] == 1 and out["partial"] is True and out["skipped"] == [bad["id"]]
    written = json.loads((settings.data_dir / "exports" / "b.json").read_text())
    assert [m["id"] for m in written["memories"]] == [good["id"]]


def test_rest_health_and_get_report_unreadable_rows(client, mem):
    ids = [client.post("/memory", json={"content": f"m{n}", "tier": "episodic"}).json()["id"] for n in range(20)]
    assert client.get("/health").json()["status"] == "ok"
    corrupt(ids[0])
    h = client.get("/health")
    assert h.status_code == 200 and h.json()["status"] == "degraded"
    assert h.json()["unreadable"] == [ids[0]] and h.json()["unreadable_count"] == 1
    r = client.get(f"/memory/{ids[0]}")
    assert r.status_code == 422 and ids[0] in r.json()["detail"]
    # Past 10% the store is failing, not degraded.
    for i in ids[1:4]:
        corrupt(i)
    h = client.get("/health")
    assert h.status_code == 503 and h.json()["status"] == "unhealthy" and h.json()["unreadable_count"] == 4


def test_rest_health_is_unhealthy_when_nothing_is_readable(client):
    mid = client.post("/memory", json={"content": "only one", "tier": "episodic"}).json()["id"]
    corrupt(mid)
    assert client.get("/health").status_code == 503


# --- R12: a missing knowledge graph shows in /health ------------------------

def test_health_is_degraded_without_the_graph(client, mem):
    mem.kg_error = "memory.kg.db: file is not a database"
    h = client.get("/health").json()
    assert h["status"] == "degraded" and any("knowledge graph" in r for r in h["reasons"])


async def test_health_flags_a_graph_recreated_for_a_populated_store(mem):
    await call_tool(mem, "memory_add", {"content": "x", "tier": "episodic"})
    mem.kg.created_fresh = True
    with TestClient(create_app(mem)) as c:
        h = c.get("/health").json()
    assert h["status"] == "degraded" and any("created empty" in r for r in h["reasons"])


def test_fresh_graph_on_empty_store_is_not_degraded(isolate_data_dir, mem):
    mem.kg.created_fresh = True
    with TestClient(create_app(mem)) as c:
        c.post("/memory", json={"content": "first ever", "tier": "episodic"})
        assert c.get("/health").json()["status"] == "ok"


# --- F57 / F79: GET /memory/{id} content and read semantics ------------------

def test_rest_get_returns_plaintext_fields_and_does_not_rehearse(client):
    mid = client.post("/memory", json={"content": "x", "tier": "episodic", "entities": ["NAS"]}).json()["id"]
    r = client.get(f"/memory/{mid}").json()
    assert r["entities"] == ["NAS"] and isinstance(r["timestamp"], float)
    assert r["forgetting"]["rehearsals"] == 0
    assert client.get(f"/memory/{mid}").json()["forgetting"]["rehearsals"] == 0
    assert client.get(f"/memory/{mid}?touch=true").json()["forgetting"]["rehearsals"] == 1


async def test_mcp_get_rehearses_unless_touch_is_false(mem):
    item = await call_tool(mem, "memory_add", {"content": "x", "tier": "episodic"})
    assert (await call_tool(mem, "memory_get", {"id": item["id"], "touch": False}))["rehearsals"] == 0
    assert (await call_tool(mem, "memory_get", {"id": item["id"]}))["rehearsals"] == 1


async def test_recall_can_skip_rehearsal(mem, client):
    item = await call_tool(mem, "memory_add", {"content": "wireguard tunnel", "tier": "episodic"})
    await call_tool(mem, "memory_recall", {"query": "wireguard", "rehearse": False})
    client.post("/recall", json={"query": "wireguard", "rehearse": False})
    assert (await mem.store.get(item["id"])).forgetting.rehearsals == 0
    client.post("/recall", json={"query": "wireguard"})
    assert (await mem.store.get(item["id"])).forgetting.rehearsals == 1
    desc = next(t for t in get_tools_schema() if t["name"] == "memory_recall")["description"]
    assert "top 3" in desc


# --- R26 / F42: honest REST results and status codes ------------------------

def test_forget_returns_the_full_lifecycle_report(client):
    r = client.post("/forget").json()
    assert set(r) >= {"forgotten", "demoted", "promoted", "errors"}


async def test_mcp_forget_and_consolidate_report_counts(mem):
    assert set(await call_tool(mem, "memory_forget", {})) >= {"forgotten", "demoted", "promoted", "errors"}
    assert (await call_tool(mem, "memory_consolidate", {}))["promoted"] == 0


def test_add_value_error_is_422(client, mem, monkeypatch):
    async def refuse(*a, **k):
        raise ValueError("embedding has 3 dimensions, vector store expects 384")
    monkeypatch.setattr(mem, "add", refuse)
    r = client.post("/memory", json={"content": "x"})
    assert r.status_code == 422 and "dimensions" in r.json()["detail"]


def test_recall_with_every_retriever_failing_is_503(client, mem, monkeypatch):
    async def fail(*a, **k):
        raise RuntimeError("every query-dependent retriever failed: [...]")
    monkeypatch.setattr(mem, "recall", fail)
    r = client.post("/recall", json={"query": "x"})
    assert r.status_code == 503 and "every query-dependent" in r.json()["detail"]


def test_master_key_error_mid_request_is_503(client, mem, monkeypatch):
    from memcore_memory.crypto.key_manager import MasterKeyMismatch

    async def fail(*a, **k):
        raise MasterKeyMismatch("the loaded key decrypts none of the rows")
    monkeypatch.setattr(mem.store, "list_all", fail)
    r = client.get("/memories")
    assert r.status_code == 503 and "MasterKeyMismatch" in r.json()["detail"]


def test_unknown_tier_filter_is_422(client):
    assert client.get("/memories?tier=bogus").status_code == 422
