"""Group F of the 2026-09-24 audit: the CLI, the SDK clients and the mnemosyne shim.

CLI tests run synchronously through CliRunner, because every command calls
asyncio.run() itself. Everything runs in the per-test data dir from conftest.
"""

import asyncio
import json
import os
import subprocess
import sys
import time

import httpx
import pytest
from typer.testing import CliRunner

from memcore_memory import config
from memcore_memory.cli import main as cli
from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.sdk import AsyncMnemosyneClient, MnemosyneClient
from memcore_memory.storage.vector_store import VectorStore

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_prod_marker(monkeypatch):
    for var in ("MEMCORE_ENV", "MNEM_ENV", "MNEM_BACKEND"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config.settings, "backend", "sqlite")
    cli._STATE["password"] = None
    yield
    cli._STATE["password"] = None


def invoke(*args, **kw):
    return runner.invoke(cli.app, list(args), **kw)


def add(content, *extra):
    res = invoke("memory", "add", content, *extra)
    assert res.exit_code == 0, res.output
    return res.stdout.split()[1]


def _system():
    return asyncio.run(cli.get_memory_system())


# --- F14: the shim shares module objects at every depth ----------------------

def test_mnemosyne_alias_is_the_same_module_at_every_depth(tmp_path):
    code = """
import importlib, sys
import memcore_memory.embeddings.factory as orig
import mnemosyne.mcp.server, mnemosyne.embeddings.factory, mnemosyne.core.tiers
import memcore_memory
dup = [k for k in sys.modules if k.startswith('mnemosyne.')
       and sys.modules[k] is not sys.modules.get('memcore_memory.' + k[len('mnemosyne.'):])]
assert dup == [], dup
assert memcore_memory.embeddings.factory is orig
assert memcore_memory.embeddings.__spec__.name == 'memcore_memory.embeddings'
assert memcore_memory.mcp.server.__spec__.name == 'memcore_memory.mcp.server'
from mnemosyne.core.tiers import Tier as A
from memcore_memory.core.tiers import Tier as B
assert A is B
importlib.reload(memcore_memory.core.tiers)
try:
    import mnemosyne.does_not_exist
except ModuleNotFoundError as e:
    assert e.name == 'mnemosyne.does_not_exist', e.name
else:
    raise AssertionError('missing module imported')
print('OK')
"""
    env = {**os.environ, "MNEM_DATA_DIR": str(tmp_path)}
    env.pop("MNEM_MASTER_PASSWORD", None)
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("OK")


# --- F28: sync commands are honest -------------------------------------------

def test_add_peer_fails_and_stores_nothing(isolate_data_dir):
    before = sorted(p.name for p in isolate_data_dir.iterdir())
    res = invoke("sync", "add-peer", "ws://10.0.0.9:7742")
    assert res.exit_code == 1
    assert "not implemented" in res.stderr and "Added" not in res.output
    assert sorted(p.name for p in isolate_data_dir.iterdir()) == before
    status = invoke("sync", "status")
    assert status.exit_code == 0 and "not running (not implemented)" in status.stdout


# --- F29: one construction path, which honours MNEM_BACKEND ------------------

def test_cli_builds_through_create_memory_system(monkeypatch):
    import memcore_memory
    seen = {}

    async def fake(password=None, **kw):
        seen["password"] = password
        return "system"

    monkeypatch.setattr(memcore_memory, "create_memory_system", fake)
    cli._STATE["password"] = "pw"
    assert asyncio.run(cli.get_memory_system()) == "system"
    assert seen == {"password": "pw"}


def test_postgres_backend_is_not_silently_served_from_sqlite(monkeypatch):
    monkeypatch.setattr(config.settings, "backend", "postgres")
    monkeypatch.setattr(config.settings, "database_url", "postgresql+asyncpg://x:y@127.0.0.1:1/none")
    res = invoke("memory", "add", "should not land in sqlite")
    assert res.exit_code != 0
    assert not config.settings.db_path.exists()


def test_server_start_builds_the_system_in_the_serving_loop(monkeypatch):
    import uvicorn
    loops = {}

    async def fake_system(password=None):
        loops["built"] = asyncio.get_running_loop()
        return _FakeMem()

    async def fake_serve(self, sockets=None):
        loops["served"] = asyncio.get_running_loop()

    monkeypatch.setattr(cli, "get_memory_system", fake_system)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    res = invoke("server", "start", "--port", "1")
    assert res.exit_code == 0, res.output
    assert loops["built"] is loops["served"]


class _FakeMem:
    store = vectors = kg = None


# --- F45: stored text is printed verbatim, never parsed as markup ------------

MARKUP = "note [/x] [y](z) [dim]d[/dim] [link text]"


def test_bracketed_content_survives_list_get_and_recall():
    mem_id = add(MARKUP, "--tier", "working")
    listed = invoke("memory", "list")
    assert listed.exit_code == 0, listed.output
    assert MARKUP in listed.stdout and "[working" in listed.stdout
    got = invoke("memory", "get", mem_id)
    assert got.exit_code == 0 and MARKUP in got.stdout and "[working]" in got.stdout
    recalled = invoke("memory", "recall", "note")
    assert recalled.exit_code == 0, recalled.output
    assert "[/x]" in recalled.stdout
    tier_list = invoke("memory", "working-list")
    assert tier_list.exit_code == 0 and MARKUP in tier_list.stdout


# --- F46: system health checks the files and emits JSON -----------------------

def test_health_on_an_empty_dir_is_degraded_json_and_writes_nothing(isolate_data_dir):
    before = sorted(p.name for p in isolate_data_dir.iterdir())
    res = invoke("system", "health")
    assert res.exit_code == 1
    out = json.loads(res.stdout)
    assert out["status"] == "degraded" and out["p2p"] == "not_implemented"
    assert out["checks"]["key"] == "missing" and out["checks"]["db"] == "missing"
    assert sorted(p.name for p in isolate_data_dir.iterdir()) == before


def test_health_after_init_is_ok():
    add("hello")
    res = invoke("system", "health")
    assert res.exit_code == 0, res.output
    out = json.loads(res.stdout)
    assert out["status"] == "ok" and out["checks"]["key"] == "raw"
    assert out["checks"]["db"]["rows"] == 1


# --- F47 / F83: SDK error handling and parity ----------------------------------

def _transport(status):
    return httpx.MockTransport(lambda req: httpx.Response(status, json={"detail": "nope"}))


@pytest.mark.parametrize("status", [404, 429, 500])
def test_sync_maintenance_calls_raise_on_http_errors(status):
    c = MnemosyneClient("http://test")
    c.client.close()
    c.client = httpx.Client(transport=_transport(status))
    for call in (c.consolidate, c.forget_expired):
        with pytest.raises(httpx.HTTPStatusError):
            call()
    c.close()


@pytest.mark.parametrize("status", [404, 429])
async def test_async_client_raises_on_http_errors(status):
    async with AsyncMnemosyneClient("http://test") as c:
        await c.client.aclose()
        c.client = httpx.AsyncClient(transport=_transport(status))
        for call in (c.consolidate, c.forget_expired, c.health, c.list):
            with pytest.raises(httpx.HTTPStatusError):
                await call()
    assert c.client.is_closed


def _public(cls):
    return {n for n in dir(cls) if not n.startswith("_")}


def test_sync_and_async_clients_offer_the_same_methods():
    assert _public(MnemosyneClient) == _public(AsyncMnemosyneClient)
    assert {"close", "kg_traverse", "consolidate"} <= _public(MnemosyneClient)


def test_sync_client_closes_as_a_context_manager():
    with MnemosyneClient("http://test") as c:
        pass
    assert c.client.is_closed


async def test_async_client_talks_to_the_real_app():
    from memcore_memory.api.rest import create_app
    mem = await cli.get_memory_system()
    transport = httpx.ASGITransport(app=create_app(mem))
    async with AsyncMnemosyneClient("http://test") as c:
        await c.client.aclose()
        c.client = httpx.AsyncClient(transport=transport)
        added = await c.add("async parity check")
        assert (await c.get(added["id"]))["content"] == "async parity check"
        assert any(r["id"] == added["id"] for r in await c.list())
        assert "tier_counts" in await c.health()


# --- F69: tier listings are newest first ----------------------------------------

def test_tier_list_limit_shows_the_newest():
    async def seed():
        mem = await cli.get_memory_system()
        now = time.time()
        for n in range(5):
            await mem.store.put(MemoryItem(content=f"m{n}", tier=Tier.EPISODIC, timestamp=now - 100 + n))
        oldest = await mem.store.get((await mem.store.list_by_tier(Tier.EPISODIC))[-1].id)
        await mem.store.put(oldest)   # INSERT OR REPLACE moves it to the end of rowid order
    asyncio.run(seed())
    res = invoke("memory", "list", "--tier", "episodic", "--limit", "2")
    assert res.exit_code == 0, res.output
    contents = [line.rsplit("| ", 1)[1] for line in res.stdout.splitlines()]
    assert contents == ["m4", "m3"]
    res = invoke("memory", "episodic-list", "--limit", "2")
    assert [line.split(" ", 1)[1] for line in res.stdout.splitlines()] == ["m4", "m3"]


# --- F70: no stale numbers -----------------------------------------------------

def test_help_count_matches_the_command_tree():
    import re
    total = sum(len(g.typer_instance.registered_commands) for g in cli.app.registered_groups)
    total += len(cli.app.registered_commands)
    stated = re.search(r"\((\d+) commands\)", cli.app.info.help)
    assert stated and int(stated.group(1)) == total


def test_recall_footer_shows_the_weights_in_use():
    add("WireGuard tunnel on SkyNAS")
    res = invoke("memory", "recall", "WireGuard")
    assert res.exit_code == 0, res.output
    assert "MRR" not in res.stdout and "vector(0.35)" not in res.stdout
    weights = _system().retriever.weights
    assert f"bm25({weights['bm25']:.2f})" in res.stdout


# --- F84 / R24: clean errors, validated options, detectable misses --------------

def test_invalid_tier_and_importance_are_rejected_without_a_traceback():
    for args in (("memory", "list", "--tier", "longterm"),
                 ("memory", "add", "x", "--tier", "bogus"),
                 ("memory", "add", "x", "--importance", "7")):
        res = invoke(*args)
        assert res.exit_code == 2, (args, res.output)
        assert "Traceback" not in res.output
    assert not config.settings.db_path.exists()


def test_missing_password_is_one_line():
    assert invoke("--password", "pw", "system", "init").exit_code == 0
    res = invoke("memory", "list")
    assert res.exit_code == 1
    assert "Traceback" not in res.output
    ours = [line for line in res.stderr.splitlines() if line.startswith("[memcore] ")]
    assert len(ours) == 1 and "password" in ours[0]
    assert len(res.stderr.splitlines()) <= 3   # plus the embedder's own notices


def test_get_and_delete_of_a_missing_id_exit_nonzero():
    add("present")
    res = invoke("memory", "get", "no-such-id")
    assert res.exit_code == 1 and "Not found" in res.stderr
    res = invoke("memory", "delete", "no-such-id")
    assert res.exit_code == 1 and "Not found" in res.stderr


def test_delete_removes_the_memory_and_its_vector():
    mem_id = add("to be deleted")
    res = invoke("memory", "delete", mem_id)
    assert res.exit_code == 0, res.output
    mem = _system()
    assert asyncio.run(mem.store.get(mem_id)) is None
    assert mem_id not in mem.vectors.ids


# --- F85: init says what it did, and makes 'initialized' true -------------------

def test_init_reports_created_then_loaded_and_never_replaces_the_key():
    first = invoke("system", "init")
    assert first.exit_code == 0 and "Created new key" in first.stdout
    key = config.settings.key_path.read_bytes()
    assert config.settings.db_path.exists()   # tables created by init, not lazily later
    second = invoke("system", "init")
    assert second.exit_code == 0 and "Loaded existing key" in second.stdout
    assert config.settings.key_path.read_bytes() == key


def test_protect_wraps_the_key_without_changing_it():
    add("secret")
    key = config.settings.key_path.read_bytes()
    res = invoke("system", "protect", "--password", "pw")
    assert res.exit_code == 0, res.output
    assert config.settings.key_path.read_bytes() != key
    listed = invoke("--password", "pw", "memory", "list")
    assert listed.exit_code == 0 and "secret" in listed.stdout
    again = invoke("system", "protect", "--password", "pw")
    assert again.exit_code == 1 and "already" in again.stderr


# --- R6: reindex-vectors is a rebuild -------------------------------------------

def test_reindex_vectors_drops_ghosts_and_reembeds_stale_rows():
    keep = add("kept memory")
    mem = _system()
    asyncio.run(mem.vectors.add("ghost", [1.0] + [0.0] * (mem.embedder.dim - 1), {}))
    # A row written by an embedder of another width, e.g. before a model switch.
    item = asyncio.run(mem.store.get(keep))
    item.embedding = [0.5] * 8
    asyncio.run(mem.store.update_content(item))

    res = invoke("system", "reindex-vectors")
    assert res.exit_code == 0, res.output
    assert "re-embedded 1" in res.stdout and "dropped 1" in res.stdout
    fresh = _system()
    assert fresh.vectors.ids == [keep]
    assert len(asyncio.run(fresh.store.get(keep)).embedding) == fresh.embedder.dim
    disk = json.loads(VectorStore(config.settings.vector_path, dim=fresh.embedder.dim)
                      .sidecar_path.read_text())
    assert disk["ids"] == [keep]


def test_reembed_refuses_the_hash_embedder_without_force():
    add("anything")
    res = invoke("system", "reindex-vectors", "--reembed")
    assert res.exit_code == 1 and "--force" in res.stderr
    res = invoke("system", "reindex-vectors", "--reembed", "--force")
    assert res.exit_code == 0 and "re-embedded 1" in res.stdout
