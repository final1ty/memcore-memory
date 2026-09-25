"""Group F, round 2: CLI, SDK and shim fixes left open or found by review.

Every test runs in the per-test data dir from conftest; nothing here opens a
real store or talks to a real server.
"""

import asyncio
import json
import os
import shutil
import sqlite3
import stat
import subprocess
import sys

import httpx
import pytest
from typer.testing import CliRunner

from memcore_memory import config
from memcore_memory.cli import main as cli
from memcore_memory.sdk import AsyncMnemosyneClient, MnemosyneClient

runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    for var in ("MEMCORE_ENV", "MNEM_ENV", "MNEM_BACKEND", "MNEM_REMOTE_URL", "MNEM_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(config.settings, "backend", "sqlite")
    monkeypatch.setattr(config.settings, "api_key", None)
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


def _listing(d):
    return sorted(p.name for p in d.iterdir())


# --- F46: health really is passive ---------------------------------------------

def test_health_after_init_creates_no_wal_or_shm(isolate_data_dir):
    assert invoke("system", "init").exit_code == 0
    before = _listing(isolate_data_dir)
    res = invoke("system", "health")
    assert res.exit_code == 0, res.output
    assert _listing(isolate_data_dir) == before


def test_health_on_a_read_only_copy_is_ok(isolate_data_dir, tmp_path, monkeypatch):
    add("hello")
    copy = tmp_path / "ro"
    shutil.copytree(isolate_data_dir, copy)
    for p in copy.iterdir():
        p.chmod(stat.S_IRUSR)
    copy.chmod(stat.S_IRUSR | stat.S_IXUSR)
    try:
        monkeypatch.setattr(config.settings, "key_path", copy / "master.key")
        monkeypatch.setattr(config.settings, "db_path", copy / "memory.db")
        monkeypatch.setattr(config.settings, "vector_path", copy / "vectors")
        res = invoke("system", "health")
        assert res.exit_code == 0, res.output
        assert json.loads(res.stdout)["checks"]["db"]["rows"] == 1
    finally:
        copy.chmod(stat.S_IRWXU)


def test_health_counts_rows_still_in_a_live_wal(isolate_data_dir):
    assert invoke("system", "init").exit_code == 0
    writer = sqlite3.connect(config.settings.db_path)
    try:
        writer.execute("PRAGMA journal_mode=WAL")
        writer.execute("PRAGMA wal_autocheckpoint=0")
        cols = [r[1] for r in writer.execute("PRAGMA table_info(memories)")]
        writer.execute(f"INSERT INTO memories (id) VALUES ('w1')" if cols == ["id"] else
                       "INSERT INTO memories (id, tier, timestamp) VALUES ('w1', 'working', 0)")
        writer.commit()
        assert config.settings.db_path.with_name("memory.db-wal").exists()
        res = invoke("system", "health")
        assert json.loads(res.stdout)["checks"]["db"]["rows"] == 1
    finally:
        writer.close()


# --- F14: `python -m mnemosyne...` works through the alias ---------------------

def test_run_module_through_the_mnemosyne_alias(tmp_path):
    env = {**os.environ, "MNEM_DATA_DIR": str(tmp_path)}
    env.pop("MNEM_MASTER_PASSWORD", None)
    out = subprocess.run([sys.executable, "-m", "mnemosyne.cli.main", "--help"],
                         env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "python -m mnemosyne.cli.main" in out.stdout


# --- F83 / R16 / F30: SDK clients ------------------------------------------------

def _recording():
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json={"results": [], "id": "x"})
    return seen, httpx.MockTransport(handler)


def test_path_segments_are_escaped_in_both_clients():
    seen, transport = _recording()
    c = MnemosyneClient("http://test")
    c.client.close()
    c.client = httpx.Client(transport=transport)
    c.kg_traverse("C#/x?y")
    c.get("a/b")
    c.delete("a#b")
    c.close()
    paths = [r.url.raw_path.decode() for r in seen]
    assert paths[0].startswith("/kg/traverse/C%23%2Fx%3Fy?")
    assert paths[1] == "/memory/a%2Fb" and paths[2] == "/memory/a%23b"

    async def run():
        aseen, atransport = _recording()
        async with AsyncMnemosyneClient("http://test") as ac:
            await ac.client.aclose()
            ac.client = httpx.AsyncClient(transport=atransport)
            await ac.kg_traverse("C#")
        return aseen
    assert asyncio.run(run())[0].url.raw_path.decode().startswith("/kg/traverse/C%23?")


def test_clients_have_no_default_url(monkeypatch):
    for cls in (MnemosyneClient, AsyncMnemosyneClient):
        with pytest.raises(ValueError, match="base_url"):
            cls()
    monkeypatch.setenv("MNEM_REMOTE_URL", "http://from-env:1/")
    monkeypatch.setenv("MNEM_API_KEY", "k1")
    c = MnemosyneClient()
    assert c.base_url == "http://from-env:1"
    assert c.client.headers["Authorization"] == "Bearer k1"
    c.close()
    explicit = MnemosyneClient("http://x", api_key="k2")
    assert explicit.client.headers["Authorization"] == "Bearer k2"
    explicit.close()


# --- F84: kg add-entity does not reset an existing node --------------------------

def test_add_entity_keeps_an_existing_nodes_type_and_props():
    mem = _system()
    asyncio.run(mem.kg.add_entity("Budapest", type="city", props={"country": "HU"}))
    res = invoke("kg", "add-entity", "budapest")
    assert res.exit_code == 0 and "already exists" in res.stdout
    ents = asyncio.run(_system().kg.list_entities())
    assert [e["type"] for e in ents if e["id"] == "budapest"] == ["city"]
    res = invoke("kg", "add-entity", "Szeged")
    assert res.exit_code == 0 and "Added entity Szeged" in res.stdout


# --- F73: search-blind, verify, migrate-aad ---------------------------------------

def test_search_blind_finds_by_keyword():
    mid = add("WireGuard tunnel on the router")
    add("unrelated note")
    res = invoke("memory", "search-blind", "wireguard", "--k", "5")
    assert res.exit_code == 0, res.output
    assert res.stdout.split()[0] == mid[:8] and len(res.stdout.strip().splitlines()) == 1
    none = invoke("memory", "search-blind", "zzzqqq")
    assert none.exit_code == 0 and none.stdout.strip() == ""


def _corrupt(mem_id):
    con = sqlite3.connect(config.settings.db_path)
    con.execute("UPDATE memories SET content_enc=? WHERE id=?", (b"\0" * 40, mem_id))
    con.commit()
    con.close()


def test_verify_exits_nonzero_and_names_unreadable_rows():
    add("fine")
    assert invoke("system", "verify").exit_code == 0
    bad = add("will break")
    _corrupt(bad)
    res = invoke("system", "verify")
    assert res.exit_code == 1
    assert "1 of 2 memories readable" in res.stdout and f"UNREADABLE {bad}" in res.stdout


def test_migrate_aad_needs_confirmation_when_not_a_tty():
    add("x")
    res = invoke("system", "migrate-aad")
    assert res.exit_code == 1 and "--yes" in res.stderr
    res = invoke("system", "migrate-aad", "--yes")
    assert res.exit_code == 0 and "Upgraded" in res.stdout


# --- F37: forget and consolidate report what they did -----------------------------

def test_forget_reports_errors_and_exits_nonzero():
    add("fine")
    ok = invoke("memory", "forget")
    assert ok.exit_code == 0 and "demoted 0" in ok.stdout
    _corrupt(add("broken"))
    res = invoke("memory", "forget")
    assert res.exit_code == 1 and "skipped" in res.stderr


def test_consolidate_prints_the_count():
    res = invoke("memory", "consolidate")
    assert res.exit_code == 0 and "Promoted 0" in res.stdout


# --- R12: reindex-kg --------------------------------------------------------------

def test_reindex_kg_relinks_memories():
    mid = add("Kovacs lives in Budapest", "--entities", "Budapest,Kovacs")
    mem = _system()
    asyncio.run(mem.kg.remove_memory(mid))
    assert asyncio.run(mem.kg.get_related_memories("Budapest")) == []
    res = invoke("system", "reindex-kg")
    assert res.exit_code == 0 and "Relinked 1" in res.stdout
    assert asyncio.run(_system().kg.get_related_memories("Budapest")) == [mid]


def test_unreadable_graph_is_one_line(monkeypatch):
    from memcore_memory import _UnavailableGraph

    class Mem:
        kg = _UnavailableGraph("x.kg.db", Exception("kg is broken"))

    async def fake(password=None):
        return Mem()
    monkeypatch.setattr(cli, "get_memory_system", fake)
    for args in (("kg", "traverse", "a"), ("kg", "add-entity", "a"), ("system", "reindex-kg")):
        res = invoke(*args)
        assert res.exit_code == 1 and res.stderr.startswith("[memcore] kg is broken"), args
        assert "Traceback" not in res.output


# --- F62: init and protect ----------------------------------------------------------

def test_init_with_a_password_on_a_raw_key_is_not_reported_protected():
    assert invoke("system", "init").exit_code == 0
    res = invoke("system", "init", "--password", "pw")
    assert res.exit_code == 0
    assert "UNPROTECTED" in res.stdout and "password-protected" not in res.stdout
    assert "Data dir:" in res.stdout


def test_protect_without_a_key_fails_before_asking():
    res = invoke("system", "protect")
    assert res.exit_code == 1 and "does not exist" in res.stderr


def test_protect_warns_about_the_next_restart():
    add("x")
    res = invoke("system", "protect", "--password", "pw")
    assert res.exit_code == 0 and "MNEM_MASTER_PASSWORD" in res.stderr


# --- F40: reindex-vectors reports rows of another width -----------------------------

def test_reindex_vectors_counts_other_width_rows():
    mid = add("kept")
    mem = _system()
    item = asyncio.run(mem.store.get(mid))
    item.embedding = [0.5] * 8
    asyncio.run(mem.store.update_content(item))
    res = invoke("system", "reindex-vectors")
    assert res.exit_code == 0, res.output
    assert "1 of another width" in res.stdout


# --- F30 / R4: server commands ---------------------------------------------------------

class _Server:
    async def serve_stdio(self):
        return None


def test_mcp_bridge_names_its_mode_and_passes_the_key(monkeypatch):
    import memcore_memory.mcp.remote as remote
    seen = {}

    class Fake(_Server):
        def __init__(self, url, timeout=30.0, api_key=None):
            seen.update(url=url, api_key=api_key)

        async def connect(self):
            return self

    monkeypatch.setattr(remote, "RemoteMCPServer", Fake)
    monkeypatch.setenv("MNEM_REMOTE_URL", "http://bridge:1")
    res = invoke("server", "mcp", "--api-key", "sekrit")
    assert res.exit_code == 0, res.output
    assert "[memcore] bridge mode -> http://bridge:1 (from MNEM_REMOTE_URL)" in res.stderr
    assert res.stdout == ""
    assert seen == {"url": "http://bridge:1", "api_key": "sekrit"}



def test_mcp_bridge_reads_the_key_from_a_file(monkeypatch, tmp_path):
    import memcore_memory.mcp.remote as remote
    seen = {}

    class Fake(_Server):
        def __init__(self, url, timeout=30.0, api_key=None):
            seen.update(api_key=api_key)

        async def connect(self):
            return self

    monkeypatch.setattr(remote, "RemoteMCPServer", Fake)
    monkeypatch.delenv("MNEM_API_KEY", raising=False)
    key = tmp_path / "api-key"
    key.write_text("from-file\n")
    res = invoke("server", "mcp", "--remote", "http://bridge:1", "--api-key-file", str(key))
    assert res.exit_code == 0, res.output
    assert seen == {"api_key": "from-file"}

    res = invoke("server", "mcp", "--remote", "http://bridge:1", "--api-key-file", str(tmp_path / "nope"))
    assert res.exit_code == 1
    assert "cannot read --api-key-file" in res.stderr


def test_mcp_local_mode_names_the_data_dir(monkeypatch):
    import memcore_memory.mcp.server as server

    async def fake(password=None):
        return object()
    monkeypatch.setattr(cli, "get_memory_system", fake)
    monkeypatch.setattr(server, "MCPServer", lambda mem: _Server())
    res = invoke("server", "mcp")
    assert res.exit_code == 0, res.output
    assert f"[memcore] local store (sqlite): {config.settings.data_dir.absolute()}" in res.stderr
    assert res.stdout == ""


@pytest.mark.parametrize("host,warned", [("0.0.0.0", True), ("127.0.0.1", False)])
def test_server_start_warns_when_exposed_without_a_key(monkeypatch, host, warned):
    import uvicorn

    async def fake_system(password=None):
        class M:
            store = vectors = kg = None
        return M()

    async def fake_serve(self, sockets=None):
        return None

    monkeypatch.setattr(cli, "get_memory_system", fake_system)
    monkeypatch.setattr(uvicorn.Server, "serve", fake_serve)
    res = invoke("server", "start", "--host", host, "--port", "1")
    assert res.exit_code == 0, res.output
    assert ("UNAUTHENTICATED" in res.stderr) is warned
    monkeypatch.setattr(config.settings, "api_key", "k")
    res = invoke("server", "start", "--host", host, "--port", "1")
    assert "UNAUTHENTICATED" not in res.stderr
