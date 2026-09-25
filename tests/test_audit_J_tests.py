"""Test-suite gaps from the 2026-09-24 audit (group J).

- Every one of the 29 MCP tools is called through the handler table. The old
  "every tool has a handler" test compared name sets only, so 17 tools had never
  been invoked by any test.
- One test speaks real MCP over stdio to a ``server mcp`` subprocess. Everything
  else calls ``handle_tool_call`` directly, which skips the JSON-RPC framing, the
  ToolError -> isError mapping and ``json.dumps`` of the result, and cannot notice
  a stray ``print()`` corrupting stdout.
- The suite stays hermetic when the shell exports MEMCORE_ENV=prod or MNEM_* vars.

Assertions are kept at the level of the tool contract (ids found, counts, state
changes), not incidental output, because the tools themselves are still evolving.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import anyio
import pytest

from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.mcp.server import HANDLERS, MCPServer, ToolError
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore

REPO = Path(__file__).resolve().parent.parent


async def _build_server():
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    mem = MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)
    return MCPServer(mem)


@pytest.fixture
async def seeded(isolate_data_dir):
    server = await _build_server()
    call = server.handle_tool_call
    a = await call("memory_add", {"content": "SkyNAS runs Docker", "tier": "episodic",
                                  "importance": 0.7, "entities": ["SkyNAS", "Docker"]})
    b = await call("memory_add", {"content": "WireGuard tunnel on SkyNAS", "tier": "working"})
    return server, a["id"], b["id"]


def _ids(result, key="results"):
    return [r["id"] for r in result[key]]


async def _check_add(call, a, b):
    r = await call("memory_add", {"content": "third", "tier": "semantic"})
    assert (await call("memory_get", {"id": r["id"]}))["content"] == "third"


async def _check_get(call, a, b):
    r = await call("memory_get", {"id": a})
    assert r["id"] == a and r["content"] == "SkyNAS runs Docker"


async def _check_delete(call, a, b):
    await call("memory_delete", {"id": b})
    with pytest.raises(ToolError):
        await call("memory_get", {"id": b})
    assert (await call("health_check", {}))["memories"] == 1


async def _check_update(call, a, b):
    await call("memory_update", {"id": a, "content": "SkyNAS runs Jellyfin"})
    assert (await call("memory_get", {"id": a}))["content"] == "SkyNAS runs Jellyfin"
    # The BM25 index must follow the rewrite, not serve the old text.
    assert a in _ids(await call("memory_search_bm25", {"query": "Jellyfin", "k": 5}))


async def _check_recall(call, a, b):
    assert _ids(await call("memory_recall", {"query": "Docker", "k": 5}))[0] == a


async def _check_bm25(call, a, b):
    ids = _ids(await call("memory_search_bm25", {"query": "WireGuard", "k": 5}))
    assert ids == [b]


async def _check_vector(call, a, b):
    # Hash embeddings make the order meaningless; the contract is that it searches.
    # With k above the store size every memory comes back, so an arm that silently
    # returns nothing fails here instead of passing a subset check.
    ids = _ids(await call("memory_search_vector", {"query": "Docker", "k": 5}))
    assert set(ids) == {a, b}


async def _check_temporal(call, a, b):
    ids = _ids(await call("memory_search_temporal", {"query": "anything", "k": 5}))
    assert set(ids) <= {a, b} and ids


async def _check_graph(call, a, b):
    assert a in _ids(await call("memory_search_graph", {"entity": "SkyNAS", "k": 5}))


async def _check_list(call, a, b):
    r = await call("memory_list", {"tier": "working"})
    assert _ids(r, "items") == [b]


async def _check_list_all(call, a, b):
    assert set(_ids(await call("memory_list_all", {}), "items")) == {a, b}


async def _check_promote(call, a, b):
    await call("memory_promote", {"id": b, "target_tier": "semantic"})
    assert (await call("memory_get", {"id": b}))["tier"] == "semantic"


async def _check_demote(call, a, b):
    await call("memory_promote", {"id": b, "target_tier": "semantic"})
    await call("memory_demote", {"id": b, "target_tier": "episodic"})
    assert (await call("memory_get", {"id": b}))["tier"] == "episodic"


async def _check_touch(call, a, b):
    first = await call("memory_touch", {"id": b})
    second = await call("memory_touch", {"id": b})
    assert second["rehearsals"] == first["rehearsals"] + 1
    assert second["strength"] > first["strength"]


async def _check_forget(call, a, b):
    # Both memories are seconds old, so a forget pass must not take either of them.
    r = await call("memory_forget", {})
    assert r["forgotten"] == 0
    assert set(_ids(await call("memory_list_all", {}), "items")) == {a, b}


async def _check_consolidate(call, a, b):
    # One memory that meets the semantic rule (importance > 0.8, rehearsed past the
    # threshold) next to one that does not, so a no-op pass cannot go green.
    c = (await call("memory_add", {"content": "consolidate me", "tier": "episodic",
                                   "importance": 0.9}))["id"]
    for _ in range(settings.semantic_consolidation_threshold):
        await call("memory_touch", {"id": c})
    r = await call("memory_consolidate", {})
    assert r["promoted"] == 1
    assert r["before"].get("semantic", 0) + 1 == r["after"].get("semantic", 0)
    assert (await call("memory_get", {"id": c}))["tier"] == "semantic"
    assert (await call("memory_get", {"id": a}))["tier"] == "episodic"
    assert (await call("health_check", {}))["memories"] == 3


async def _check_stats(call, a, b):
    assert (await call("memory_stats", {}))["total"] == 2


async def _check_export(call, a, b):
    r = await call("memory_export", {"path": "backup.json"})
    assert r["exported"] == 2
    data = json.loads(Path(r["path"]).read_text())
    entries = data["memories"] if isinstance(data, dict) else data
    assert {e["content"] for e in entries} == {"SkyNAS runs Docker", "WireGuard tunnel on SkyNAS"}


async def _check_import(call, a, b):
    await call("memory_export", {"path": "backup.json"})
    await call("memory_import", {"path": "backup.json"})
    # Restoring a store's own backup into it used to double every memory.
    assert (await call("health_check", {}))["memories"] == 2


async def _check_kg_add_entity(call, a, b):
    await call("kg_add_entity", {"entity": "Ubuntu"})
    names = {e["id"] for e in (await call("kg_list_entities", {}))["entities"]}
    assert "ubuntu" in names


async def _check_kg_add_relation(call, a, b):
    await call("kg_add_relation", {"src": "SkyNAS", "dst": "Ubuntu", "relation": "runs"})
    assert "ubuntu" in (await call("kg_get_related", {"entity": "SkyNAS"}))["related"]


async def _check_kg_traverse(call, a, b):
    await call("kg_add_relation", {"src": "SkyNAS", "dst": "Ubuntu", "relation": "runs"})
    edges = (await call("kg_traverse", {"entity": "SkyNAS"}))["edges"]
    assert {"src", "dst", "relation", "weight"} <= set(edges[0])
    assert ("skynas", "ubuntu", "runs") in {(e["src"], e["dst"], e["relation"]) for e in edges}


async def _check_kg_get_related(call, a, b):
    # memory_add recorded SkyNAS and Docker as co-occurring entities.
    assert "docker" in (await call("kg_get_related", {"entity": "SkyNAS"}))["related"]


async def _check_kg_list_entities(call, a, b):
    names = {e["id"] for e in (await call("kg_list_entities", {}))["entities"]}
    assert {"skynas", "docker"} <= names


async def _check_kg_delete_entity(call, a, b):
    await call("kg_add_relation", {"src": "SkyNAS", "dst": "Ubuntu", "relation": "runs"})
    await call("kg_delete_entity", {"entity": "Ubuntu"})
    assert "ubuntu" not in (await call("kg_get_related", {"entity": "SkyNAS"}))["related"]


async def _check_sync_status(call, a, b):
    # P2P is not implemented; the tool must not claim otherwise.
    assert (await call("sync_status", {})).get("implemented") is False


async def _check_sync_peers(call, a, b):
    r = await call("sync_peers", {})
    assert r["peers"] == list(settings.p2p_peers) and r["count"] == len(r["peers"])


async def _check_config_get(call, a, b):
    cfg = await call("config_get", {})
    assert Path(cfg["data_dir"]) == Path(settings.data_dir)
    assert not any("password" in k.lower() for k in cfg)


async def _check_health(call, a, b):
    r = await call("health_check", {})
    assert r["status"] == "ok" and r["memories"] == 2


CHECKS = {
    "memory_add": _check_add, "memory_get": _check_get, "memory_delete": _check_delete,
    "memory_update": _check_update, "memory_recall": _check_recall,
    "memory_search_bm25": _check_bm25, "memory_search_vector": _check_vector,
    "memory_search_temporal": _check_temporal, "memory_search_graph": _check_graph,
    "memory_list": _check_list, "memory_list_all": _check_list_all,
    "memory_promote": _check_promote, "memory_demote": _check_demote,
    "memory_touch": _check_touch, "memory_forget": _check_forget,
    "memory_consolidate": _check_consolidate, "memory_stats": _check_stats,
    "memory_export": _check_export, "memory_import": _check_import,
    "kg_add_entity": _check_kg_add_entity, "kg_add_relation": _check_kg_add_relation,
    "kg_traverse": _check_kg_traverse, "kg_get_related": _check_kg_get_related,
    "kg_list_entities": _check_kg_list_entities, "kg_delete_entity": _check_kg_delete_entity,
    "sync_status": _check_sync_status, "sync_peers": _check_sync_peers,
    "config_get": _check_config_get, "health_check": _check_health,
}


def test_every_handler_has_a_behavioural_check():
    """A new tool must come with a call below, not just a matching name."""
    assert set(CHECKS) == set(HANDLERS)


@pytest.mark.parametrize("tool", sorted(CHECKS))
async def test_tool_does_what_it_says(seeded, tool):
    server, a, b = seeded
    await CHECKS[tool](server.handle_tool_call, a, b)


async def test_export_import_restores_ids_and_rehearsals_into_a_fresh_store(seeded, tmp_path, monkeypatch):
    server, a, b = seeded
    call = server.handle_tool_call
    await call("memory_touch", {"id": a})
    touched = await call("memory_get", {"id": a})
    exported = Path((await call("memory_export", {"path": "backup.json"}))["path"])

    # A second, empty store: same process, different data dir.
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(settings, "data_dir", other)
    for field in ("db_path", "key_path", "vector_path", "audit_log_path", "working_buffer_path"):
        monkeypatch.setattr(settings, field, other / Path(getattr(settings, field)).name)
    fresh = await _build_server()
    (other / "exports").mkdir(exist_ok=True)
    (other / "exports" / "backup.json").write_bytes(exported.read_bytes())
    await fresh.handle_tool_call("memory_import", {"path": "backup.json"})

    restored = await fresh.handle_tool_call("memory_get", {"id": a})
    assert restored["content"] == touched["content"]
    # memory_get itself rehearses, so the restored count may be one higher.
    assert restored["rehearsals"] >= touched["rehearsals"] >= 1
    assert (await fresh.handle_tool_call("health_check", {}))["memories"] == 2


# ---------------------------------------------------------------- real transport

def _stdio_env(data_dir: Path) -> dict:
    # Built from scratch rather than inherited, so no MNEM_MASTER_PASSWORD, no
    # MEMCORE_ENV and no home data dir can reach the child.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(data_dir),
           "MNEM_DATA_DIR": str(data_dir), "PYTHONPATH": str(REPO / "src")}
    return env


def _is_error(result) -> bool:
    # The SDK renamed isError to is_error; the wire field is isError either way.
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


async def test_stdio_server_speaks_mcp(isolate_data_dir):
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "memcore_memory.cli.main", "server", "mcp"],
        env=_stdio_env(isolate_data_dir), cwd=str(REPO),
    )
    # Anything on stdout that is not JSON-RPC arrives here as an Exception.
    stream_errors = []

    async def on_message(msg):
        if isinstance(msg, Exception):
            stream_errors.append(msg)

    errlog = open(isolate_data_dir / "server.stderr", "w")
    try:
        with anyio.fail_after(60):
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write, message_handler=on_message) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
                    assert {t.name for t in tools} == set(HANDLERS)

                    added = await session.call_tool(
                        "memory_add", {"content": "stdio transport check", "tier": "semantic"})
                    assert not _is_error(added)
                    mem_id = json.loads(added.content[0].text)["id"]

                    got = await session.call_tool("memory_get", {"id": mem_id})
                    assert not _is_error(got)
                    assert json.loads(got.content[0].text)["content"] == "stdio transport check"

                    missing = await session.call_tool("memory_get", {"id": "no-such-id"})
                    assert _is_error(missing)
    finally:
        errlog.close()
    assert stream_errors == [], (stream_errors, (isolate_data_dir / "server.stderr").read_text()[-2000:])
    # The child wrote into the tmp dir it was given, nowhere else.
    assert (isolate_data_dir / "memory.db").exists()


# ---------------------------------------------------------------- hermetic suite

def test_the_session_environment_is_scrubbed():
    assert "MEMCORE_ENV" not in os.environ
    leaked = [k for k in os.environ
              if k.upper().startswith("MNEM_") and k.upper() not in {"MNEM_BACKEND", "MNEM_DATABASE_URL"}]
    assert leaked == []


def test_suite_survives_a_hostile_shell_environment(tmp_path):
    """MEMCORE_ENV=prod broke 23 tests, masked only by a test that deleted the variable
    for everything after it; a stray MNEM_DB_PATH redirected Settings()."""
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("MNEM_", "MEMCORE_"))}
    env.update(MEMCORE_ENV="prod", MNEM_DB_PATH=str(tmp_path / "stray" / "other.db"),
               MNEM_DATA_DIR=str(tmp_path / "stray"), HOME=str(tmp_path))
    files = ["tests/test_mcp_server.py", "tests/test_key_persistence.py", "tests/test_memory.py",
             "tests/test_config_paths.py", "tests/test_mcp_remote.py"]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *files],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    assert not (tmp_path / "stray" / "other.db").exists()
