"""Round-2 test gaps from the 2026-09-24 audit (group J).

R22: the round-1 stdio test proved the local transport, but nothing ran the bridge
over a real socket, so "local and bridge agree" was a claim, not a test. Here the
same call sequence goes through ``server mcp`` (local store) and ``server mcp
--remote`` (uvicorn on 127.0.0.1:0) and the two transcripts are compared. The NaN
and read-timeout defects the finding named are pinned over the real transports too.
Both are fixed, so these are plain tests rather than the xfails the finding
proposed: a regression now fails instead of quietly turning an xfail into a pass.

F29 / F98: stronger versions of two round-1 tests whose assertions could not fail
on the behaviour they named.
"""

import asyncio
import contextlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import anyio
import httpx
import pytest

from memcore_memory.api.rest import create_app
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.mcp.server import HANDLERS
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore

REPO = Path(__file__).resolve().parent.parent

# Group I's docker stub and seeded store, reused rather than copied: the F98 case
# below needs the whole harness, and two copies would drift apart.
try:
    from test_audit_i_scripts import _calls as _i_calls, _deploy as _i_deploy, deploy_env  # noqa: F401
except Exception as _e:  # noqa: BLE001 - that file is another group's and may be mid-edit
    _i_import_error = _e

    @pytest.fixture
    def deploy_env():
        pytest.skip(f"test_audit_i_scripts harness unavailable: {_i_import_error!r}")


# ---------------------------------------------------------------- helpers

async def _build_memory():
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    return MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)


@contextlib.contextmanager
def _rest_server():
    """create_app over a real uvicorn socket, in a thread with its own loop.

    The store is built inside that loop, so nothing async is shared with the test's
    loop. It serves the per-test data dir that isolate_data_dir set up.
    """
    import uvicorn

    ready, box = threading.Event(), {}

    async def main():
        app = create_app(await _build_memory())
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
        box["server"] = server
        task = asyncio.create_task(server.serve())
        while not server.started and not task.done():
            await asyncio.sleep(0.02)
        if server.started:
            box["port"] = server.servers[0].sockets[0].getsockname()[1]
        ready.set()
        await task

    def run():
        try:
            asyncio.run(main())
        except BaseException as e:  # noqa: BLE001 - reported to the test below
            box["error"] = e
            ready.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert ready.wait(30), "uvicorn did not start"
    if "port" not in box:
        raise RuntimeError(f"uvicorn failed to start: {box.get('error')!r}")
    try:
        yield f"http://127.0.0.1:{box['port']}"
    finally:
        box["server"].should_exit = True
        thread.join(15)


def _child_env(data_dir: Path) -> dict:
    # From scratch, not inherited: no MNEM_MASTER_PASSWORD, MEMCORE_ENV, MNEM_API_KEY
    # or home data dir can reach the child.
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(data_dir),
            "MNEM_DATA_DIR": str(data_dir), "PYTHONPATH": str(REPO / "src")}


def _is_error(result) -> bool:
    return bool(getattr(result, "is_error", getattr(result, "isError", False)))


@contextlib.asynccontextmanager
async def _mcp_session(args, data_dir: Path):
    """A ClientSession over a real ``server mcp`` subprocess. Yields (session, stream_errors)."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "memcore_memory.cli.main", "server", "mcp", *args],
        env=_child_env(data_dir), cwd=str(REPO))
    # Anything on stdout that is not JSON-RPC arrives at the handler as an Exception.
    errors = []

    async def on_message(msg):
        if isinstance(msg, Exception):
            errors.append(msg)

    errlog_path = data_dir / "server.stderr"
    with open(errlog_path, "w") as errlog:
        async with stdio_client(params, errlog=errlog) as (read, write):
            async with ClientSession(read, write, message_handler=on_message) as session:
                await session.initialize()
                yield session, errors
    assert errors == [], (errors, errlog_path.read_text()[-2000:])


# Values that legitimately differ between two stores fed the same calls: fresh ids,
# clocks, and the decay state derived from them.
_VOLATILE = {"timestamp", "last_access", "last_accessed", "created_at", "updated_at",
             "retention", "strength", "avg_retention", "score", "scores", "rrf_score",
             "uptime", "uptime_seconds", "path", "data_dir"}


def _normalise(value, own_id):
    if isinstance(value, dict):
        return {k: ("<volatile>" if k in _VOLATILE else _normalise(v, own_id)) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalise(v, own_id) for v in value]
    if isinstance(value, str):
        return value.replace(own_id, "<id>") if own_id else value
    return value


async def _transcript(session):
    """One fixed call sequence; returns [(label, is_error, normalised payload)]."""
    out = []
    tools = (await session.list_tools()).tools
    out.append(("list_tools", False, sorted(t.name for t in tools)))

    added = await session.call_tool("memory_add", {
        "content": "transport parity check on SkyNAS", "tier": "semantic", "importance": 0.9,
        "entities": ["SkyNAS"], "metadata": {"source": "e2e"}})
    assert not _is_error(added), added.content[0].text
    own = json.loads(added.content[0].text)["id"]

    calls = [
        ("memory_add", None),
        ("memory_get", {"id": own, "touch": False}),
        ("memory_recall", {"query": "parity", "k": 5}),
        ("memory_search_bm25", {"query": "parity", "k": 5}),
        ("memory_search_graph", {"entity": "SkyNAS", "k": 5}),
        ("kg_list_entities", {}),
        ("memory_stats", {}),
        # Every error class once: missing row, unknown tool, schema violation,
        # missing required argument, non-finite number the SDK serialises as null.
        ("memory_get", {"id": "no-such-id"}),
        ("definitely_not_a_tool", {}),
        ("memory_add", {"content": "x", "importance": "high"}),
        ("memory_get", {}),
        ("memory_add", {"content": "x", "importance": float("nan")}),
        ("memory_delete", {"id": own}),
        ("memory_get", {"id": own, "touch": False}),
        ("memory_stats", {}),
    ]
    for name, args in calls:
        if args is None:
            out.append((name, False, _normalise(json.loads(added.content[0].text), own)))
            continue
        r = await session.call_tool(name, args)
        text = r.content[0].text if r.content else ""
        payload = text if _is_error(r) else json.loads(text)
        out.append((name, _is_error(r), _normalise(payload, own)))
    return out


# ---------------------------------------------------------------- R22: bridge parity

async def test_bridge_and_local_transports_answer_identically(isolate_data_dir, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    local_dir = isolate_data_dir / "local-child"
    bridge_dir = isolate_data_dir / "bridge-child"
    local_dir.mkdir()
    bridge_dir.mkdir()

    with anyio.fail_after(120):
        async with _mcp_session([], local_dir) as (session, _):
            local = await _transcript(session)
        with _rest_server() as url:
            async with _mcp_session(["--remote", url], bridge_dir) as (session, _):
                bridged = await _transcript(session)

    # The error cases must really be errors, or agreeing on them proves nothing.
    errors = [(name, err) for name, err, _ in local]
    assert errors.count(("definitely_not_a_tool", True)) == 1
    assert sum(err for _, err in errors) == 6, local
    for (name, l_err, l_val), (_, b_err, b_val) in zip(local, bridged):
        assert (l_err, l_val) == (b_err, b_val), f"{name}: local {l_val!r} != bridge {b_val!r}"
    assert len(local) == len(bridged)
    # The bridge child opened no store of its own; everything went to the server's.
    assert not (bridge_dir / "memory.db").exists()
    assert (isolate_data_dir / "memory.db").exists()


# ---------------------------------------------------------------- R22: NaN over the wire

def test_non_finite_numbers_are_refused_over_a_real_socket(isolate_data_dir, monkeypatch):
    # Raw bytes, because httpx and the MCP SDK both refuse to encode NaN, while
    # Python's json (and so Starlette) happily decodes it from anything else.
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    with _rest_server() as url, httpx.Client(base_url=url, timeout=30) as client:
        for body in (b'{"name":"memory_add","arguments":{"content":"x","importance":NaN}}',
                     b'{"name":"memory_add","arguments":{"content":"x","metadata":{"w":Infinity}}}'):
            r = client.post("/mcp/call", content=body, headers={"content-type": "application/json"})
            assert r.status_code == 200, r.text
            assert r.json()["ok"] is False and "NaN and Infinity" in r.json()["error"]
        # A response that carried NaN would not be JSON at all; this one parses.
        stats = client.post("/mcp/call", json={"name": "memory_stats", "arguments": {}}).json()
        assert stats["ok"] and stats["result"]["total"] == 0


def test_non_finite_numbers_are_refused_over_raw_stdio(isolate_data_dir):
    """The SDK client sends NaN as null; a hand-written client can send the literal."""
    child = isolate_data_dir / "child"
    child.mkdir()
    proc = subprocess.Popen(
        [sys.executable, "-m", "memcore_memory.cli.main", "server", "mcp"], cwd=REPO,
        env=_child_env(child), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    replies, deadline = {}, time.monotonic() + 60

    def send(*lines):
        proc.stdin.write("".join(line + "\n" for line in lines))
        proc.stdin.flush()

    def wait_for(*ids):
        while not set(ids) <= replies.keys() and time.monotonic() < deadline:
            line = proc.stdout.readline()
            if not line:
                break
            # Every stdout line must be JSON-RPC; json.loads raising here is the failure.
            msg = json.loads(line)
            if "id" in msg:
                replies[msg["id"]] = msg
        assert set(ids) <= replies.keys(), (replies, proc.stderr.read()[-2000:] if proc.poll() is not None else "")

    def tool(req_id, raw_args, name="memory_add"):
        return (f'{{"jsonrpc":"2.0","id":{req_id},"method":"tools/call",'
                f'"params":{{"name":"{name}","arguments":{raw_args}}}}}')

    try:
        send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
                 "protocolVersion": "2025-06-18", "capabilities": {},
                 "clientInfo": {"name": "raw", "version": "0"}}}),
             json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        wait_for(1)
        # importance NaN slipped past minimum/maximum; Infinity inside metadata was
        # stored as-is, which only the explicit non-finite check stops.
        send(tool(2, '{"content":"x","importance":NaN}'),
             tool(3, '{"content":"y","metadata":{"w":Infinity}}'))
        wait_for(2, 3)
        # Asked only after both answers: the server handles requests concurrently.
        send(tool(4, "{}", name="memory_stats"))
        wait_for(4)
    finally:
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    for req_id in (2, 3):
        # Refused either by the SDK's own parser (a protocol error) or by
        # validate_args (a tool error); never stored.
        if "error" not in replies[req_id]:
            assert replies[req_id]["result"]["isError"] is True, replies[req_id]
    assert json.loads(replies[4]["result"]["content"][0]["text"])["total"] == 0, replies


# ---------------------------------------------------------------- R22 / R8: timeout

async def test_bridge_timeout_says_the_call_may_still_complete(isolate_data_dir, monkeypatch):
    """R8: a read timeout was reported as '<url> unreachable: ' while the write landed."""
    monkeypatch.setattr(settings, "rate_limit_enabled", False)
    real_add = HANDLERS["memory_add"]

    async def slow_add(memory, a):
        await asyncio.sleep(1.5)
        return await real_add(memory, a)

    # The uvicorn thread dispatches through this same dict.
    monkeypatch.setitem(HANDLERS, "memory_add", slow_add)
    child = isolate_data_dir / "bridge-child"
    child.mkdir()
    with _rest_server() as url:
        async with _mcp_session(["--remote", url, "--timeout", "0.3"], child) as (session, _):
            with anyio.fail_after(60):
                r = await session.call_tool("memory_add", {"content": "slow write", "tier": "semantic"})
            assert _is_error(r)
            text = r.content[0].text
            assert "did not answer 'memory_add' within 0.3s" in text, text
            assert "may still complete" in text and "unreachable" not in text

            # ...and it did complete, which is why "unreachable" was the wrong thing to say.
            deadline = time.monotonic() + 15
            async with httpx.AsyncClient(base_url=url, timeout=10) as client:
                while time.monotonic() < deadline:
                    health = (await client.get("/health")).json()
                    if sum(health.get("tier_counts", {}).values()) == 1:
                        break
                    await asyncio.sleep(0.1)
            assert sum(health["tier_counts"].values()) == 1, health


# ---------------------------------------------------------------- F29: one clean line

def test_postgres_backend_fails_with_one_line_and_no_traceback(isolate_data_dir):
    """CliRunner swallows tracebacks into res.exception, so only a real process shows them."""
    env = _child_env(isolate_data_dir) | {
        "MNEM_BACKEND": "postgres",
        "MNEM_DATABASE_URL": "postgresql+asyncpg://x:y@127.0.0.1:1/none"}
    done = subprocess.run(
        [sys.executable, "-m", "memcore_memory.cli.main", "memory", "add", "must not land in sqlite"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0
    assert "Traceback" not in done.stderr + done.stdout, done.stderr[-2000:]
    assert len([ln for ln in done.stderr.splitlines() if ln.startswith("[memcore]")]) == 1, done.stderr
    assert not (isolate_data_dir / "memory.db").exists()


def test_unreachable_postgres_is_one_line_not_a_traceback(monkeypatch):
    # What a host with pgvector installed would see: the driver loads, the connect fails.
    import typer
    from typer.testing import CliRunner
    from memcore_memory.cli import main as cli

    async def refused(*a, **k):
        raise ConnectionRefusedError(111, "Connect call failed ('127.0.0.1', 1)")

    monkeypatch.setattr(cli, "get_memory_system", refused)
    res = CliRunner().invoke(cli.app, ["memory", "add", "x"])
    assert res.exit_code != 0
    # _fail ends in typer.Exit/SystemExit; an escaped OSError means a user-facing traceback.
    assert not isinstance(res.exception, OSError), repr(res.exception)
    assert isinstance(res.exception, (SystemExit, typer.Exit))


# ---------------------------------------------------------------- F98: relative SOURCE

def test_f98_deploy_stages_an_existing_relative_source_from_its_absolute_path(deploy_env, tmp_path):
    src = tmp_path / "relsrc"
    shutil.copytree(deploy_env["vol"], src)
    rel = os.path.relpath(src, REPO)
    assert not os.path.isabs(rel)
    done = _i_deploy(deploy_env, rel)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    calls = _i_calls(deploy_env)
    stage = [c for c in calls if c.startswith("run") and "/src:ro" in c]
    assert len(stage) == 1, calls
    mount = next(tok for tok in stage[0].split() if tok.endswith(":/src:ro"))
    # A relative -v source is a named volume to docker, not the directory meant.
    assert mount == f"{src.resolve()}:/src:ro", stage[0]
    assert f"Deployed from   : {src.resolve()}" in done.stdout
