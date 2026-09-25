"""Audit group I, round 2: what the review of the round-1 script changes found.

Same rules as test_audit_i_scripts.py, whose helpers and docker stub this reuses:
nothing touches docker, the live container or a real store. Every rescue here
runs with snapshot, fetch_key and find_data_dir replaced, and with a `docker` on
PATH that refuses everything, so a missed stub fails instead of reaching the
real container.
"""

import ast
import asyncio
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

import test_audit_i_scripts as base
from test_audit_i_scripts import deploy_env  # noqa: F401 - fixture

REPO, SCRIPTS, PY = base.REPO, base.SCRIPTS, base.PY
rescue, sync = base.rescue, base.sync
fp = base._load("store_fingerprint")


# --- a store with hand-made graph content, and a REST server over it ----------

_KG_EXTRAS = r'''
import asyncio
from memcore_memory.config import settings
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph

async def main():
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), AES256GCM(KeyManager(settings.key_path).load_or_create()))
    await kg.init()
    await kg.add_entity("Jellyfin", "service", {"port": 8096})
    await kg.add_relation("Router", "SkyNAS", "routes_for", 0.7)

asyncio.run(main())
'''


@pytest.fixture(scope="module")
def kg_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("kgrest")
    data = root / "data"
    base._seed(data)
    subprocess.run([PY, "-c", _KG_EXTRAS], check=True, cwd=REPO, env=base._clean_env(MNEM_DATA_DIR=str(data)))
    port = base._free_port()
    proc = base._start_server(data, port, root / "server.log")
    yield {"data": data, "url": f"http://127.0.0.1:{port}"}
    base._stop(proc.pid)


_RUNNER = r'''
import asyncio, importlib.util, os, shutil, sqlite3, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location("rescue", sys.argv[1])
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
src = Path(sys.argv[2])
def snapshot(container, path, dst, required):
    name = Path(path).name
    override = os.environ.get("TEST_KG_SOURCE")
    source = Path(override) if (override and name == "memory.kg.db") else src / name
    if not source.exists():
        if required: sys.exit("missing " + str(source))
        return False
    s = sqlite3.connect(source); d = sqlite3.connect(dst); s.backup(d); d.close(); s.close()
    return True
def fetch_key(container, path, dst):
    if os.environ.get("TEST_NO_KEY"): return False
    shutil.copyfile(src / "master.key", dst); return True
m.snapshot, m.fetch_key, m.find_data_dir = snapshot, fetch_key, (lambda c: "/data")
asyncio.run(m.main(sys.argv[3:]))
'''


def _no_docker(tmp_path):
    bindir = tmp_path / "nodocker"
    bindir.mkdir(exist_ok=True)
    stub = bindir / "docker"
    stub.write_text("#!/bin/sh\necho 'test: real docker must not be reached' >&2\nexit 97\n")
    stub.chmod(0o755)
    return f"{bindir}:{os.environ['PATH']}"


def _rescue(store, out, tmp_path, *flags, **env):
    return subprocess.run(
        [PY, "-c", _RUNNER, str(SCRIPTS / "rescue-container-store.py"), str(store["data"]), str(out),
         "--endpoint", store["url"], *flags],
        cwd=REPO, env=base._clean_env(PATH=_no_docker(tmp_path), **env), capture_output=True, text=True)


_GRAPH = r'''
import asyncio, json, sys
from pathlib import Path
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.graph import kg as kgmod
d = Path(sys.argv[1])
async def main():
    g = kgmod.KnowledgeGraph(d / "memory.kg.db", AES256GCM((d / "master.key").read_bytes()))
    await g.init()
    import sqlite3
    con = sqlite3.connect(d / "memory.kg.db")
    nodes = {}
    for nid, type_, le, n, pe, pn, aad_v in con.execute(
            "SELECT id, type, label_enc, nonce, props_enc, props_nonce, aad_v FROM kg_nodes"):
        label = g._label(nid, le, n, aad_v)
        nodes[label] = {"type": type_, "props": g._dec_json(pn, pe, kgmod._node_aad(nid, "props") if aad_v else b"")}
    hand = con.execute("SELECT COUNT(*) FROM kg_edges WHERE memory_id IS NULL").fetchone()[0]
    links = con.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0]
    print(json.dumps({"nodes": nodes, "hand_edges": hand, "links": links,
                      "traverse": await g.traverse("Router", depth=1)}))
asyncio.run(main())
'''


def _graph(data_dir: Path):
    done = subprocess.run([PY, "-c", _GRAPH, str(data_dir)], cwd=REPO, env=base._clean_env(),
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


# --- F10: hand-made graph content is carried, or the rescue refuses ------------

def test_f10_rescue_carries_hand_made_entities_and_relations_with_the_containers_key(kg_store, tmp_path):
    out = tmp_path / "rescued"
    done = _rescue(kg_store, out, tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    g = _graph(out)
    assert g["nodes"]["Jellyfin"] == {"type": "service", "props": {"port": 8096}}
    assert "Router" in g["nodes"]                     # the caller's spelling, not 'router'
    assert g["hand_edges"] == 1
    rel = [e for e in g["traverse"] if e["relation"] == "routes_for"]
    assert len(rel) == 1 and rel[0]["weight"] == 0.7
    assert g["links"] == 2                            # SkyNAS and Docker, linked to their memory


def _v1_graph(path: Path, key: bytes, memory_id: str):
    """A graph as the pre-audit (HEAD) code wrote it: lowercased ids, no AAD."""
    from memcore_memory.crypto.aes_gcm import AES256GCM
    c = AES256GCM(key)
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, nonce BLOB, "
                "props_enc BLOB, props_nonce BLOB)")
    con.execute("CREATE TABLE kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT, weight REAL, "
                "timestamp REAL, props_enc BLOB, props_nonce BLOB)")
    for name, type_, props in (("SkyNAS", "entity", {}), ("Docker", "entity", {}), ("Router", "device", {}),
                               ("Jellyfin", "service", {"port": 8096})):
        n, ct = c.encrypt(name.encode())
        pn, pct = c.encrypt(json.dumps(props).encode())
        con.execute("INSERT INTO kg_nodes VALUES (?,?,?,?,?,?)", (name.lower(), type_, ct, n, pct, pn))
    n, ct = c.encrypt(json.dumps({"memory_id": memory_id}).encode())
    con.execute("INSERT INTO kg_edges VALUES ('e1','skynas','docker','co_occurs',1.0,0,?,?)", (ct, n))
    n, ct = c.encrypt(b"{}")
    con.execute("INSERT INTO kg_edges VALUES ('e2','router','skynas','routes_for',0.7,0,?,?)", (ct, n))
    con.commit()
    con.close()


def test_f10_rescue_of_a_v1_graph_keeps_label_case_type_and_props(kg_store, tmp_path):
    items = base._store_items(kg_store["data"])
    mid = next(i["id"] for i in items if i["content"] == "SkyNAS runs Docker")
    v1 = tmp_path / "v1.kg.db"
    _v1_graph(v1, (kg_store["data"] / "master.key").read_bytes(), mid)
    out = tmp_path / "rescued"
    done = _rescue(kg_store, out, tmp_path, TEST_KG_SOURCE=str(v1))
    assert done.returncode == 0, done.stdout + done.stderr
    g = _graph(out)
    assert g["nodes"]["Router"]["type"] == "device"
    assert g["nodes"]["Jellyfin"] == {"type": "service", "props": {"port": 8096}}
    assert [e["relation"] for e in g["traverse"]] == ["routes_for"]
    assert g["hand_edges"] == 1                       # the co_occurs edge was a memory's, not carried by hand
    # The snapshot the rescue upgraded was its private copy.
    assert rescue.graph_version(v1) == "v1"


def test_f10_without_the_key_hand_made_entities_are_counted_and_refused(kg_store, tmp_path):
    out = tmp_path / "rescued"
    done = _rescue(kg_store, out, tmp_path, TEST_NO_KEY="1")
    text = done.stdout + done.stderr
    assert done.returncode != 0
    # Jellyfin (no memory names it) and Router (only a hand-added relation does).
    assert "2 hand-made entities" in text and "1 hand-added relation" in text
    assert "--drop-kg-hand-added" in text
    assert not out.exists() and not rescue.plaintext_path(out).exists()

    done = _rescue(kg_store, out, tmp_path, "--drop-kg-hand-added", TEST_NO_KEY="1")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "Jellyfin" not in _graph(out)["nodes"]


def test_f10_keyless_count_includes_customised_linked_nodes(tmp_path):
    kg = tmp_path / "v1.kg.db"
    con = sqlite3.connect(kg)
    con.execute("CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, nonce BLOB, "
                "props_enc BLOB, props_nonce BLOB)")
    con.execute("CREATE TABLE kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT, weight REAL, "
                "timestamp REAL, props_enc BLOB, props_nonce BLOB)")
    rows = [("skynas", "entity", b"x" * 18), ("docker", "tool", b"x" * 18),        # linked, retyped
            ("nas", "entity", b"x" * 40),                                           # linked, has props
            ("router", "entity", b"x" * 18), ("orphan", "entity", b"x" * 18)]
    for nid, type_, props in rows:
        con.execute("INSERT INTO kg_nodes VALUES (?,?,?,?,?,?)", (nid, type_, b"", b"", props, b""))
    con.commit()
    con.close()
    carried = [("router", "skynas", "routes_for", 1.0)]
    assert rescue.keyless_lost_nodes(kg, {"SkyNAS", "Docker", "NAS"}, carried) == 3


def test_f10_deploy_passes_drop_kg_hand_added_through_and_f3_removes_a_failed_rescues_plaintext(
        deploy_env, tmp_path):
    log = tmp_path / "python.log"
    wrapper = tmp_path / "python-wrapper"
    wrapper.write_text(f'''#!/bin/bash
case "$1" in
    */rescue-container-store.py)
        echo "$@" >> {log}
        printf '[{{"content": "secret"}}]' > "$2.plaintext.json"
        exit 1 ;;
esac
exec {PY} "$@"
''')
    wrapper.chmod(0o755)
    env = dict(deploy_env["env"], MEMCORE_PYTHON=str(wrapper))
    done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh"), "--drop-kg-hand-added"], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0
    assert "--drop-kg-hand-added" in log.read_text()
    backups = deploy_env["tmp"] / "backups"
    assert not list(backups.glob("*.plaintext.json"))
    assert "Removed the plaintext snapshot" in done.stderr
    assert not any(c.startswith("compose stop") for c in base._calls(deploy_env))

    done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh"), "--keep-plaintext"], cwd=REPO,
                          env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode != 0
    assert len(list(backups.glob("*.plaintext.json"))) == 1
    assert "Plaintext snapshot kept" in done.stderr


def test_deploy_help_prints_the_header():
    done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh"), "--help"], capture_output=True, text=True)
    assert done.returncode == 0
    assert "--drop-kg-hand-added" in done.stdout and "set -euo" not in done.stdout


# --- F16: a write between the rescue snapshot and the stop aborts the deploy ---

_INJECT = r'''#!PY
import json, os, sys, urllib.request
args = sys.argv[1:]
flag = os.environ["INJECT"]
if args[:2] == ["compose", "stop"] and not os.path.exists(flag + ".done"):
    open(flag + ".done", "w").close()
    spec = json.load(open(flag))
    body = json.dumps(spec["body"]).encode() if "body" in spec else None
    req = urllib.request.Request(os.environ["MEMCORE_ENDPOINT"] + spec["path"], data=body, method=spec["method"],
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=20).read()
os.execv(sys.executable, [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "docker-base")] + args)
'''


def _inject(deploy_env, spec):
    bindir = deploy_env["tmp"] / "bin"
    (bindir / "docker").rename(bindir / "docker-base")
    (bindir / "docker").write_text(_INJECT.replace("PY", PY, 1))
    (bindir / "docker").chmod(0o755)
    flag = deploy_env["tmp"] / "inject.json"
    flag.write_text(json.dumps(spec))
    return dict(deploy_env["env"], INJECT=str(flag))


@pytest.mark.parametrize("what", ["delete", "rehearse"])
def test_f16_deploy_aborts_when_the_store_changes_after_the_rescue(deploy_env, what):
    items = base._store_items(deploy_env["vol"])
    victim = next(i for i in items if i["content"] == "SkyNAS runs Docker")
    spec = ({"method": "DELETE", "path": f"/memory/{victim['id']}"} if what == "delete" else
            {"method": "POST", "path": "/recall", "body": {"query": "SkyNAS Docker", "k": 3}})
    env = _inject(deploy_env, spec)
    done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh")], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode != 0, done.stdout + done.stderr
    assert "changed between the rescue snapshot and the stop" in done.stderr
    assert ("deleted" if what == "delete" else "changed") in done.stderr
    calls = base._calls(deploy_env)
    assert not any("/src:ro" in c for c in calls)     # nothing staged
    assert any(c.startswith("compose start") for c in calls)
    after = base._store_items(deploy_env["vol"])
    if what == "delete":
        # The delete stands; staging would have brought the memory back.
        assert victim["id"] not in {i["id"] for i in after}
    else:
        assert sum(i["rehearsals"] for i in after) > sum(i["rehearsals"] for i in items)


# --- R5: SOURCE's key must decrypt SOURCE, checked before anything stops -------

def test_r5_deploy_refuses_a_source_whose_raw_key_does_not_match(deploy_env, tmp_path):
    src = tmp_path / "mismatched"
    base._seed(src)
    (src / "master.key").write_bytes(os.urandom(32))
    done = base._deploy(deploy_env, str(src))
    assert done.returncode != 0
    assert "is not the key" in done.stderr
    assert not any(c.startswith("compose build") or c.startswith("compose stop") for c in base._calls(deploy_env))


# --- F97: no assert-based verification ---------------------------------------

def test_f97_rescue_verification_survives_python_O():
    tree = ast.parse((SCRIPTS / "rescue-container-store.py").read_text())
    assert not [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Assert)]
    with pytest.raises(SystemExit) as e:
        rescue.check(False, "content mismatch")
    assert "content mismatch" in str(e.value)
    rescue.check(True, "never raised")


# --- F75: the snapshot includes commits still in the -wal ---------------------

def test_f75_backup_snapshot_includes_uncheckpointed_wal_commits(tmp_path):
    db = tmp_path / "memory.db"
    w = sqlite3.connect(db)
    w.execute("PRAGMA journal_mode=WAL")
    w.execute("PRAGMA wal_autocheckpoint=0")
    w.execute("CREATE TABLE memories (id TEXT PRIMARY KEY)")
    w.commit()
    w.execute("INSERT INTO memories VALUES ('in-wal')")
    w.commit()
    assert (tmp_path / "memory.db-wal").stat().st_size > 0
    try:
        subprocess.run([PY, "-c", rescue._BACKUP, str(db), str(tmp_path / "snap.db")], check=True)
    finally:
        w.close()
    got = sqlite3.connect(tmp_path / "snap.db").execute("SELECT id FROM memories").fetchall()
    assert got == [("in-wal",)]


# --- store_fingerprint.py ------------------------------------------------------

def _mini_store(d: Path):
    d.mkdir()
    con = sqlite3.connect(d / "memory.db")
    con.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, tier TEXT, forgetting_json TEXT, content_enc BLOB)")
    con.execute("CREATE TABLE store_meta (k TEXT PRIMARY KEY, v TEXT)")
    con.executemany("INSERT INTO memories VALUES (?,?,?,?)",
                    [("a", "working", '{"strength": 1}', b"\x01"), ("b", "episodic", '{"strength": 2}', b"\x02")])
    con.commit()
    return con


def test_fingerprint_names_every_kind_of_change(tmp_path):
    con = _mini_store(tmp_path / "s")
    before = fp.fingerprint(str(tmp_path / "s"))
    assert fp.ids(before) == ["a", "b"]
    assert fp.differences(before, fp.fingerprint(str(tmp_path / "s"))) == []

    con.execute("UPDATE memories SET forgetting_json='{\"strength\": 3}' WHERE id='a'")   # a rehearsal
    con.execute("DELETE FROM memories WHERE id='b'")
    con.execute("INSERT INTO memories VALUES ('c', 'working', NULL, x'03')")
    con.execute("INSERT INTO store_meta VALUES ('k', 'v')")
    con.commit()
    diff = "\n".join(fp.differences(before, fp.fingerprint(str(tmp_path / "s"))))
    assert "1 memory added, e.g. c" in diff
    assert "1 memory deleted, e.g. b" in diff
    assert "1 memory changed, e.g. a" in diff
    assert "table store_meta changed" in diff


def test_fingerprint_replays_a_leftover_wal_and_never_writes_the_source(tmp_path):
    d = tmp_path / "s"
    con = _mini_store(d)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("INSERT INTO memories VALUES ('w', 'working', NULL, x'04')")
    con.commit()
    files = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    try:
        assert "w" in fp.ids(fp.fingerprint(str(d)))
        # Read from a copy: the source and its -wal are byte for byte as they were.
        assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()} == files
    finally:
        con.close()


def test_fingerprint_cli_compare_exit_codes(tmp_path):
    _mini_store(tmp_path / "s").close()
    a = tmp_path / "a.json"
    a.write_text(json.dumps(fp.fingerprint(str(tmp_path / "s"))))
    same = subprocess.run([sys.executable, str(SCRIPTS / "store_fingerprint.py"), "--compare", str(a), str(a)],
                          capture_output=True, text=True)
    assert same.returncode == 0 and same.stdout == ""
    b = tmp_path / "b.json"
    b.write_text(json.dumps({}))
    differ = subprocess.run([sys.executable, str(SCRIPTS / "store_fingerprint.py"), "--compare", str(a), str(b)],
                            capture_output=True, text=True)
    assert differ.returncode == 1 and "memory.db is gone" in differ.stdout


# --- sync-rest-to-local.py: F33 and F57 --------------------------------------

def _seed_protected(data: Path):
    data.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "-c", base._SEED, base.LONG], check=True, cwd=REPO,
                   env=base._clean_env(MNEM_DATA_DIR=str(data), MNEM_MASTER_PASSWORD="pw"))


def _sync(store, *args, **env):
    e = base._clean_env(MNEM_MASTER_PASSWORD="pw", MEMCORE_ENDPOINT=store["url"], MEMCORE_PYTHON=PY, **env)
    return subprocess.run([PY, str(SCRIPTS / "sync-rest-to-local.py"), *args], cwd=REPO, env=e,
                          capture_output=True, text=True, timeout=180)


def test_f33_sync_refuses_a_data_dir_without_a_store_and_creates_nothing(tmp_path):
    typo = tmp_path / "memcroe"
    done = _sync({"url": "http://127.0.0.1:1"}, "--data-dir", str(typo))
    assert done.returncode != 0 and "no store at" in done.stderr and "--create" in done.stderr
    assert not typo.exists()


def test_f33_dry_run_never_starts_the_server_or_migrates(kg_store, tmp_path):
    local = tmp_path / "local"
    _seed_protected(local)
    (local / "memory.kg.db").unlink()          # a server init() would recreate it
    for f in local.glob("memory.kg.db-*"):
        f.unlink()
    digest = hashlib.sha256((local / "memory.db").read_bytes()).hexdigest()
    done = _sync(kg_store, "--dry-run", "--data-dir", str(local))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "would be added" in done.stdout and "nothing written" in done.stdout
    assert not (local / "memory.kg.db").exists()
    assert hashlib.sha256((local / "memory.db").read_bytes()).hexdigest() == digest


def test_f57_sync_carries_entities_and_reports_a_server_without_them(kg_store, tmp_path):
    local = tmp_path / "local"
    done = _sync(kg_store, "--create", "--data-dir", str(local))
    assert done.returncode == 0, done.stdout + done.stderr
    assert "without_entities" not in done.stdout and "WARNING" not in done.stdout
    items = {i["content"]: i for i in base._store_items(local, password="pw")}
    assert items["SkyNAS runs Docker"]["entities"] == ["SkyNAS", "Docker"]

    assert sync.without_entities([{"id": "a", "entities": []}, {"id": "b"}]) == 1
