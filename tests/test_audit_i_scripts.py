"""Audit group I: the operator scripts that guard the live store.

Nothing here touches docker, the live container or a real store. The scripts run
against a throwaway REST server on 127.0.0.1 serving a temp store, and
deploy-skynas.sh runs against a `docker` stub that plays the container: `exec` and
`run` are executed on the host with the container's paths mapped into temp dirs,
and `compose stop/up` stop and restart that throwaway server.
"""

import asyncio
import importlib.util
import json
import os
import signal
import socket
import sqlite3
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PY = sys.executable
LONG = "SkyNAS long record " + "x" * 700


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rescue = _load("rescue-container-store")
sync = _load("sync-rest-to-local")


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("MNEM_") and k != "MEMCORE_ENV"}
    env.update(extra)
    return env


# --- a store and a REST server over it ---------------------------------------

_SEED = r'''
import asyncio, sys
from pathlib import Path
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore

async def main():
    key = KeyManager(settings.key_path).load_or_create()
    c = AES256GCM(key)
    s = EncryptedStore(settings.db_path, c); await s.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), c); await kg.init()
    m = MnemosyneMemory(s, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)
    await m.add("SkyNAS runs Docker", entities=["SkyNAS", "Docker"], metadata={"project": "nas"}, tier="episodic")
    await m.add(sys.argv[1], importance=0.9, tier="semantic")
    await m.add("WireGuard on the router", tier="working")

asyncio.run(main())
'''


def _seed(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run([PY, "-c", _SEED, LONG], check=True, cwd=REPO,
                   env=_clean_env(MNEM_DATA_DIR=str(data_dir)))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.load(r)


def _start_server(data_dir: Path, port: int, log: Path):
    proc = subprocess.Popen(
        [PY, "-m", "memcore_memory.cli.main", "server", "start", "--host", "127.0.0.1", "--port", str(port)],
        cwd=REPO, env=_clean_env(MNEM_DATA_DIR=str(data_dir), MNEM_RATE_LIMIT_ENABLED="false"),
        stdout=open(log, "a"), stderr=subprocess.STDOUT, start_new_session=True)
    for _ in range(150):
        try:
            _get(f"http://127.0.0.1:{port}/health")
            return proc
        except Exception:
            if proc.poll() is not None:
                raise RuntimeError(log.read_text()[-2000:])
            time.sleep(0.2)
    proc.kill()
    raise RuntimeError("server did not come up: " + log.read_text()[-2000:])


def _stop(pid: int):
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.1)
    os.killpg(pid, signal.SIGKILL)


@pytest.fixture(scope="module")
def rest_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("rest")
    data = root / "data"
    _seed(data)
    port = _free_port()
    proc = _start_server(data, port, root / "server.log")
    yield {"data": data, "url": f"http://127.0.0.1:{port}"}
    _stop(proc.pid)


def _store_items(data_dir: Path, password=None):
    code = r'''
import asyncio, json, sys
from memcore_memory.config import settings
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
async def main():
    key = KeyManager(settings.key_path).load_or_create()
    items, bad = await EncryptedStore(settings.db_path, AES256GCM(key)).scan()
    print(json.dumps([{"id": i.id, "content": i.content, "metadata": i.metadata, "entities": i.entities,
                       "rehearsals": i.forgetting.rehearsals, "tier": i.tier.value} for i in items]))
asyncio.run(main())
'''
    env = _clean_env(MNEM_DATA_DIR=str(data_dir))
    if password:
        env["MNEM_MASTER_PASSWORD"] = password
    out = subprocess.run([PY, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


# --- rescue-container-store.py: pure parts -----------------------------------

def test_f58_rescue_refuses_an_existing_store_dir_and_never_deletes_it(tmp_path):
    store = tmp_path / "store"
    store.mkdir()
    (store / "master.key").write_bytes(b"k" * 32)
    (store / "memory.db").write_bytes(b"db")
    with pytest.raises(SystemExit) as e:
        rescue.check_output_dir(store)
    assert "holds a store" in str(e.value)
    assert (store / "master.key").read_bytes() == b"k" * 32

    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("x")
    with pytest.raises(SystemExit):
        rescue.check_output_dir(other)
    assert (other / "notes.txt").exists()

    empty = tmp_path / "empty"
    empty.mkdir()
    assert rescue.check_output_dir(empty) == empty.resolve()
    assert rescue.check_output_dir(tmp_path / "new") == (tmp_path / "new").resolve()


def test_f3_rescue_refuses_to_overwrite_a_plaintext_snapshot(tmp_path):
    rescue.plaintext_path(tmp_path / "out").write_text("[]")
    with pytest.raises(SystemExit):
        rescue.check_output_dir(tmp_path / "out")


def test_r27_scrub_env_drops_every_mnem_override():
    env = {"MNEM_DB_PATH": "/live/memory.db", "MNEM_KEY_PATH": "/live/master.key",
           "MNEM_MASTER_PASSWORD": "pw", "HOME": "/home/x", "PATH": "/bin"}
    removed = rescue.scrub_env(env)
    assert env == {"HOME": "/home/x", "PATH": "/bin"}
    assert set(removed) == {"MNEM_DB_PATH", "MNEM_KEY_PATH", "MNEM_MASTER_PASSWORD"}


def test_f10_hand_added_edges_v1_are_carried_and_v2_are_counted(tmp_path):
    v1 = tmp_path / "v1.kg.db"
    con = sqlite3.connect(v1)
    con.execute("CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, nonce BLOB, "
                "props_enc BLOB, props_nonce BLOB)")
    con.execute("CREATE TABLE kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT, weight REAL, "
                "timestamp REAL, props_enc BLOB, props_nonce BLOB)")
    con.execute("INSERT INTO kg_edges VALUES ('1','skynas','docker','co_occurs',1.0,0,NULL,NULL)")
    con.execute("INSERT INTO kg_edges VALUES ('2','skynas','wireguard','runs',0.5,0,NULL,NULL)")
    con.commit()
    con.close()
    assert rescue.graph_version(v1) == "v1"
    assert rescue.hand_added_edges(v1) == ([("skynas", "wireguard", "runs", 0.5)], 0)

    v2 = tmp_path / "v2.kg.db"
    con = sqlite3.connect(v2)
    con.execute("CREATE TABLE kg_edges (id TEXT, src TEXT, dst TEXT, relation TEXT, weight REAL, "
                "timestamp REAL, props_enc BLOB, props_nonce BLOB, memory_id TEXT)")
    con.execute("CREATE TABLE kg_meta (k TEXT PRIMARY KEY, v TEXT)")
    con.execute("INSERT INTO kg_meta VALUES ('schema_v', '2')")
    con.execute("INSERT INTO kg_edges VALUES ('1','h1','h2',NULL,1.0,0,NULL,NULL,'mem-1')")
    con.execute("INSERT INTO kg_edges VALUES ('2','h1','h3',NULL,1.0,0,NULL,NULL,NULL)")
    con.commit()
    con.close()
    assert rescue.hand_added_edges(v2) == ([], 1)
    assert rescue.hand_added_edges(tmp_path / "absent.kg.db") == ([], 0)


# --- rescue-container-store.py end to end ------------------------------------

# Runs the rescue with the container replaced: the snapshot is taken with sqlite's
# backup API straight from the served store, which is what the real one does
# inside the container.
_RESCUE_RUNNER = r'''
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
m.snapshot = snapshot
m.find_data_dir = lambda c: "/data"
# Never the real `docker cp`: that would copy the live container's master.key.
def fetch_key(container, path, dst):
    if not (src / "master.key").exists(): return False
    shutil.copyfile(src / "master.key", dst); return True
m.fetch_key = fetch_key
asyncio.run(m.main(sys.argv[3:]))
'''


def _run_rescue(rest_store, out, *flags, **env):
    return subprocess.run(
        [PY, "-c", _RESCUE_RUNNER, str(SCRIPTS / "rescue-container-store.py"), str(rest_store["data"]),
         str(out), "--endpoint", rest_store["url"], *flags],
        cwd=REPO, env=_clean_env(**env), capture_output=True, text=True)


def test_rescue_end_to_end_is_complete_private_and_ignores_inherited_paths(rest_store, tmp_path):
    decoy = tmp_path / "decoy"
    out = tmp_path / "backups" / "rescued"
    done = _run_rescue(rest_store, out, MNEM_DB_PATH=str(decoy / "memory.db"),
                       MNEM_KEY_PATH=str(decoy / "master.key"), MNEM_MASTER_PASSWORD="pw",
                       MNEM_REMOTE_URL="http://127.0.0.1:1")
    assert done.returncode == 0, done.stdout + done.stderr
    # R27: nothing went where the inherited variables pointed.
    assert not decoy.exists()
    # The key is raw (the container has no password) and private (F3).
    key = out / "master.key"
    assert key.stat().st_size == 32
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(out.stat().st_mode) == 0o700
    # F3: the plaintext snapshot is gone once the store verified.
    assert not rescue.plaintext_path(out).exists()

    items = {i["content"]: i for i in _store_items(out)}
    source = {i["content"]: i for i in _store_items(rest_store["data"])}
    assert set(items) == set(source)
    assert LONG in items                                  # full content, not the 500-char listing
    assert items["SkyNAS runs Docker"]["metadata"]["project"] == "nas"
    assert {i["id"] for i in items.values()} == {i["id"] for i in source.values()}

    # F10: the graph is rebuilt from entities_json rather than left empty.
    con = sqlite3.connect(out / "memory.kg.db")
    assert con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM memory_entities").fetchone()[0] == 2
    con.close()
    # F97: the vector sidecar exists, one vector per memory, tier-only metadata.
    sidecar = json.loads((out / "vectors.vectors.json").read_text())
    assert "x" * 50 not in json.dumps(sidecar)
    assert len(sidecar["ids"]) == 3


def test_rescue_keep_plaintext_leaves_a_0600_snapshot(rest_store, tmp_path):
    out = tmp_path / "rescued"
    done = _run_rescue(rest_store, out, "--keep-plaintext")
    assert done.returncode == 0, done.stdout + done.stderr
    snap = rescue.plaintext_path(out)
    assert stat.S_IMODE(snap.stat().st_mode) == 0o600
    assert len(json.loads(snap.read_text())) == 3


def test_f10_rescue_refuses_to_drop_unreadable_hand_added_relations(rest_store, tmp_path):
    kg = tmp_path / "foreign.kg.db"
    con = sqlite3.connect(kg)
    con.execute("CREATE TABLE kg_edges (id TEXT, src TEXT, dst TEXT, relation TEXT, weight REAL, "
                "timestamp REAL, props_enc BLOB, props_nonce BLOB, memory_id TEXT)")
    con.execute("CREATE TABLE kg_meta (k TEXT PRIMARY KEY, v TEXT)")
    con.execute("INSERT INTO kg_meta VALUES ('schema_v', '2')")
    con.execute("INSERT INTO kg_edges VALUES ('1','h1','h3',NULL,1.0,0,NULL,NULL,NULL)")
    con.commit()
    con.close()
    out = tmp_path / "rescued"
    done = _run_rescue(rest_store, out, TEST_KG_SOURCE=str(kg))
    assert done.returncode != 0
    assert "--drop-kg-hand-added" in done.stdout + done.stderr
    assert not out.exists() and not rescue.plaintext_path(out).exists()
    done = _run_rescue(rest_store, out, "--drop-kg-relations", TEST_KG_SOURCE=str(kg))
    assert done.returncode == 0, done.stdout + done.stderr


def test_r27_rescue_refuses_prod_before_writing_anything(rest_store, tmp_path):
    out = tmp_path / "rescued"
    done = _run_rescue(rest_store, out, MEMCORE_ENV="prod")
    assert done.returncode != 0 and "prod" in done.stdout + done.stderr
    assert not out.exists()


def test_f97_rescue_refuses_a_partial_snapshot(rest_store, tmp_path):
    # A snapshot of an empty store while the API reports three.
    empty = tmp_path / "empty"
    _seed_empty = r'''
import sqlite3, sys
c = sqlite3.connect(sys.argv[1])
c.execute("CREATE TABLE memories (id TEXT PRIMARY KEY, tier TEXT, timestamp REAL, content_enc BLOB, nonce BLOB, metadata_enc BLOB, meta_nonce BLOB, forgetting_json TEXT, entities_json TEXT, embedding BLOB)")
c.commit()
'''
    empty.mkdir()
    subprocess.run([PY, "-c", _seed_empty, str(empty / "memory.db")], check=True)
    out = tmp_path / "rescued"
    done = subprocess.run(
        [PY, "-c", _RESCUE_RUNNER, str(SCRIPTS / "rescue-container-store.py"), str(empty),
         str(out), "--endpoint", rest_store["url"]],
        cwd=REPO, env=_clean_env(), capture_output=True, text=True)
    assert done.returncode != 0 and "--allow-partial" in done.stdout + done.stderr
    assert not out.exists()


# --- sync-rest-to-local.py ---------------------------------------------------

def test_f57_plan_matches_on_origin_id_then_content():
    source = [{"id": "a", "content": "same"}, {"id": "b", "content": "new text"},
              {"id": "c", "content": "legacy"}, {"id": "d", "content": "never seen"}]
    local = [{"id": "L1", "content": "same", "origin_id": "a"},
             {"id": "L2", "content": "old text", "origin_id": "b"},
             {"id": "L3", "content": "legacy", "origin_id": None}]
    present, changed, missing = sync.plan(source, local)
    assert [r["id"] for r in present] == ["a", "c"]
    assert [(r["id"], lid) for r, lid in changed] == [("b", "L2")]
    assert [r["id"] for r in missing] == ["d"]
    meta = sync.import_metadata({"id": "a", "metadata": {"project": "x", "origin_id": "spoof"}})
    assert meta == {"project": "x", "imported_from": "docker-rest-store", "origin_id": "a"}


def test_r4_child_env_drops_inherited_mnem_vars_but_keeps_the_password(tmp_path):
    env = sync.child_env({"MNEM_REMOTE_URL": "http://x", "MNEM_DB_PATH": "/elsewhere",
                          "MNEM_MASTER_PASSWORD": "pw", "PATH": "/bin"}, tmp_path)
    assert env == {"MNEM_MASTER_PASSWORD": "pw", "PATH": "/bin", "MNEM_DATA_DIR": str(tmp_path)}


def test_f96_duplicates_and_prefix_argument():
    assert sync.duplicates([{"content": "a"}, {"content": "a"}, {"content": "b"}]) == 1
    done = subprocess.run([PY, str(SCRIPTS / "sync-rest-to-local.py"), "--prefix"],
                          capture_output=True, text=True, env=_clean_env())
    assert done.returncode == 2 and "Traceback" not in done.stderr


def test_r4_f33_f57_sync_writes_the_named_store_without_rehearsing(rest_store, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = _clean_env(HOME=str(home), MNEM_MASTER_PASSWORD="pw", MEMCORE_ENDPOINT=rest_store["url"],
                     MEMCORE_PYTHON=PY, MNEM_REMOTE_URL=rest_store["url"],
                     MNEM_DATA_DIR=str(tmp_path / "elsewhere"))
    script = str(SCRIPTS / "sync-rest-to-local.py")
    first = subprocess.run([PY, script, "--create"], cwd=REPO, env=env, capture_output=True, text=True,
                           timeout=180)
    assert first.returncode == 0, first.stdout + first.stderr
    assert "added=3" in first.stdout
    # R4: the bridge variable did not turn the local store into the REST store,
    # and MNEM_DATA_DIR did not move it either: the default ~/.memcore was written.
    local = _store_items(home / ".memcore", password="pw")
    assert len(local) == 3
    assert not (tmp_path / "elsewhere").exists()

    by_content = {i["content"]: i for i in local}
    assert LONG in by_content
    # F57: source metadata and identity carried across.
    tagged = by_content["SkyNAS runs Docker"]
    assert tagged["metadata"]["project"] == "nas"
    assert tagged["metadata"]["imported_from"] == "docker-rest-store"
    assert tagged["metadata"]["origin_id"] in {i["id"] for i in _store_items(rest_store["data"])}

    second = subprocess.run([PY, script], cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    assert second.returncode == 0, second.stdout + second.stderr
    assert "added=0" in second.stdout
    # F33: two full syncs, and not one local memory was rehearsed by them.
    assert all(i["rehearsals"] == 0 for i in _store_items(home / ".memcore", password="pw"))


# --- deploy-skynas.sh against a docker stub ----------------------------------

_DOCKER_STUB = r'''#!PYTHON
import json, os, re, shutil, signal, subprocess, sys, time
from pathlib import Path
args = sys.argv[1:]
S = os.environ
with open(S["STUB_LOG"], "a") as f:
    f.write(" ".join(args) + "\n")
vol, ctmp = S["STUB_VOL"], S["STUB_CTMP"]

def remap(text, mounts):
    # One pass, so a mapped path is never mapped again.
    alt = "|".join(re.escape(d) for d in sorted(mounts, key=len, reverse=True))
    return re.sub(r"(?<![\w/])(" + alt + r")(?=/|\s|$|'|\")", lambda m: mounts[m.group(1)], text)

def container(text):
    return remap(text, {"/data": vol, "/tmp": ctmp})

if args[0] == "compose":
    rest = [a for a in args[1:] if a not in ("--no-build", "--force-recreate", "-d")]
    if rest[:1] == ["config"]:
        print(json.dumps({"volumes": {"mnem_data": {"name": "testvol"}}}))
    elif rest[:1] == ["stop"]:
        pid = int(Path(S["STUB_PIDFILE"]).read_text())
        os.killpg(pid, signal.SIGTERM)
        for _ in range(100):
            try: os.kill(pid, 0); time.sleep(0.1)
            except ProcessLookupError: break
    elif rest[:1] in (["up"], ["start"]):
        proc = subprocess.Popen([sys.executable, "-m", "memcore_memory.cli.main", "server", "start",
                                 "--host", "127.0.0.1", "--port", S["STUB_PORT"]],
                                cwd=S["STUB_REPO"], start_new_session=True,
                                stdout=open(S["STUB_SERVER_LOG"], "a"), stderr=subprocess.STDOUT,
                                env={k: v for k, v in S.items() if not k.startswith("MNEM_")} |
                                    {"MNEM_DATA_DIR": vol, "MNEM_RATE_LIMIT_ENABLED": "false"})
        Path(S["STUB_PIDFILE"]).write_text(str(proc.pid))
    sys.exit(0)
if args[:2] == ["volume", "inspect"]:
    sys.exit(0)
if args[0] == "inspect":
    fmt = args[args.index("--format") + 1]
    print("testvol" if "Mounts" in fmt else "sha256:stubimage")
    sys.exit(0)
if args[0] == "cp":
    shutil.copy(container(args[1].split(":", 1)[1]), args[2]); sys.exit(0)
if args[0] == "exec":
    cmd = [container(a) for a in args[2:]]
    if cmd[0] == "python":
        cmd[0] = sys.executable
    sys.exit(subprocess.run(cmd, env=dict(S, MNEM_DATA_DIR=vol)).returncode)
if args[0] == "run":
    mounts, i, entry = {}, 1, None
    while args[i].startswith("-"):
        if args[i] == "--rm": i += 1; continue
        if args[i] == "-v":
            src, dst = args[i + 1].split(":")[:2]
            mounts[dst] = vol if src == "testvol" else src
            i += 2; continue
        if args[i] == "--entrypoint":
            entry = args[i + 1]; i += 2; continue
        raise SystemExit("stub: unknown run flag " + args[i])
    image, cmd = args[i], [remap(a, mounts) for a in args[i + 1:]]
    if entry == "python":
        cmd = [sys.executable] + cmd
    sys.exit(subprocess.run(cmd).returncode)
raise SystemExit("stub: unhandled " + " ".join(args))
'''


@pytest.fixture
def deploy_env(tmp_path):
    vol = tmp_path / "volume"
    _seed(vol)
    port = _free_port()
    proc = _start_server(vol, port, tmp_path / "server.log")
    (tmp_path / "pid").write_text(str(proc.pid))
    bindir = tmp_path / "bin"
    bindir.mkdir()
    stub = bindir / "docker"
    stub.write_text(_DOCKER_STUB.replace("PYTHON", PY))
    stub.chmod(0o755)
    (tmp_path / "ctmp").mkdir()
    env = _clean_env(
        PATH=f"{bindir}:{os.environ['PATH']}", MEMCORE_ENDPOINT=f"http://127.0.0.1:{port}",
        MEMCORE_BACKUP_ROOT=str(tmp_path / "backups"), MEMCORE_PYTHON=PY,
        STUB_LOG=str(tmp_path / "docker.log"), STUB_VOL=str(vol), STUB_CTMP=str(tmp_path / "ctmp"),
        STUB_PIDFILE=str(tmp_path / "pid"), STUB_PORT=str(port), STUB_REPO=str(REPO),
        STUB_SERVER_LOG=str(tmp_path / "server.log"))
    yield {"env": env, "vol": vol, "tmp": tmp_path, "log": tmp_path / "docker.log"}
    _stop(int((tmp_path / "pid").read_text()))


def _deploy(deploy_env, *args):
    return subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh"), *args], cwd=REPO,
                          env=deploy_env["env"], capture_output=True, text=True, timeout=300)


def _calls(deploy_env):
    return deploy_env["log"].read_text().splitlines() if deploy_env["log"].exists() else []


def test_f16_deploy_stops_before_staging_and_recreates_on_the_staged_store(deploy_env):
    before = {i["id"] for i in _store_items(deploy_env["vol"])}
    done = _deploy(deploy_env)
    assert done.returncode == 0, done.stdout + done.stderr
    calls = _calls(deploy_env)

    def first(pred):
        return next(n for n, c in enumerate(calls) if pred(c))
    build = first(lambda c: c.startswith("compose build"))
    stop = first(lambda c: c.startswith("compose stop"))
    tar = first(lambda c: c.startswith("run") and "tar czf" in c)
    stage = first(lambda c: c.startswith("run") and "/src:ro" in c)
    up = first(lambda c: c.startswith("compose up"))
    assert build < stop < tar < stage < up
    assert "--force-recreate" in calls[up]
    # The volume now holds the rescued store, under its new key, with every id.
    assert {i["id"] for i in _store_items(deploy_env["vol"])} == before
    backups = deploy_env["tmp"] / "backups"
    rescued = next(backups.glob("*-recovered-store"))
    assert (deploy_env["vol"] / "master.key").read_bytes() == (rescued / "master.key").read_bytes()
    assert next(backups.glob("*-volume-before-deploy.tgz")).stat().st_size > 0
    # F3: the plaintext copy did not outlive the verified deploy.
    assert not list(backups.glob("*.plaintext.json"))


def test_r5_deploy_refuses_a_source_that_lacks_live_ids_and_never_stops(deploy_env, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    _seed(other)   # same contents, different ids: counts match, ids do not
    done = _deploy(deploy_env, str(other))
    assert done.returncode != 0
    assert "live id(s) are not in the store to stage" in done.stderr
    calls = _calls(deploy_env)
    assert not any(c.startswith("compose stop") for c in calls)
    assert not any("/src:ro" in c for c in calls)
    # The rescue still ran first, and its plaintext did not outlive the abort.
    backups = tmp_path / "backups"
    assert list(backups.glob("*-recovered-store"))
    assert not list(backups.glob("*.plaintext.json"))


def test_r5_deploy_refuses_a_password_wrapped_source_key(deploy_env, tmp_path):
    src = tmp_path / "wrapped"
    src.mkdir()
    (src / "memory.db").write_bytes(b"")
    (src / "master.key").write_text(json.dumps({"salt": "00", "nonce": "00", "ct": "00"}))
    done = _deploy(deploy_env, str(src))
    assert done.returncode != 0 and "not a raw 32-byte key" in done.stderr
    assert not any(c.startswith("compose") and "config" not in c for c in _calls(deploy_env))


def test_f98_deploy_resolves_a_relative_source(deploy_env, tmp_path):
    rel = os.path.relpath(tmp_path / "missing", REPO)
    done = _deploy(deploy_env, rel)
    assert done.returncode != 0 and "cannot resolve SOURCE" in done.stderr


def test_f98_deploy_fails_closed_on_health_without_tier_counts(deploy_env, tmp_path):
    web = tmp_path / "web"
    web.mkdir()
    (web / "health").write_text('{"status": "ok"}')
    port = _free_port()
    srv = subprocess.Popen([PY, "-m", "http.server", str(port), "--bind", "127.0.0.1", "-d", str(web)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(50):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
        env = dict(deploy_env["env"], MEMCORE_ENDPOINT=f"http://127.0.0.1:{port}")
        done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh")], cwd=REPO, env=env,
                              capture_output=True, text=True, timeout=60)
    finally:
        srv.terminate()
    assert done.returncode != 0 and "tier_counts" in done.stderr
    assert not any(c.startswith("compose build") for c in _calls(deploy_env))


def test_f59_deploy_refuses_when_the_container_serves_another_volume(deploy_env):
    env = dict(deploy_env["env"], MEMCORE_VOLUME="othervol")
    done = subprocess.run(["bash", str(SCRIPTS / "deploy-skynas.sh")], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=60)
    assert done.returncode != 0 and "different store" in done.stderr
    assert not any(c.startswith("compose build") for c in _calls(deploy_env))
