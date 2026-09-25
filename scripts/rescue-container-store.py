#!/usr/bin/env python3
"""Rebuild a deployable copy of the container's store under a fresh key.

Why re-encrypt rather than copy: this script was written while the container's
on-disk master.key no longer decrypted its own database (see CLAUDE.md, master-key
bug), so the running process was the only decryption oracle left. Since the 16:39
deploy the key on disk is valid again, but a rescue is still the one procedure that
works whatever state the key file is in, so it keeps using the API.

What makes that lossless: `id`, `tier`, `timestamp`, `forgetting_json` and
`embedding` are plaintext columns of memory.db, so they come straight out of a
snapshot of the database and are carried across verbatim - including the
Ebbinghaus curve and the stored vector. `content`, `metadata` and, in stores
written by the current code, `entities` are encrypted, and are read back through
GET /memory/{id}. Older stores kept entities in the plaintext entities_json, which
is used when the API does not return them; a row whose entities are encrypted and
that the API does not return refuses the rescue.

Content is read through GET /memory/{id}, never the /memories listing, which cuts
content at 500 characters. Importing from that endpoint silently stores truncated
records.

The knowledge graph is rebuilt from each memory's entities. What that cannot
recover - entities added by hand (with their type and props), and relations added
by hand - is carried across from the graph snapshot, read with the container's
own master.key once that key has proven itself by decrypting every row of the
snapshot. When the key cannot be used (missing, password-wrapped, or no longer
the key the rows were written under), a v1 graph's hand-added relations are still
readable, since v1 kept them in clear, but nothing else is. The rescue then counts
what it would drop and refuses unless --drop-kg-hand-added is given.

The vector sidecar is rebuilt from the stored embeddings.

Usage:
    rescue-container-store.py OUTPUT_DIR [--container NAME] [--endpoint URL]
                              [--keep-plaintext] [--allow-partial] [--drop-kg-hand-added]

OUTPUT_DIR must not exist or be empty; it is never deleted or overwritten. It ends
up a complete data dir (master.key, memory.db, memory.kg.db, vectors.vectors.json)
ready to be staged into the volume, plus source-fingerprint.json: a hash of every
row of the snapshot, which deploy-skynas.sh compares with the stopped volume to
catch writes that landed after the snapshot.

A plaintext JSON copy of every record is written to OUTPUT_DIR.plaintext.json
(0600) before anything is re-encrypted, so a crash mid-rescue loses nothing. It is
deleted once the new store verifies, unless --keep-plaintext is given; deploy-skynas.sh
keeps it until the deploy itself has verified, then deletes it.
"""

import argparse
import asyncio
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

# The store files this script produces, and so the only ones deploy-skynas.sh stages.
STORE_FILES = ("master.key", "memory.db", "memory.kg.db", "vectors.vectors.json")
FINGERPRINT_FILE = "source-fingerprint.json"
# An encrypted '{}': two bytes of plaintext plus the 16-byte GCM tag. Anything
# longer is a node that carries props.
EMPTY_PROPS_CT = len(b"{}") + 16

# Read once, before scrub_env clears every MNEM_* variable from the environment.
_API_KEY = os.environ.get("MNEM_API_KEY")


def _fingerprint_module():
    spec = importlib.util.spec_from_file_location("store_fingerprint",
                                                  Path(__file__).resolve().with_name("store_fingerprint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--container", default="mnemosyne")
    p.add_argument("--endpoint", default="http://192.168.1.183:8000")
    p.add_argument("--data-dir", default=None,
                   help="Store path inside the container (default: probe for it)")
    p.add_argument("--keep-plaintext", action="store_true",
                   help="keep OUTPUT_DIR.plaintext.json after the new store verifies")
    p.add_argument("--allow-partial", action="store_true",
                   help="continue when the snapshot holds fewer records than the API reports")
    # --drop-kg-relations is the round-1 name, from when relations were the only
    # thing a rescue could lose.
    p.add_argument("--drop-kg-hand-added", "--drop-kg-relations", dest="drop_kg", action="store_true",
                   help="continue when the graph holds hand-added entities or relations that cannot be carried")
    return p.parse_args(argv)


def check(ok, message):
    # Not assert: python -O / PYTHONOPTIMIZE strips those, and the rescue would then
    # report "verified" without having checked anything.
    if not ok:
        sys.exit(f"verification failed: {message}")


def plaintext_path(out: Path) -> Path:
    """Where the plaintext snapshot of a rescue into `out` goes. deploy-skynas.sh relies on this."""
    return out.with_name(out.name + ".plaintext.json")


def check_output_dir(out: Path) -> Path:
    """Resolve OUTPUT_DIR and refuse anything that is not absent or an empty directory.

    This used to rmtree whatever was there, so `rescue-container-store.py ~/.memcore`
    deleted that store and its key before a single record was written. A rescue is
    cheap to redo into a new path, so there is no override.
    """
    out = out.expanduser().resolve()
    if out.exists():
        if not out.is_dir():
            sys.exit(f"{out} exists and is not a directory - choose a new path")
        if any(out.iterdir()):
            what = [n for n in STORE_FILES if (out / n).exists()]
            sys.exit(f"{out} is not empty{' and holds a store (' + ', '.join(what) + ')' if what else ''}"
                     f" - refusing to touch it; choose a new directory")
    snap = plaintext_path(out)
    if snap.exists():
        sys.exit(f"{snap} already exists - refusing to overwrite it; choose a new directory")
    return out


def scrub_env(environ) -> dict:
    """Remove the inherited variables that would redirect or reshape the new store.

    Settings lets MNEM_DB_PATH / MNEM_KEY_PATH / ... win over the data dir, so an
    operator shell that exported them made the rescue write into, and verify
    against, some other store. MNEM_MASTER_PASSWORD would wrap the new key, which
    the container (no password in docker-compose.yml) could not unlock. Returns
    what was removed, for the log.
    """
    removed = {}
    for name in list(environ):
        if name.upper().startswith("MNEM_"):
            removed[name] = environ.pop(name)
    return removed


def graph_version(kg_db: Path):
    """'v1' (plaintext names), 'v2' (hashed ids, any later schema too) or None when there is no graph."""
    if not kg_db.exists():
        return None
    con = sqlite3.connect(f"{kg_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "kg_edges" not in tables:
            return None
        if "kg_meta" in tables and con.execute("SELECT v FROM kg_meta WHERE k='schema_v'").fetchone():
            return "v2"
        return "v1"
    finally:
        con.close()


def _columns(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def hand_added_edges(kg_db: Path):
    """(carryable, unreadable) hand-added relations, read without the graph's key.

    carryable is a list of (src, dst, relation, weight) that add_relation can
    replay. v1 kept names and relation in plaintext; co_occurs edges are skipped
    because add_memory_entities regenerates them from the memories. The names are
    the lowercased v1 ids, since the spelling is only in the encrypted label. In v2
    names are HMACs and relations are encrypted under the old key, so the
    hand-added ones (memory_id IS NULL) can only be counted, never carried.
    """
    version = graph_version(kg_db)
    if version is None:
        return [], 0
    con = sqlite3.connect(f"{kg_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        if version == "v1":
            rows = con.execute("SELECT src, dst, relation, weight FROM kg_edges "
                               "WHERE relation IS NOT NULL AND relation != 'co_occurs'").fetchall()
            return [(s, d, r, 1.0 if w is None else float(w)) for s, d, r, w in rows if s and d], 0
        n = con.execute("SELECT COUNT(*) FROM kg_edges WHERE memory_id IS NULL").fetchone()[0]
        return [], n
    finally:
        con.close()


def keyless_lost_nodes(kg_db: Path, entity_names, carried_edges) -> int:
    """Nodes a rebuild without the graph's key would lose or strip, counted only.

    A rebuild recreates exactly the nodes the memories name, as type 'entity'
    with no props. Lost is therefore every node made by hand that no memory names
    (and, in v1, no carried relation touches), plus every node whose type or props
    were set by hand. Nodes that memory linking created are regenerated whenever
    they still matter.
    """
    version = graph_version(kg_db)
    if version is None:
        return 0
    con = sqlite3.connect(f"{kg_db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        if "kg_nodes" not in {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}:
            return 0
        cols = _columns(con, "kg_nodes")
        auto = "auto" if "auto" in cols else "NULL"
        nodes = con.execute(f"SELECT id, type, length(props_enc), {auto} FROM kg_nodes").fetchall()
        if version == "v1":
            kept = {n.lower() for n in entity_names} | {x for s, d, *_ in carried_edges for x in (s, d)}
        else:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            kept = set()
            if "memory_entities" in tables:
                kept |= {r[0] for r in con.execute("SELECT node_id FROM memory_entities")}
            if "memory_id" in _columns(con, "kg_edges"):
                for s, d in con.execute("SELECT src, dst FROM kg_edges WHERE memory_id IS NOT NULL"):
                    kept |= {s, d}
        lost = 0
        for nid, type_, props_len, is_auto in nodes:
            if is_auto == 1:
                continue
            customised = (type_ or "entity") != "entity" or (props_len or 0) > EMPTY_PROPS_CT
            if nid not in kept or customised:
                lost += 1
        return lost
    finally:
        con.close()


async def read_graph_with_key(tmp: Path, key: bytes, records):
    """Everything hand-made in the graph snapshot, read with the container's key.

    Returns (nodes, edges, lost), or raises with the reason the key cannot be used.
    nodes are (label, type, props) of every node that memory linking did not create,
    edges are (src, dst, relation, weight) of every relation without a memory, and
    lost counts hand-added relations whose endpoints cannot be read.

    The key is trusted only after it decrypts every row of the memory snapshot to
    the content the API returned. The graph snapshot is a private copy, so
    KnowledgeGraph.init() may upgrade a v1 or v2 file in place; after that every
    row is in the current format. The row decoders are KnowledgeGraph's own: a
    second copy of the AAD layout here would drift from it.
    """
    from memcore_memory.crypto.aes_gcm import AES256GCM
    from memcore_memory.graph import kg as kgmod
    from memcore_memory.storage.encrypted_sqlite import EncryptedStore

    if len(key) != 32:
        raise ValueError(f"master.key is {len(key)} bytes, not a raw key (password-wrapped?)")
    cipher = AES256GCM(key)
    items, unreadable = await EncryptedStore(tmp / "memory.db", cipher, blind_index=False).scan()
    if unreadable:
        raise ValueError(f"it does not decrypt {len(unreadable)} row(s) of the snapshot")
    by_id = {i.id: i.content for i in items}
    if any(by_id.get(row["id"]) != d["content"] for row, d in records):
        raise ValueError("the rows it decrypts do not match what the API serves")

    graph = kgmod.KnowledgeGraph(tmp / "memory.kg.db", cipher)
    await graph.init()
    con = sqlite3.connect(tmp / "memory.kg.db")
    try:
        labels, nodes, lost = {}, [], 0
        for nid, type_, label_enc, nonce, props_enc, props_nonce, aad_v, auto in con.execute(
                "SELECT id, type, label_enc, nonce, props_enc, props_nonce, aad_v, auto FROM kg_nodes"):
            try:
                label = graph._label(nid, label_enc, nonce, aad_v)
                props = graph._dec_json(props_nonce, props_enc, kgmod._node_aad(nid, "props") if aad_v else b"")
            except Exception:  # noqa: BLE001 - counted below if it mattered
                if auto != 1:
                    lost += 1
                continue
            labels[nid] = label
            if auto != 1:
                nodes.append((label, type_ or "entity", props))
        edges = []
        for eid, src, dst, weight, props_enc, props_nonce, aad_v in con.execute(
                "SELECT id, src, dst, weight, props_enc, props_nonce, aad_v FROM kg_edges WHERE memory_id IS NULL"):
            try:
                if src not in labels or dst not in labels:
                    raise KeyError("endpoint without a readable label")
                props = graph._edge_props(eid, src, dst, props_enc, props_nonce, aad_v)
            except Exception:  # noqa: BLE001
                lost += 1
                continue
            edges.append((labels[src], labels[dst], props.get("relation", "related_to"),
                          1.0 if weight is None else float(weight)))
        return nodes, edges, lost
    finally:
        con.close()


CANDIDATE_DIRS = ("/data", "/root/.memcore")


def find_data_dir(container: str) -> str:
    # Probed, not assumed: assuming is what made the first run of this script abort
    # after the deploy had already moved the store.
    for candidate in CANDIDATE_DIRS:
        probe = subprocess.run(
            ["docker", "exec", container, "ls", f"{candidate}/memory.db"],
            capture_output=True,
        )
        if probe.returncode == 0:
            return candidate
    sys.exit(f"no memory.db in any of {CANDIDATE_DIRS} inside {container}")


# sqlite's backup API, run inside the container: a consistent snapshot even while
# the server writes, and it includes whatever is still in the -wal, because it
# reads through a connection rather than copying files. A docker cp of memory.db
# alone could be torn mid-checkpoint or miss the -wal entirely.
_BACKUP = ("import sqlite3,sys\n"
           "s=sqlite3.connect('file:'+sys.argv[1]+'?mode=ro',uri=True)\n"
           "d=sqlite3.connect(sys.argv[2]); s.backup(d); d.close(); s.close()\n")


def snapshot(container: str, src: str, dst: Path, required: bool) -> bool:
    inside = f"/tmp/rescue-{os.getpid()}-{Path(src).name}"
    made = subprocess.run(["docker", "exec", container, "python", "-c", _BACKUP, src, inside],
                          capture_output=True, text=True)
    try:
        if made.returncode != 0:
            if required:
                sys.exit(f"could not snapshot {container}:{src}: {made.stderr.strip()[-300:]}")
            return False
        subprocess.run(["docker", "cp", f"{container}:{inside}", str(dst)], check=True)
        return True
    finally:
        subprocess.run(["docker", "exec", container, "rm", "-f", inside], capture_output=True)


def fetch_key(container: str, src: str, dst: Path) -> bool:
    """Copy the container's master.key into the (0700) temp dir; False when there is none."""
    done = subprocess.run(["docker", "cp", f"{container}:{src}", str(dst)], capture_output=True)
    return done.returncode == 0 and dst.is_file()


def _headers():
    return {"Authorization": f"Bearer {_API_KEY}"} if _API_KEY else {}


def fetch(url, timeout=20):
    req = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def write_private(path: Path, text: str):
    # O_EXCL: never follow or replace something already at that path.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)


def entities_of(row, d):
    """The entities to carry for one record, or None when they cannot be read.

    The API is the source when it returns them. Older servers do not, and older
    stores kept them in the plaintext entities_json; the current store encrypts
    them into entities_enc and leaves entities_json NULL.
    """
    if isinstance(d.get("entities"), list):
        return d["entities"]
    if row["entities_json"]:
        return json.loads(row["entities_json"])
    if "entities_enc" in row.keys() and row["entities_enc"] is not None:
        return None
    return []


async def main(argv=None):
    args = parse_args(argv)
    # Everything this creates - the output dir, the key, both databases, the
    # sidecar, the plaintext snapshot - is for the owner's eyes only.
    os.umask(0o077)
    out = check_output_dir(args.output_dir)

    # Read here rather than through key_manager.deployment_env: importing the package
    # builds its settings singleton from the environment this script is about to scrub.
    env_name, env_value = next(((n, os.environ[n].lower()) for n in ("MNEM_ENV", "MEMCORE_ENV")
                                if os.environ.get(n)), ("", ""))
    if env_value in ("prod", "production"):
        # The rescue creates an unprotected key because the container has no password
        # to unlock any other kind; prod refuses exactly that. Checked before anything
        # is written, not half-way through.
        sys.exit(f"{env_name}={env_value}: a rescue creates an unprotected master key for the "
                 f"container, which prod refuses. Unset {env_name} to run it deliberately.")

    print(f"== endpoint {args.endpoint}")
    health = fetch(f"{args.endpoint}/health")
    tier_counts = health.get("tier_counts")
    if not isinstance(tier_counts, dict):
        sys.exit("/health has no tier_counts - cannot tell how many records must be rescued")
    live_total = sum(tier_counts.values())
    print(f"   health {health.get('status')}, {live_total} memories live")

    data_dir = args.data_dir or find_data_dir(args.container)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        print(f"== snapshotting the databases in {args.container}:{data_dir}")
        snapshot(args.container, f"{data_dir}/memory.db", tmp / "memory.db", required=True)
        have_kg = snapshot(args.container, f"{data_dir}/memory.kg.db", tmp / "memory.kg.db", required=False)
        # Before anything opens the snapshot: KnowledgeGraph.init() below upgrades
        # an old graph in place.
        source_fp = _fingerprint_module().fingerprint(str(tmp))
        have_key = fetch_key(args.container, f"{data_dir}/master.key", tmp / "old.key")

        db = sqlite3.connect(tmp / "memory.db")
        db.row_factory = sqlite3.Row
        rows = list(db.execute("SELECT * FROM memories ORDER BY timestamp"))
        db.close()
        print(f"   {len(rows)} rows on disk")

        if len(rows) < live_total:
            print(f"   WARNING: API reports {live_total} but the snapshot has only {len(rows)}")
            if not args.allow_partial:
                sys.exit("refusing to rescue a snapshot smaller than the live store "
                         "(--allow-partial to override)")
        if live_total <= 500:
            # The listing caps at 500, but below that it names every live id, which
            # catches a snapshot that has the right count and the wrong records.
            listed = fetch(f"{args.endpoint}/memories?limit=500")
            missing = {m["id"] for m in listed} - {r["id"] for r in rows}
            if missing and not args.allow_partial:
                sys.exit(f"{len(missing)} live id(s) are not in the snapshot, e.g. {sorted(missing)[:3]}")

        print("== reading plaintext back through the API")
        records, failed = [], []
        for i, row in enumerate(rows, 1):
            try:
                d = fetch(f"{args.endpoint}/memory/{row['id']}")
            except Exception as e:  # noqa: BLE001 - want the id with the reason
                failed.append((row["id"], repr(e)))
                continue
            entities = entities_of(row, d)
            if entities is None:
                failed.append((row["id"], "entities are encrypted and the API does not return them"))
                continue
            records.append((row, d, entities))
            if i % 25 == 0:
                print(f"   {i}/{len(rows)}")

        if failed:
            print(f"\n{len(failed)} record(s) could not be read:")
            for mid, err in failed[:10]:
                print(f"   {mid} {err}")
            sys.exit("refusing to write a partial store")

        removed = scrub_env(os.environ)
        if removed:
            print(f"   ignoring inherited {', '.join(sorted(removed))}")

        # The graph: what the memories cannot regenerate, and whether it can be carried.
        carry_nodes, carry_edges, lost_nodes, lost_edges = [], [], 0, 0
        if have_kg:
            reason = "the container has no master.key" if not have_key else None
            if have_key:
                try:
                    carry_nodes, carry_edges, lost_edges = await read_graph_with_key(
                        tmp, (tmp / "old.key").read_bytes(), [(r, d) for r, d, _ in records])
                    print(f"   graph read with the container's key: {len(carry_nodes)} hand-made "
                          f"node(s), {len(carry_edges)} hand-added relation(s)")
                except Exception as e:  # noqa: BLE001 - any failure means: fall back
                    reason = f"{type(e).__name__}: {e}"
            if reason:
                print(f"   the container's key cannot read the graph ({reason}); "
                      f"only a v1 graph's relations can be carried")
                carry_edges, lost_edges = hand_added_edges(tmp / "memory.kg.db")
                names = {n for _, _, ents in records for n in ents if isinstance(n, str)}
                lost_nodes = keyless_lost_nodes(tmp / "memory.kg.db", names, carry_edges)
        if lost_nodes or lost_edges:
            print(f"   WARNING: {lost_nodes} hand-made entit{'y' if lost_nodes == 1 else 'ies'} (or their "
                  f"type/props) and {lost_edges} hand-added relation(s) cannot be carried across")
            if not args.drop_kg:
                sys.exit("refusing to drop them (--drop-kg-hand-added to override)")

        snap = plaintext_path(out)
        snap.parent.mkdir(parents=True, exist_ok=True)
        write_private(snap, json.dumps(
            [{"id": r["id"], "tier": r["tier"], "timestamp": r["timestamp"],
              "content": d["content"], "metadata": d.get("metadata", {}), "entities": ents}
             for r, d, ents in records], indent=2, ensure_ascii=False))
        print(f"   plaintext snapshot (0600): {snap}")

        print(f"== re-encrypting {len(records)} records under a fresh key in {out}")
        out.mkdir(parents=True, exist_ok=True)

        from memcore_memory.config import Settings
        from memcore_memory.core.ebbinghaus import ForgettingCurve
        from memcore_memory.core.tiers import MemoryItem, Tier
        from memcore_memory.crypto.aes_gcm import AES256GCM
        from memcore_memory.crypto.key_manager import KeyManager
        from memcore_memory.graph.kg import KnowledgeGraph
        from memcore_memory.storage.encrypted_sqlite import EncryptedStore
        from memcore_memory.storage.vector_store import VectorStore

        # Every path spelt out: keyword arguments beat the environment, so nothing
        # inherited can point the key or the database anywhere but OUTPUT_DIR.
        settings = Settings(data_dir=out, db_path=out / "memory.db", key_path=out / "master.key",
                            vector_path=out / "vectors.hnsw", audit_log_path=out / "audit.log",
                            working_buffer_path=out / "working_buffer.jsonl")
        for p in (settings.db_path, settings.key_path, settings.vector_path):
            check(p.parent == out, f"{p} is outside {out}")
        check(not settings.key_path.exists(), f"{settings.key_path} exists - expected a fresh key")
        key = KeyManager(settings.key_path).load_or_create(db_path=settings.db_path)
        cipher = AES256GCM(key)
        store = EncryptedStore(settings.db_path, cipher, blind_index=settings.blind_index_enabled)
        await store.init()
        kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
        await kg.init()

        items = []
        for row, d, ents in records:
            item = MemoryItem(
                id=row["id"],
                content=d["content"],
                tier=Tier(row["tier"]),
                timestamp=row["timestamp"],
                embedding=json.loads(row["embedding"].decode()) if row["embedding"] else None,
                metadata=d.get("metadata", {}),
                forgetting=ForgettingCurve.from_dict(json.loads(row["forgetting_json"]))
                           if row["forgetting_json"] else None,
                entities=ents,
            )
            await store.put(item)
            items.append(item)

        # Links and co-occurrence edges from the memories first, then the hand-made
        # nodes over them (their label, type and props win), then the relations.
        linked = await kg.rebuild_links(items)
        for label, type_, props in carry_nodes:
            await kg.add_entity(label, type_, props)
        for src, dst, relation, weight in carry_edges:
            await kg.add_relation(src, dst, relation, weight)
        print(f"   graph: {linked} memories linked, {len(carry_nodes)} hand-made node(s) and "
              f"{len(carry_edges)} hand-added relation(s) carried")

        # The sidecar, sized from the stored vectors: a new embedder's dim could
        # disagree with them, and VectorStore refuses a mismatch.
        with_vec = [i for i in items if i.embedding]
        if with_vec:
            vs = VectorStore(settings.vector_path, dim=len(with_vec[0].embedding))
            await vs.add_many((i.id, i.embedding, {"tier": i.tier.value}) for i in with_vec)

        print("== verifying through a fresh open of the new store")
        verify = EncryptedStore(settings.db_path, AES256GCM(KeyManager(settings.key_path).load_or_create()))
        got, unreadable = await verify.scan()
        check(not unreadable, f"{len(unreadable)} rows unreadable under the new key")
        check(len(got) == len(records), f"{len(got)} written, {len(records)} expected")
        by_id = {i.id: i for i in got}
        for row, d, ents in records:
            item = by_id.get(row["id"])
            check(item is not None, f"{row['id']}: missing from the new store")
            check(item.content == d["content"], f"{row['id']}: content mismatch")
            check(item.metadata == d.get("metadata", {}), f"{row['id']}: metadata mismatch")
            check(item.tier.value == row["tier"], f"{row['id']}: tier mismatch")
            check(item.timestamp == row["timestamp"], f"{row['id']}: timestamp mismatch")
            want_emb = json.loads(row["embedding"].decode()) if row["embedding"] else None
            check(item.embedding == want_emb, f"{row['id']}: embedding mismatch")
            check(item.entities == ents, f"{row['id']}: entities mismatch")
            if row["forgetting_json"]:
                for field, value in json.loads(row["forgetting_json"]).items():
                    check(getattr(item.forgetting, field) == value, f"{row['id']}: {field}")

        kg_db = settings.db_path.with_suffix(".kg.db")
        con = sqlite3.connect(kg_db)
        nodes = con.execute("SELECT COUNT(*) FROM kg_nodes").fetchone()[0]
        hand_edges = con.execute("SELECT COUNT(*) FROM kg_edges WHERE memory_id IS NULL").fetchone()[0]
        con.close()
        check(not linked or nodes, "memories carry entities but the rebuilt graph is empty")
        check(hand_edges == len(carry_edges), f"{hand_edges} hand-added relations in the new graph, "
                                              f"{len(carry_edges)} carried")
        if carry_nodes:
            have = {(e["label"], e["type"]) for e in await kg.list_entities(limit=nodes)}
            for label, type_, _ in carry_nodes:
                check((label, type_) in have, f"hand-made entity {label!r} ({type_}) missing from the new graph")
        if with_vec:
            vcheck = VectorStore(settings.vector_path, dim=len(with_vec[0].embedding))
            check(len(vcheck.ids) == len(with_vec),
                  f"sidecar holds {len(vcheck.ids)} vectors, {len(with_vec)} expected")
        for name in ("master.key", "memory.db"):
            check((out / name).exists(), f"{name} missing from {out}")
        for name in ("memory.db-wal", "memory.kg.db-wal"):
            # deploy-skynas.sh stages the database files alone.
            check(not (out / name).exists() or not (out / name).stat().st_size,
                  f"{name} still holds data - the store was not closed cleanly")

        write_private(out / FINGERPRINT_FILE, json.dumps(source_fp, sort_keys=True))
        print(f"   {len(got)} records verified byte-for-byte, graph {nodes} nodes, "
              f"{len(with_vec)} vectors")
        print(f"   files: {sorted(p.name for p in out.iterdir())}")

        if args.keep_plaintext:
            print(f"   plaintext snapshot kept: {snap} - delete it once it is no longer needed")
        else:
            snap.unlink()
            print("   plaintext snapshot deleted")
        print(f"\nReady to stage: {out}")


if __name__ == "__main__":
    asyncio.run(main())
