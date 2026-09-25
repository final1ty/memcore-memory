#!/usr/bin/env python3
#
# Copy the records of the containerised REST store into a local encrypted store
# (default ~/.memcore, the one .mcp.json's local mode opens).
#
# The two stores are separate databases and nothing syncs them. Anything written
# through the container is invisible to a local-mode MCP client until this script
# brings it across.
#
# What a copy keeps: content, tier, importance, the source's metadata, and its
# entities when the REST API returns them (GET /memory/{id} does since the audit
# round of 2026-09-24; against an older server every copy arrives without them, and
# the run says how many). What it cannot keep: the local copy gets
# a new id, timestamp = now and a fresh forgetting curve, because memory_add takes
# none of those - so rehearsal history is reset and the temporal prior ranks every
# import as brand new. Each copy records the source id as metadata.origin_id.
#
# Matching is on origin_id first, then on exact full content for records imported
# before origin_id was read back. A source record whose content changed after it
# was imported is reported, and only rewritten with --update-changed: the local
# copy may have been edited on purpose.
#
# Content is always read through /memory/{id}: the /memories listing truncates the
# content field at 500 characters (api/rest.py list_memories), and importing from
# it writes truncated records. The local store is read directly and read-only,
# never through memory_get, which counts every read as a rehearsal - a sync used to
# rehearse every local memory twice per run.
#
# Rollback: this script only adds, or with --update-changed updates its own
# imports. Every record it writes carries metadata.imported_from ==
# "docker-rest-store"; delete those to undo an import.
#
# The container store is shared by every project, so --prefix restricts a run to one
# subject's records (matched on the start of the content) when a full sync is too much.
#
# --dry-run opens nothing but the read-only reader: no MCP server, so no schema
# migration, key_id stamp or graph upgrade runs against the local store either.
#
# A --data-dir without a store in it (memory.db and master.key) is refused unless
# --create is given: a typo in the path used to get a brand-new store, and the
# sync reported success into it.
#
# Usage:
#   MNEM_MASTER_PASSWORD=... scripts/sync-rest-to-local.py [--dry-run] [--prefix TEXT]
#                                   [--data-dir DIR] [--create] [--update-changed]
#
# Environment:
#   MNEM_MASTER_PASSWORD  required, unlocks the local store
#   MNEM_API_KEY          sent to the REST store when it requires one
#   MEMCORE_ENDPOINT      REST store base URL (default http://192.168.1.183:8000)
#   MEMCORE_PYTHON        interpreter that has memcore installed (default <repo>/.venv/bin/python)
#
# Every other inherited MNEM_* variable is withheld from the local server: an
# exported MNEM_REMOTE_URL turned it into a bridge to the REST store itself, and the
# sync then compared that store with itself and reported full coverage.

import argparse
import collections
import json
import os
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = os.environ.get("MEMCORE_ENDPOINT", "http://192.168.1.183:8000").rstrip("/")
PYTHON = os.environ.get("MEMCORE_PYTHON", os.path.join(REPO, ".venv", "bin", "python"))
MARKER = "docker-rest-store"
# The listing's own cap (api/rest.py). There is no offset, so a larger store can't
# be enumerated through it.
LIST_CAP = 500


def fail(msg):
    sys.exit(f"sync-rest-to-local: {msg}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Copy REST-store records into a local store.")
    p.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    p.add_argument("--prefix", metavar="TEXT", help="only records whose content starts with TEXT")
    p.add_argument("--data-dir", type=Path, default=Path.home() / ".memcore",
                   help="local store to write into (default ~/.memcore)")
    p.add_argument("--update-changed", action="store_true",
                   help="rewrite imports whose source content has changed since")
    p.add_argument("--create", action="store_true",
                   help="create the local store when --data-dir holds none")
    return p.parse_args(argv)


def child_env(environ, data_dir: Path) -> dict:
    """The environment for the local server and reader: no inherited MNEM_* but the password.

    The store is named explicitly instead, so what gets opened never depends on
    whatever the calling shell happened to export.
    """
    env = {k: v for k, v in environ.items() if not k.upper().startswith("MNEM_")}
    if environ.get("MNEM_MASTER_PASSWORD"):
        env["MNEM_MASTER_PASSWORD"] = environ["MNEM_MASTER_PASSWORD"]
    env["MNEM_DATA_DIR"] = str(data_dir)
    return env


def _request(path, timeout):
    headers = {}
    if os.environ.get("MNEM_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['MNEM_API_KEY']}"
    req = urllib.request.Request(f"{ENDPOINT}{path}", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def rest(path):
    return _request(path, 60)


def preflight():
    if not os.environ.get("MNEM_MASTER_PASSWORD"):
        fail("MNEM_MASTER_PASSWORD is not set; the local store cannot be opened.")
    if not os.path.isfile(PYTHON):
        fail(f"interpreter not found: {PYTHON} (set MEMCORE_PYTHON)")
    try:
        health = _request("/health", 15)
    except (urllib.error.URLError, OSError) as exc:
        fail(f"REST store unreachable at {ENDPOINT}: {exc}")
    if health.get("status") != "ok":
        fail(f"REST store reports status={health.get('status')!r}")
    tier_counts = health.get("tier_counts")
    if not isinstance(tier_counts, dict):
        fail("REST /health has no tier_counts; cannot tell how many records there are")
    print(f"pre-flight ok: {ENDPOINT} healthy, interpreter {PYTHON}")
    return sum(tier_counts.values())


def read_source(expected):
    if expected > LIST_CAP:
        fail(f"the REST store holds {expected} records but /memories lists at most {LIST_CAP}")
    listing = rest(f"/memories?limit={LIST_CAP}")
    listing = listing if isinstance(listing, list) else listing.get("memories", [])
    if len(listing) < expected:
        fail(f"/memories listed {len(listing)} of {expected} records")
    # full content only: the listing truncates it
    return [rest(f"/memory/{m['id']}") for m in listing]


# Runs in the memcore interpreter. Opens the store read-only in effect: no init(),
# so no schema writes, and scan() rather than get(), so nothing is rehearsed.
_READER = r'''
import asyncio, json, sys
from memcore_memory.config import settings
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.storage.encrypted_sqlite import EncryptedStore

async def main():
    if not settings.key_path.exists() or not settings.db_path.exists():
        sys.exit(f"no store at {settings.data_dir}")
    key = KeyManager(settings.key_path).load_or_create(db_path=settings.db_path)
    items, unreadable = await EncryptedStore(settings.db_path, AES256GCM(key), blind_index=False).scan()
    if unreadable:
        sys.exit(f"{len(unreadable)} row(s) in {settings.db_path} cannot be read")
    print(json.dumps({"data_dir": str(settings.data_dir), "items": [
        {"id": i.id, "content": i.content, "origin_id": (i.metadata or {}).get("origin_id")}
        for i in items]}))

asyncio.run(main())
'''


def read_local(env):
    done = subprocess.run([PYTHON, "-c", _READER], capture_output=True, text=True, cwd=REPO, env=env)
    if done.returncode != 0:
        fail(f"could not read the local store: {done.stderr.strip()[-300:]}")
    lines = [ln for ln in done.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def has_store(data_dir: Path) -> bool:
    return (data_dir / "memory.db").is_file() and (data_dir / "master.key").is_file()


def without_entities(source):
    """Records the REST server returned no entities field for (an older server)."""
    return sum(1 for r in source if "entities" not in r)


def duplicates(items):
    return sum(v - 1 for v in collections.Counter(i["content"] for i in items).values() if v > 1)


def plan(source, local_items):
    """Split source records into (present, changed, missing).

    changed pairs a source record with the local id of its import whose content
    no longer matches.
    """
    by_origin = {}
    for item in local_items:
        if item.get("origin_id"):
            by_origin.setdefault(item["origin_id"], item)
    contents = {item["content"] for item in local_items}
    present, changed, missing = [], [], []
    for record in source:
        local = by_origin.get(record.get("id"))
        if local is not None:
            if local["content"] == record["content"]:
                present.append(record)
            else:
                changed.append((record, local["id"]))
        elif record["content"] in contents:
            present.append(record)
        else:
            missing.append(record)
    return present, changed, missing


def import_metadata(record):
    # The markers last, so a source key of the same name cannot hide an import.
    return {**(record.get("metadata") or {}), "imported_from": MARKER, "origin_id": record.get("id")}


class LocalStore:
    """The local encrypted store's MCP server, used for writes only."""

    def __init__(self, env, data_dir: Path):
        # A file, not a pipe: the server logs to stderr, and a pipe nobody drains
        # fills at 64 KB and blocks it mid-call while this script waits on stdout.
        self._stderr = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(
            [PYTHON, "-u", "-m", "memcore_memory.cli.main", "server", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
            text=True, bufsize=1, cwd=REPO, env=env)
        self._next = 1
        reply = self._rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "sync-rest-to-local", "version": "1"}})
        if "result" not in reply:
            fail(f"MCP handshake failed: {reply}")
        self._notify("notifications/initialized")
        # Proof of which store the server opened. In bridge mode health_check answers
        # with the remote's data dir, so this is where a redirect shows.
        err, text = self.call("health_check", {})
        opened = None if err else json.loads(text).get("data_dir")
        if opened is None or Path(opened).resolve() != data_dir.resolve():
            fail(f"the local MCP server opened {opened!r}, not {data_dir}")

    def _stderr_tail(self):
        self._stderr.seek(0)
        return self._stderr.read()[-300:]

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _notify(self, method, params=None):
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def _rpc(self, method, params):
        self._next += 1
        rid = self._next
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        while True:
            line = self.proc.stdout.readline()
            if not line:
                fail(f"MCP server closed the connection during {method}; stderr: {self._stderr_tail()}")
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("id") == rid:
                return msg

    def call(self, tool, args):
        result = self._rpc("tools/call", {"name": tool, "arguments": args}).get("result", {})
        text = "".join(c.get("text", "") for c in result.get("content", []))
        return result.get("isError", False), text

    def add(self, record):
        args = {
            "content": record["content"],
            "tier": record.get("tier", "episodic"),
            "importance": float((record.get("metadata") or {}).get("importance", 0.5)),
            "metadata": import_metadata(record)}
        if record.get("entities"):
            args["entities"] = record["entities"]
        return self.call("memory_add", args)

    def update(self, local_id, record):
        return self.call("memory_update", {"id": local_id, "content": record["content"],
                                           "metadata": import_metadata(record)})

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self._stderr.close()


def main(argv=None):
    args = parse_args(argv)
    if args.prefix is not None and (not args.prefix or args.prefix.startswith("--")):
        fail("--prefix needs a value")
    data_dir = args.data_dir.expanduser().resolve()
    exists = has_store(data_dir)
    if not exists and not args.create:
        fail(f"no store at {data_dir} (memory.db and master.key); --create to make a new one there")
    expected = preflight()
    source = read_source(expected)
    total = len(source)
    if args.prefix:
        source = [r for r in source if r.get("content", "").startswith(args.prefix)]
        print(f"REST store: {total} records, {len(source)} matching prefix {args.prefix!r}")
    else:
        print(f"REST store: {total} records")

    no_entities = without_entities(source)
    if no_entities:
        print(f"WARNING: the REST server returned no entities for {no_entities} record(s) - it "
              f"predates entities in GET /memory/{{id}}; those copies arrive without them")

    env = child_env(os.environ, data_dir)
    if args.dry_run:
        # The reader alone: starting the server would run init() against the store.
        before = read_local(env) if exists else {"data_dir": str(data_dir), "items": []}
        _, changed, missing = plan(source, before["items"])
        print(f"local store: {before['data_dir']}, {len(before['items'])} records"
              f"{'' if exists else ' (does not exist yet)'}")
        print(f"dry run: {len(missing)} record(s) would be added, "
              f"{len(changed)} changed at the source, nothing written")
        for r in missing[:10]:
            print(f"  + [{r.get('tier')}] {r['content'][:70]}")
        for r, _ in changed[:10]:
            print(f"  ~ [{r.get('tier')}] {r['content'][:70]}")
        return 0

    store = LocalStore(env, data_dir)
    try:
        before = read_local(env)
        print(f"local store: {before['data_dir']}, {len(before['items'])} records")
        pre_dupes = duplicates(before["items"])

        present, changed, missing = plan(source, before["items"])
        added = updated = failed = 0
        for record in missing:
            err, text = store.add(record)
            if err:
                failed += 1
                print(f"  FAILED: {text[:150]}")
            else:
                added += 1
        if args.update_changed:
            for record, local_id in changed:
                err, text = store.update(local_id, record)
                if err:
                    failed += 1
                    print(f"  FAILED update of {local_id}: {text[:150]}")
                else:
                    updated += 1
        elif changed:
            print(f"{len(changed)} import(s) differ from their source record; "
                  f"--update-changed rewrites them:")
            for record, local_id in changed[:10]:
                print(f"  ~ {local_id} <- {record.get('id')}: {record['content'][:60]}")

        after = read_local(env)
        _, still_changed, still_missing = plan(source, after["items"])
        dupes = duplicates(after["items"])
        new_dupes = max(0, dupes - pre_dupes)
        print(f"added={added} updated={updated} already_present={len(present)} failed={failed}"
              f"{f' without_entities={no_entities}' if no_entities else ''}")
        print(f"local store now: {len(after['items'])} records, {dupes} duplicate(s) "
              f"({new_dupes} new this run)")
        print(f"coverage: every REST record present locally: {not still_missing}"
              f"{'' if not still_changed else f', {len(still_changed)} still differ'}")
        # Duplicates that predate this run are reported, not failed on: one old
        # duplicate used to make every later run exit 1.
        return 1 if (failed or still_missing or still_changed or new_dupes) else 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
