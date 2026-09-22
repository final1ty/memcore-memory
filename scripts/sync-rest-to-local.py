#!/usr/bin/env python3
#
# Copy every record from the containerised REST store into the local encrypted
# store, so the store the MCP server opens holds everything.
#
# The two stores are separate databases and nothing syncs them. Since .mcp.json
# stopped passing --remote, `server mcp` opens ~/.memcore, while the mnemosyne
# container keeps its own volume. Anything written through the container is
# invisible to the MCP client until this script brings it across.
#
# Records are matched on exact full content, so re-running only adds what is new.
# Content is always read through /memory/{id}: the /memories listing truncates the
# content field at ~200 characters, and importing from it writes truncated records.
#
# Rollback: this script only adds. Every record it writes carries
# metadata.imported_from == "docker-rest-store"; delete those to undo an import.
#
# The container store is shared by every project, so --prefix restricts a run to one
# subject's records (matched on the start of the content) when a full sync is too much.
#
# Usage:
#   MNEM_MASTER_PASSWORD=... scripts/sync-rest-to-local.py [--dry-run] [--prefix TEXT]
#
# Environment:
#   MNEM_MASTER_PASSWORD  required, unlocks the local store
#   MEMCORE_ENDPOINT      REST store base URL (default http://192.168.1.183:8000)
#   MEMCORE_PYTHON        interpreter that has memcore installed (default <repo>/.venv/bin/python)

import collections
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENDPOINT = os.environ.get("MEMCORE_ENDPOINT", "http://192.168.1.183:8000").rstrip("/")
PYTHON = os.environ.get("MEMCORE_PYTHON", os.path.join(REPO, ".venv", "bin", "python"))
ARGS = sys.argv[1:]
DRY_RUN = "--dry-run" in ARGS
PREFIX = ARGS[ARGS.index("--prefix") + 1] if "--prefix" in ARGS else None
MARKER = "docker-rest-store"


def fail(msg):
    sys.exit(f"sync-rest-to-local: {msg}")


def preflight():
    if not os.environ.get("MNEM_MASTER_PASSWORD"):
        fail("MNEM_MASTER_PASSWORD is not set; the local store cannot be opened.")
    if not os.path.isfile(PYTHON):
        fail(f"interpreter not found: {PYTHON} (set MEMCORE_PYTHON)")
    try:
        with urllib.request.urlopen(f"{ENDPOINT}/health", timeout=15) as r:
            health = json.load(r)
    except (urllib.error.URLError, OSError) as exc:
        fail(f"REST store unreachable at {ENDPOINT}: {exc}")
    if health.get("status") != "ok":
        fail(f"REST store reports status={health.get('status')!r}")
    print(f"pre-flight ok: {ENDPOINT} healthy, interpreter {PYTHON}")


def rest(path):
    with urllib.request.urlopen(f"{ENDPOINT}{path}", timeout=60) as r:
        return json.load(r)


def read_source():
    listing = rest("/memories?limit=1000")
    listing = listing if isinstance(listing, list) else listing.get("memories", [])
    # full content only: the listing truncates it
    return [rest(f"/memory/{m['id']}") for m in listing]


class LocalStore:
    """The local encrypted store, driven over the MCP server's stdio interface."""

    def __init__(self):
        self.proc = subprocess.Popen(
            [PYTHON, "-u", "-m", "memcore_memory.cli.main", "server", "mcp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1, cwd=REPO, env=os.environ.copy())
        self._next = 1
        reply = self._rpc("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "sync-rest-to-local", "version": "1"}})
        if "result" not in reply:
            fail(f"MCP handshake failed: {reply}")
        self._notify("notifications/initialized")

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
                fail(f"MCP server closed the connection during {method}; "
                     f"stderr: {self.proc.stderr.read()[-300:]}")
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

    def contents(self):
        _, text = self.call("memory_list", {"limit": 1000})
        out = {}
        for item in json.loads(text)["items"]:
            err, full = self.call("memory_get", {"id": item["id"]})
            if not err:
                out[item["id"]] = json.loads(full)["content"]
        return out

    def add(self, record):
        return self.call("memory_add", {
            "content": record["content"],
            "tier": record.get("tier", "episodic"),
            "importance": float((record.get("metadata") or {}).get("importance", 0.5)),
            "metadata": {"imported_from": MARKER, "origin_id": record.get("id")}})

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def main():
    preflight()
    source = read_source()
    total = len(source)
    if PREFIX:
        source = [r for r in source if r.get("content", "").startswith(PREFIX)]
        print(f"REST store: {total} records, {len(source)} matching prefix {PREFIX!r}")
    else:
        print(f"REST store: {total} records")

    store = LocalStore()
    try:
        local = store.contents()
        have = set(local.values())
        print(f"local store: {len(local)} records")

        missing = [r for r in source if r["content"] not in have]
        if DRY_RUN:
            print(f"dry run: {len(missing)} record(s) would be added, nothing written")
            for r in missing[:10]:
                print(f"  + [{r.get('tier')}] {r['content'][:70]}")
            return 0

        added = failed = 0
        for record in missing:
            err, text = store.add(record)
            if err:
                failed += 1
                print(f"  FAILED: {text[:150]}")
            else:
                added += 1
                have.add(record["content"])

        still_missing = [r for r in source if r["content"] not in have]
        local = store.contents()
        dupes = sum(v - 1 for v in collections.Counter(local.values()).values() if v > 1)
        print(f"added={added} already_present={len(source) - len(missing)} failed={failed}")
        print(f"local store now: {len(local)} records, {dupes} duplicate(s)")
        print(f"coverage: every REST record present locally: {not still_missing}")
        return 1 if (failed or still_missing or dupes) else 0
    finally:
        store.close()


if __name__ == "__main__":
    sys.exit(main())
