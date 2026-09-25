#!/usr/bin/env python3
"""Row-level fingerprint of a memcore data dir, to tell whether it changed.

    store_fingerprint.py DATA_DIR                 print the fingerprint as JSON
    store_fingerprint.py --compare OLD.json NEW.json

rescue-container-store.py records one of the snapshot it rescued, and
deploy-skynas.sh takes one of the stopped volume and compares the two. Any
difference is a write that landed between the snapshot and the stop - a new
memory, a delete, a content update, a rehearsal, a tier move, a hand-added
relation - and staging the rescue would silently undo it. Comparing ids alone
caught only the first of those.

Every row of every table is hashed as stored, ciphertext included, so nothing
here needs the key and nothing is decrypted. memories rows are keyed by id, so a
difference can be named; the other tables are one digest each.

Standard library only, and no f-string tricks newer than 3.8: deploy-skynas.sh
runs this with the python of whatever image the old container was built from.
"""

import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tempfile

DATABASES = ("memory.db", "memory.kg.db")


def _row_digest(row) -> str:
    h = hashlib.sha256()
    for value in row:
        # Type-tagged, so 1, 1.0, "1" and b"1" never collide.
        if isinstance(value, bytes):
            part = "b:" + value.hex()
        else:
            part = type(value).__name__ + ":" + repr(value)
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _database(path: str) -> dict:
    # Read from a private copy, the -wal with it: the source may be a read-only
    # mount, and a leftover -wal holds commits the main file does not have yet.
    tmp = tempfile.mkdtemp()
    try:
        copy = os.path.join(tmp, "db")
        for suffix in ("", "-wal"):
            if os.path.exists(path + suffix):
                shutil.copyfile(path + suffix, copy + suffix)
        con = sqlite3.connect(copy)
        try:
            out = {"rows": {}, "tables": {}}
            tables = [r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
            for table in tables:
                cols = [r[1] for r in con.execute('PRAGMA table_info("%s")' % table)]
                rows = con.execute('SELECT * FROM "%s"' % table).fetchall()
                if table == "memories" and "id" in cols:
                    at = cols.index("id")
                    out["rows"] = {r[at]: _row_digest(r) for r in rows}
                else:
                    digests = sorted(_row_digest(r) for r in rows)
                    out["tables"][table] = hashlib.sha256("".join(digests).encode()).hexdigest()
            return out
        finally:
            con.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def fingerprint(data_dir: str) -> dict:
    return {name: _database(os.path.join(data_dir, name))
            for name in DATABASES if os.path.exists(os.path.join(data_dir, name))}


def ids(fp: dict) -> list:
    return sorted(fp.get("memory.db", {}).get("rows", {}))


def differences(old: dict, new: dict) -> list:
    """Human-readable differences between two fingerprints; empty when equal."""
    out = []
    for name in sorted(set(old) | set(new)):
        if name not in new:
            out.append("%s is gone" % name)
            continue
        if name not in old:
            out.append("%s appeared" % name)
            continue
        a, b = old[name], new[name]
        added = sorted(set(b["rows"]) - set(a["rows"]))
        removed = sorted(set(a["rows"]) - set(b["rows"]))
        changed = sorted(i for i in set(a["rows"]) & set(b["rows"]) if a["rows"][i] != b["rows"][i])
        for label, found in (("added", added), ("deleted", removed), ("changed", changed)):
            if found:
                out.append("%s: %d memor%s %s, e.g. %s" % (
                    name, len(found), "y" if len(found) == 1 else "ies", label, ", ".join(found[:3])))
        for table in sorted(set(a["tables"]) | set(b["tables"])):
            if a["tables"].get(table) != b["tables"].get(table):
                out.append("%s: table %s changed" % (name, table))
    return out


def main(argv):
    if len(argv) == 3 and argv[0] == "--compare":
        with open(argv[1]) as f:
            old = json.load(f)
        with open(argv[2]) as f:
            new = json.load(f)
        diff = differences(old, new)
        for line in diff:
            print(line)
        return 1 if diff else 0
    if len(argv) == 1 and not argv[0].startswith("-"):
        print(json.dumps(fingerprint(argv[0]), sort_keys=True))
        return 0
    sys.stderr.write(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
