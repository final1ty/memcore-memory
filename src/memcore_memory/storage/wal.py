"""An application-level append log. Not used by the store.

Durability comes from SQLite itself: EncryptedStore runs memory.db in WAL journal
mode (see EncryptedStore.init). Nothing in this package imports this class; it
is kept only for outside callers and is a candidate for deletion.
"""
import json, os, sys, time
from pathlib import Path

class WAL:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, op: str, data: dict):
        entry = {'ts': time.time(), 'op': op, 'data': data}
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')
            # A log that isn't flushed to disk before returning protects nothing.
            f.flush()
            os.fsync(f.fileno())

    def replay(self):
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    # A torn final line is expected after a crash - that is what a
                    # write-ahead log is for. Skip it, but say so: a bare `except`
                    # here also swallowed KeyboardInterrupt and hid real corruption.
                    print(f"[wal] skipping unparseable entry in {self.path}", file=sys.stderr)
                    continue
        return entries
