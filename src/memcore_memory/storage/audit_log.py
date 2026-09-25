"""
Append-only audit log with HMAC chaining - excellent version
- Each entry: timestamp, actor, action, memory_id, HMAC(prev_entry + current)
- Tamper-evident: chain verification, anchored by a MACed head record
- No plaintext content logged
"""
import sys
import json, time, hmac, hashlib, os, tempfile
from pathlib import Path
from typing import Optional

try:
    import fcntl
except ImportError:  # pragma: no cover - not on Windows
    fcntl = None

GENESIS = "0" * 64

# Carried (under the chain MAC) by every entry written by code that keeps a head
# file. A log holding such entries must have a head, so deleting the head along
# with the last entries is caught; only a log with no marked entry at all - one
# written before heads existed - may verify without one.
HEAD_AWARE = "hv"

TAMPER_ACTION = "audit-tamper-detected"


class AuditLogCorrupt(ValueError):
    """The log's last line cannot be parsed, so no entry can be chained onto it.

    A ValueError, which is what a corrupt tail raised before it had a type.
    """

    def __init__(self, path: Path, offset: int, detail: str, torn: bool):
        self.path = path
        self.offset = offset
        self.torn = torn
        hint = (" It has no trailing newline, so it looks like a write cut short by a "
                "crash; AuditLog.repair_torn_tail() removes it." if torn else "")
        super().__init__(f"audit log {path}: unparseable entry at byte {offset} ({detail}).{hint}")


def _last_line(f) -> tuple:
    """(offset, bytes) of the last non-blank line of a binary file, read from the end."""
    f.seek(0, os.SEEK_END)
    pos = f.tell()
    buf = b""
    while pos > 0:
        step = min(4096, pos)
        pos -= step
        f.seek(pos)
        buf = f.read(step) + buf
        body = buf.rstrip()
        if b"\n" in body:
            head, line = body.rsplit(b"\n", 1)
            return pos + len(head) + 1, line.strip()
    return 0, buf.strip()


def _ends_with_newline(f) -> bool:
    f.seek(0, os.SEEK_END)
    if f.tell() == 0:
        return True
    f.seek(-1, os.SEEK_END)
    return f.read(1) == b"\n"


class _Walk:
    """What one pass over the whole log found."""

    def __init__(self):
        self.count = 0
        self.tail = GENESIS
        self.error = None           # first broken or unparseable entry, as a message
        self.head_aware = False     # any entry carries HEAD_AWARE
        self.tamper = []            # (line number, reason) of recorded tamper entries
        self.hash_at = {0: GENESIS}  # chain hash after entry n, for the indexes asked for


class AuditLog:
    def __init__(self, log_path: Path, hmac_key: bytes):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        # The head lives beside the log: it records how long the chain is and where
        # it ends, which the chain alone cannot say - dropping its last entries, or
        # the whole file, leaves a chain that still verifies.
        self.head_path = self.log_path.with_name(self.log_path.name + ".head")
        self.key = hmac_key
        self._last_hash = self._load_last_hash()

    def _tail_hash(self, f) -> str:
        offset, last = _last_line(f)
        if not last:
            return GENESIS
        # A corrupt tail must raise. Swallowing it (as this once did, on every
        # reopen, through an off-by-one in a backwards scan) silently restarts the
        # chain from genesis and makes the next entry look like tampering.
        try:
            return json.loads(last)['chain_hash']
        except (ValueError, KeyError, TypeError) as e:
            raise AuditLogCorrupt(self.log_path, offset, f"{type(e).__name__}: {e}",
                                  torn=not _ends_with_newline(f)) from None

    def _load_last_hash(self) -> str:
        if not self.log_path.exists():
            return GENESIS
        with open(self.log_path, 'rb') as f:
            return self._tail_hash(f)

    def _compute_chain_hash(self, prev_hash: str, entry_data: dict) -> str:
        payload = prev_hash + json.dumps(entry_data, sort_keys=True)
        return hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()

    def _head_mac(self, count: int, chain_hash: str) -> str:
        return hmac.new(self.key, f"audit-head-v1:{count}:{chain_hash}".encode(),
                        hashlib.sha256).hexdigest()

    def _read_head(self) -> Optional[dict]:
        if not self.head_path.exists():
            return None
        return json.loads(self.head_path.read_text())

    def _head_problem(self) -> tuple:
        """(head or None, reason it can't be trusted or None)."""
        try:
            head = self._read_head()
        except (OSError, ValueError) as e:
            return None, f"unreadable head {self.head_path}: {e}"
        if head is None:
            return None, None
        try:
            ok = hmac.compare_digest(str(head["mac"]),
                                     self._head_mac(int(head["count"]), str(head["chain_hash"])))
        except (KeyError, TypeError, ValueError):
            ok = False
        if not ok:
            return head, f"head {self.head_path} failed its MAC"
        return head, None

    def _write_head(self, count: int, chain_hash: str):
        data = json.dumps({"count": count, "chain_hash": chain_hash,
                           "mac": self._head_mac(count, chain_hash)})
        fd, tmp = tempfile.mkstemp(dir=self.head_path.parent, prefix=f".{self.head_path.name}.")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.head_path)
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    def _walk(self, want: tuple = ()) -> _Walk:
        """Verify the chain from genesis, recording the chain hash at each index in `want`."""
        w = _Walk()
        if not self.log_path.exists():
            return w
        prev = GENESIS
        with open(self.log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    data = {k: v for k, v in entry.items() if k not in ('prev_hash', 'chain_hash')}
                    expected = self._compute_chain_hash(prev, data)
                    if not (hmac.compare_digest(str(entry['chain_hash']), expected)
                            and hmac.compare_digest(str(entry['prev_hash']), prev)):
                        w.error = f"chain broken at line {line_num}"
                        return w
                except Exception as e:
                    w.error = f"invalid entry at line {line_num}: {e}"
                    return w
                prev = entry['chain_hash']
                w.count += 1
                if data.get(HEAD_AWARE):
                    w.head_aware = True
                if data.get('action') == TAMPER_ACTION:
                    w.tamper.append((line_num, (data.get('metadata') or {}).get('reason')))
                if w.count in want:
                    w.hash_at[w.count] = prev
        w.tail = prev
        return w

    def _head_mismatch(self, head: Optional[dict], head_problem: Optional[str],
                       prev: str) -> tuple:
        """(entry count, tamper reason or None) for appending after `prev`.

        The fast path trusts a valid head that ends where the log ends. Anything
        else walks the whole chain. The head is never quietly recomputed: that is
        what let a truncation verify again after the next legitimate append.
        """
        if head is not None and head_problem is None and head.get("chain_hash") == prev:
            return int(head["count"]), None
        head_count = None
        if head is not None and head_problem is None:
            head_count = int(head["count"])
        w = self._walk(want=(head_count,) if head_count is not None else ())
        if head_problem:
            return w.count, head_problem
        if head is None:
            if w.head_aware:
                return w.count, "head file is missing, but entries were written with one"
            return w.count, None  # a fresh log, or one from before heads existed
        if w.error is None and w.hash_at.get(head_count) == head["chain_hash"]:
            # The head lags the log: a crash between appending and updating the head.
            # Entries past it verify under the key, which no tamperer holds.
            return w.count, None
        return w.count, (f"log ends at entry {w.count} but its head records "
                         f"{head_count} entries ending at {str(head.get('chain_hash'))[:16]}")

    def _append(self, f, prev: str, entry_data: dict) -> str:
        chain_hash = self._compute_chain_hash(prev, entry_data)
        entry = {**entry_data, "prev_hash": prev, "chain_hash": chain_hash}
        f.seek(0, os.SEEK_END)
        f.write((json.dumps(entry) + "\n").encode())
        f.flush()
        os.fsync(f.fileno())
        return chain_hash

    def log(self, action: str, memory_id: str = None, actor: str = "system", metadata: dict = None):
        ts = time.time()
        entry_data = {
            "ts": ts,
            "actor": actor,
            "action": action,  # add, get, recall, delete, promote, forget, re-embed, key-rotate
            "memory_id": memory_id[:16] + "..." if memory_id and len(memory_id) > 16 else memory_id,
            "metadata": metadata or {},
            HEAD_AWARE: 1,
        }
        with open(self.log_path, 'a+b') as f:
            # Held across "read the tail, append": two processes (REST and CLI)
            # appending off the same cached tail would fork the chain.
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                prev = self._tail_hash(f)
                head, head_problem = self._head_problem()
                count, tamper = self._head_mismatch(head, head_problem, prev)
                if tamper:
                    # Logging carries on, but the evidence goes into the chain itself,
                    # where verify_chain() keeps reporting it. Removing this entry just
                    # makes the log disagree with the new head, and is caught again.
                    print(f"[memcore] audit log {self.log_path}: tampering detected ({tamper})",
                          file=sys.stderr)
                    prev = self._append(f, prev, {
                        "ts": ts, "actor": "audit", "action": TAMPER_ACTION, "memory_id": None,
                        "metadata": {"reason": tamper, "found_entries": count,
                                     "head_as_found": {k: head.get(k) for k in ("count", "chain_hash")}
                                     if isinstance(head, dict) else None},
                        HEAD_AWARE: 1,
                    })
                    count += 1
                chain_hash = self._append(f, prev, entry_data)
                self._write_head(count + 1, chain_hash)
            finally:
                if fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._last_hash = chain_hash
        return {**entry_data, "prev_hash": prev, "chain_hash": chain_hash}

    def verify_chain(self) -> bool:
        head, problem = self._head_problem()
        if problem:
            print(f"Audit {problem}", file=sys.stderr)
            return False
        if not self.log_path.exists():
            # Fine for a fresh store; not for one whose head says entries were written.
            if head is not None and int(head["count"]) > 0:
                print(f"Audit log {self.log_path} is missing but its head records "
                      f"{head['count']} entries", file=sys.stderr)
                return False
            return True
        head_count = int(head["count"]) if head is not None else None
        w = self._walk(want=(head_count,) if head_count is not None else ())
        if w.error:
            print(f"Audit log {self.log_path}: {w.error}", file=sys.stderr)
            return False
        if w.tamper:
            # Recorded by log() when it found the log and its head disagreeing. The
            # entries that were removed are gone; this is the proof that they existed.
            for line_num, reason in w.tamper:
                print(f"Audit log {self.log_path}: tampering recorded at line {line_num}: {reason}",
                      file=sys.stderr)
            return False
        if head is None:
            if w.head_aware:
                print(f"Audit head {self.head_path} is missing, but the log was written with one",
                      file=sys.stderr)
                return False
            return True
        if head_count > w.count or w.hash_at.get(head_count) != head["chain_hash"]:
            print(f"Audit log ends at entry {w.count} but its head records {head['count']}: "
                  f"entries were removed from the end", file=sys.stderr)
            return False
        return True

    def repair_torn_tail(self) -> int:
        """Cut off a final entry that a crash left half-written; returns bytes removed.

        Explicit, never automatic: only a last line that has no trailing newline and
        does not parse is removed. A corrupt line that was written in full is not a
        torn write, so it raises and is left for someone to look at.
        """
        if not self.log_path.exists():
            return 0
        with open(self.log_path, 'r+b') as f:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                try:
                    self._tail_hash(f)
                    return 0
                except AuditLogCorrupt as e:
                    if not e.torn:
                        raise
                    offset = e.offset
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.truncate(offset)
                f.flush()
                os.fsync(f.fileno())
            finally:
                if fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        self._last_hash = self._load_last_hash()
        return size - offset

    def tail(self, n: int = 100):
        if not self.log_path.exists():
            return []
        with open(self.log_path, 'r') as f:
            lines = [l for l in f.readlines() if l.strip()][-n:]
            return [json.loads(l) for l in lines]
