
"""
Append-only audit log with HMAC chaining - excellent version
- Each entry: timestamp, actor, action, memory_id, HMAC(prev_entry + current)
- Tamper-evident: chain verification
- No plaintext content logged
"""
import json, time, hmac, hashlib, os
from pathlib import Path
from typing import Optional

class AuditLog:
    def __init__(self, log_path: Path, hmac_key: bytes):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.key = hmac_key
        self._last_hash = self._load_last_hash()

    def _load_last_hash(self) -> str:
        if not self.log_path.exists():
            return "0"*64
        try:
            with open(self.log_path, 'rb') as f:
                # Read last line efficiently
                f.seek(0, os.SEEK_END)
                pos = f.tell()
                while pos > 0:
                    pos -= 1
                    f.seek(pos)
                    if f.read(1) == b'\n' and pos != f.tell()-1:
                        break
                last_line = f.readline().decode().strip()
                if last_line:
                    entry = json.loads(last_line)
                    return entry.get('chain_hash', "0"*64)
        except Exception:
            pass
        return "0"*64

    def _compute_chain_hash(self, prev_hash: str, entry_data: dict) -> str:
        payload = prev_hash + json.dumps(entry_data, sort_keys=True)
        return hmac.new(self.key, payload.encode(), hashlib.sha256).hexdigest()

    def log(self, action: str, memory_id: str = None, actor: str = "system", metadata: dict = None):
        ts = time.time()
        entry_data = {
            "ts": ts,
            "actor": actor,
            "action": action,  # add, get, recall, delete, promote, forget, re-embed, key-rotate
            "memory_id": memory_id[:16] + "..." if memory_id and len(memory_id) > 16 else memory_id,
            "metadata": metadata or {}
        }
        chain_hash = self._compute_chain_hash(self._last_hash, entry_data)
        entry = {**entry_data, "prev_hash": self._last_hash, "chain_hash": chain_hash}
        
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
        self._last_hash = chain_hash
        return entry

    def verify_chain(self) -> bool:
        if not self.log_path.exists():
            return True
        prev = "0"*64
        with open(self.log_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                    expected = self._compute_chain_hash(prev, {k: v for k, v in entry.items() if k not in ['prev_hash','chain_hash']})
                    if entry['chain_hash'] != expected or entry['prev_hash'] != prev:
                        print(f"Chain broken at line {line_num}")
                        return False
                    prev = entry['chain_hash']
                except Exception as e:
                    print(f"Invalid entry at line {line_num}: {e}")
                    return False
        return True

    def tail(self, n: int = 100):
        if not self.log_path.exists():
            return []
        with open(self.log_path, 'r') as f:
            lines = f.readlines()[-n:]
            return [json.loads(l) for l in lines]
