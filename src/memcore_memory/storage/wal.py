
import json, time
from pathlib import Path

class WAL:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, op: str, data: dict):
        entry = {'ts': time.time(), 'op': op, 'data': data}
        with open(self.path, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def replay(self):
        if not self.path.exists():
            return []
        entries = []
        with open(self.path, 'r') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except:
                    continue
        return entries
