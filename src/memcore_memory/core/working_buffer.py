
"""
Working Buffer Persistence - excellent version
- Was in-memory only, lost on crash. Now persisted to a JSONL file (atomic temp-file replace)
- Miller 7±2 with LRU eviction to Episodic
"""
import json, time
from pathlib import Path
from typing import List
from .tiers import MemoryItem, Tier

class PersistentWorkingBuffer:
    def __init__(self, data_dir: Path, capacity: int = 7):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.capacity = capacity
        self.buffer_file = self.data_dir / "working_buffer.jsonl"
        self.buffer: List[MemoryItem] = []
        self._load()

    def _load(self):
        if not self.buffer_file.exists():
            return
        try:
            with open(self.buffer_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        # Reconstruct MemoryItem minimal
                        item = MemoryItem(
                            id=data['id'],
                            content=data['content'][:500],  # truncated for buffer
                            tier=Tier.WORKING,
                            timestamp=data['timestamp'],
                            metadata=data.get('metadata', {})
                        )
                        self.buffer.append(item)
                    except Exception:
                        continue
            # Keep only last capacity
            self.buffer = self.buffer[-self.capacity:]
        except Exception:
            self.buffer = []

    def _persist(self):
        # Atomic write via temp file
        tmp = self.buffer_file.with_suffix('.tmp')
        with open(tmp, 'w') as f:
            for item in self.buffer:
                f.write(json.dumps({
                    "id": item.id,
                    "content": item.content[:500],
                    "timestamp": item.timestamp,
                    "metadata": item.metadata
                }) + "\n")
        tmp.replace(self.buffer_file)

    def append(self, item: MemoryItem):
        # LRU: if exists, move to end
        self.buffer = [b for b in self.buffer if b.id != item.id]
        self.buffer.append(item)
        if len(self.buffer) > self.capacity:
            evicted = self.buffer.pop(0)
            # Return evicted for promotion to episodic
            self._persist()
            return evicted
        self._persist()
        return None

    def touch(self, memory_id: str):
        # Move to end (most recent)
        for i, item in enumerate(self.buffer):
            if item.id == memory_id:
                moved = self.buffer.pop(i)
                self.buffer.append(moved)
                self._persist()
                break

    def get_all(self) -> List[MemoryItem]:
        return list(self.buffer)

    def clear(self):
        self.buffer = []
        if self.buffer_file.exists():
            self.buffer_file.unlink()
