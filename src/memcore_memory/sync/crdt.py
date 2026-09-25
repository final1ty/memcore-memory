
import time, json
from typing import Dict, Any
from dataclasses import dataclass, field

@dataclass
class LWWRegister:
    value: Any = None
    timestamp: float = 0.0
    node_id: str = ""

    def merge(self, other: 'LWWRegister') -> 'LWWRegister':
        if other.timestamp > self.timestamp or (other.timestamp == self.timestamp and other.node_id > self.node_id):
            return other
        return self

@dataclass
class LWWElementSet:
    # Last-writer-wins element set: one wall-clock timestamp per add and per
    # remove, remove wins ties. This is NOT an Observed-Remove (add-wins) set,
    # which it used to be called: a concurrent add and remove resolve by clock,
    # so peer clock skew decides the outcome.
    adds: Dict[str, float] = field(default_factory=dict)
    removes: Dict[str, float] = field(default_factory=dict)

    def add(self, elem: str):
        self.adds[elem] = time.time()

    def remove(self, elem: str):
        if elem in self.adds:
            self.removes[elem] = time.time()

    def contains(self, elem: str) -> bool:
        return elem in self.adds and self.adds[elem] > self.removes.get(elem, 0)

    def merge(self, other: 'LWWElementSet') -> 'LWWElementSet':
        merged = LWWElementSet()
        all_keys = set(self.adds) | set(other.adds) | set(self.removes) | set(other.removes)
        for k in all_keys:
            merged.adds[k] = max(self.adds.get(k, 0), other.adds.get(k, 0))
            merged.removes[k] = max(self.removes.get(k, 0), other.removes.get(k, 0))
        return merged

    def elements(self):
        return [e for e in self.adds if self.contains(e)]


# The old, misleading name, kept for anything importing it. The wire format
# (to_dict) never carried the class name, so nothing else changes.
ORSet = LWWElementSet

class MemoryCRDT:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.registers: Dict[str, LWWRegister] = {}
        self.tombstones = LWWElementSet()

    def update(self, mem_id: str, data: dict):
        self.registers[mem_id] = LWWRegister(value=data, timestamp=time.time(), node_id=self.node_id)

    def delete(self, mem_id: str):
        self.tombstones.add(mem_id)

    def merge(self, other: 'MemoryCRDT') -> 'MemoryCRDT':
        merged = MemoryCRDT(self.node_id)
        merged.tombstones = self.tombstones.merge(other.tombstones)
        for mid in set(self.registers) | set(other.registers):
            r1 = self.registers.get(mid)
            r2 = other.registers.get(mid)
            winner = r1.merge(r2) if (r1 and r2) else (r1 or r2)
            # A delete observed after the last write wins. Without this, merge
            # ignored tombstones entirely and every deleted memory came straight
            # back on the next sync.
            deleted_at = merged.tombstones.adds.get(mid, 0.0)
            if merged.tombstones.contains(mid) and deleted_at > winner.timestamp:
                continue
            merged.registers[mid] = winner
        return merged

    def merge_from(self, other: 'MemoryCRDT') -> 'MemoryCRDT':
        """Merge `other` into this object in place and return it.

        merge() returns a new object, but P2PNode and GossipProtocol share one
        MemoryCRDT by reference: rebinding either one's attribute to a merge
        result would silently split them.
        """
        merged = self.merge(other)
        self.registers = merged.registers
        self.tombstones = merged.tombstones
        return self

    def live_ids(self):
        """Ids that survive their tombstones - what a peer should actually hold."""
        return [mid for mid, reg in self.registers.items()
                if not (self.tombstones.contains(mid)
                        and self.tombstones.adds.get(mid, 0.0) > reg.timestamp)]

    def to_dict(self):
        return {
            'node': self.node_id,
            'registers': {k: {'value': v.value, 'ts': v.timestamp, 'node': v.node_id} for k, v in self.registers.items()},
            'tombstones': {'adds': self.tombstones.adds, 'removes': self.tombstones.removes}
        }

    @classmethod
    def from_dict(cls, data: dict, node_id: str = None) -> 'MemoryCRDT':
        """Inverse of to_dict. Without it a received payload could not be merged at
        all, which is why the sync endpoints only ever pretended to."""
        crdt = cls(node_id or data.get('node') or 'unknown')
        for mid, reg in (data.get('registers') or {}).items():
            crdt.registers[mid] = LWWRegister(
                value=reg.get('value'),
                timestamp=float(reg.get('ts', 0.0)),
                node_id=reg.get('node', ''),
            )
        tomb = data.get('tombstones') or {}
        crdt.tombstones = LWWElementSet(
            adds={k: float(v) for k, v in (tomb.get('adds') or {}).items()},
            removes={k: float(v) for k, v in (tomb.get('removes') or {}).items()},
        )
        return crdt
