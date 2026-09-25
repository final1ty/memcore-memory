# P2P sync is NOT implemented: there is no transport, and P2PNode.start() raises
# NotImplementedError. What does exist are the CRDT primitives a transport would
# carry, so this demo merges two in-memory replicas directly. It opens no socket
# and touches no store.
from memcore_memory.sync.crdt import MemoryCRDT


def main():
    a = MemoryCRDT("node-a")
    b = MemoryCRDT("node-b")

    a.update("mem-1", {"content": "written on A"})
    b.update("mem-2", {"content": "written on B"})
    b.update("mem-1", {"content": "written on A"})
    b.delete("mem-1")

    # What a receiving peer would do with a payload: rebuild it, then merge.
    received = MemoryCRDT.from_dict(b.to_dict())
    merged = a.merge(received)
    print("live after merge:", sorted(merged.live_ids()))  # mem-1's tombstone wins


if __name__ == "__main__":
    main()
