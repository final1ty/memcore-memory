# Quickstart - runs against a throwaway store in a fresh temp directory.
#
# forget_expired() and consolidate() act on the WHOLE store: pointed at a real one,
# this demo would delete and re-tier memories nobody asked it to touch. config.py
# builds its settings singleton at import time, so the isolation has to happen
# before the package is imported.
import asyncio
import os
import tempfile

_demo_dir = tempfile.mkdtemp(prefix="memcore-quickstart-")
# Overwrite, not setdefault: MNEM_DATA_DIR may already point at a live volume.
os.environ["MNEM_DATA_DIR"] = _demo_dir
# Per-path overrides beat data_dir, and postgres would bypass it entirely.
for _var in ("MNEM_DB_PATH", "MNEM_KEY_PATH", "MNEM_VECTOR_PATH", "MNEM_AUDIT_LOG_PATH",
             "MNEM_WORKING_BUFFER_PATH", "MNEM_BACKEND", "MNEM_DATABASE_URL",
             "MNEM_MASTER_PASSWORD", "MNEM_ENV", "MEMCORE_ENV"):
    os.environ.pop(_var, None)

from memcore_memory import create_memory_system  # noqa: E402  (after the env setup)


async def main():
    mem = await create_memory_system()

    # Add memories across tiers
    await mem.add("User prefers dark mode and concise answers", tier="semantic", importance=0.9, entities=["User"])
    await mem.add("Meeting with Sarah about Q3 roadmap at 10am", tier="episodic", importance=0.7, entities=["Sarah", "Q3 roadmap"])
    await mem.add("Current context: debugging vector search", tier="working", importance=0.6)
    await mem.add("mouse click at (320,240)", tier="sensory", metadata={"sensory": True})

    # Hybrid recall (BM25, graph, metadata, plus vectors when sentence-transformers is installed)
    results = await mem.recall("What does user prefer?", k=5)
    print("Recall results:")
    for r in results:
        print(f"  [{r['tier']}] {r['score']:.3f} R={r['retention']:.2f} | {r['content']}")

    # Ebbinghaus rehearsal: reading a memory strengthens it
    if results:
        await mem.get(results[0]['id'])

    # Consolidation & forgetting
    forgotten = await mem.forget_expired()
    print(f"Forgot {forgotten} expired memories")
    await mem.consolidate()

    print(f"Demo store: {_demo_dir}")


if __name__ == "__main__":
    asyncio.run(main())
