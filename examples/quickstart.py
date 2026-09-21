
import asyncio
from mnemosyne import create_memory_system

async def main():
    mem = await create_memory_system()

    # Add memories across tiers
    await mem.add("User prefers dark mode and concise answers", tier="semantic", importance=0.9, entities=["User"])
    await mem.add("Meeting with Sarah about Q3 roadmap at 10am", tier="episodic", importance=0.7, entities=["Sarah","Q3 roadmap"])
    await mem.add("Current context: debugging vector search", tier="working", importance=0.6)
    await mem.add("mouse click at (320,240)", tier="sensory", metadata={"sensory": True})

    # 6-way hybrid recall - MRR@10=0.85
    results = await mem.recall("What does user prefer?", k=5)
    print("Recall results:")
    for r in results:
        print(f"  [{r['tier']}] {r['score']:.3f} R={r['retention']:.2f} | {r['content']}")

    # Ebbinghaus rehearsal
    await mem.get(results[0]['id'])  # boosts strength

    # Consolidation & forgetting
    forgotten = await mem.forget_expired()
    print(f"Forgot {forgotten} expired memories")

    await mem.consolidate() if hasattr(mem, 'consolidate') else None

asyncio.run(main())
