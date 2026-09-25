
import pytest, asyncio
from mnemosyne import create_memory_system
import tempfile, pathlib

@pytest.mark.asyncio
async def test_add_recall():
    mem = await create_memory_system()
    item = await mem.add("Test memory", tier="semantic", importance=0.9)
    assert item.id
    results = await mem.recall("Test memory", k=1)
    assert len(results) >= 1
    assert results[0]["id"] == item.id

@pytest.mark.asyncio
async def test_ebbinghaus():
    mem = await create_memory_system()
    item = await mem.add("Ebbinghaus test", tier="episodic")
    # Comparing retention just after creation with retention just after touch() was
    # 1.0 >= 1.0 whether or not rehearsal did anything; strength is what it changes.
    s0 = item.forgetting.strength
    item.touch()
    assert item.forgetting.rehearsals == 1
    assert item.forgetting.strength == pytest.approx(s0 * 1.6 + 0.5)
    later = item.forgetting.last_access + 30 * 86400
    fresh = type(item.forgetting)(strength=s0, last_access=item.forgetting.last_access,
                                  importance=item.forgetting.importance)
    assert item.forgetting.retention(later) > fresh.retention(later)
