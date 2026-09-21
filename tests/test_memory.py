
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
    r1 = item.forgetting.retention()
    item.touch()
    r2 = item.forgetting.retention()
    assert r2 >= r1
    assert item.forgetting.rehearsals == 1
