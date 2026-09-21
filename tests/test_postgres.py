
import pytest, os

pytestmark = pytest.mark.skipif(
    os.getenv("MNEM_BACKEND") != "postgres",
    reason="Postgres backend not enabled"
)

@pytest.mark.asyncio
async def test_postgres_backend():
    from mnemosyne import create_memory_system
    mem = await create_memory_system(backend="postgres")
    item = await mem.add("Postgres pgvector test", tier="semantic", importance=0.9)
    results = await mem.recall("pgvector", k=1)
    assert len(results) >= 1
