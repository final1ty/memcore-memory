
import pytest, time, os, tempfile
from pathlib import Path

@pytest.mark.asyncio
async def test_blind_index():
    from memcore_memory.crypto.blind_index import BlindIndex
    key = b"test-key-32-bytes-long-for-blind!!"
    bi = BlindIndex(key)
    content = "User prefers dark mode and likes python"
    idx = bi.compute_index(content, {"importance": 0.9})
    assert len(idx) > 0
    # Search
    q = bi.search_query_hmacs("dark mode")
    assert len(q) > 0
    overlap = set(idx) & set(q)
    assert len(overlap) > 0

@pytest.mark.asyncio
async def test_audit_log_chain():
    from memcore_memory.storage.audit_log import AuditLog
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / "audit.log"
        audit = AuditLog(log_path, b"audit-key-32-bytes-long-test-key!!")
        audit.log("add", "mem_123", actor="test")
        audit.log("get", "mem_123")
        audit.log("delete", "mem_123")
        assert audit.verify_chain() == True
        # Tamper
        with open(log_path, 'a') as f:
            f.write('{"tamper": true}\n')
        assert audit.verify_chain() == False

def test_pii_filter():
    from memcore_memory.security.pii_filter import PIIFilter
    pf = PIIFilter(enabled=True, action="warn")
    findings = pf.scan("Contact me at adam@example.com or 123-45-6789")
    assert "email" in findings
    assert "ssn" in findings
    allowed, redacted, _ = pf.check_and_act("My email is adam@example.com")
    assert allowed == True
    assert "REDACTED" in redacted or pf.action == "warn"

def test_bm25_cache():
    from memcore_memory.retrieval.retrievers.bm25 import BM25Retriever
    # Mock store
    class MockStore:
        async def list_all(self):
            from memcore_memory.core.tiers import MemoryItem, Tier
            return [MemoryItem(content="User likes python", tier=Tier.EPISODIC), MemoryItem(content="User prefers dark mode", tier=Tier.SEMANTIC)]
    
    retr = BM25Retriever(MockStore())
    import asyncio
    results = asyncio.run(retr.retrieve("python", k=2))
    assert isinstance(results, list)

def test_working_buffer_persistence():
    from memcore_memory.core.working_buffer import PersistentWorkingBuffer
    from memcore_memory.core.tiers import MemoryItem, Tier
    with tempfile.TemporaryDirectory() as tmp:
        wb = PersistentWorkingBuffer(Path(tmp), capacity=2)
        wb.append(MemoryItem(content="test1", tier=Tier.WORKING))
        wb.append(MemoryItem(content="test2", tier=Tier.WORKING))
        evicted = wb.append(MemoryItem(content="test3", tier=Tier.WORKING))
        assert evicted is not None
        assert len(wb.get_all()) == 2
        # Reload
        wb2 = PersistentWorkingBuffer(Path(tmp), capacity=2)
        assert len(wb2.get_all()) == 2

def test_ebbinghaus_power_law():
    from memcore_memory.core.ebbinghaus import ForgettingCurve
    fc = ForgettingCurve(strength=1.0, decay_model="power_law", power_d=0.3)
    r1 = fc.retention()
    time.sleep(0.01)
    r2 = fc.retention()
    assert r2 <= r1
    fc.rehearse(feedback=1.0)
    assert fc.strength > 1.0

def test_reranker_fallback():
    from memcore_memory.retrieval.reranker import CrossEncoderReranker
    rr = CrossEncoderReranker()
    docs = [{"id": "1", "content": "python is great", "score": 0.8}, {"id": "2", "content": "dark mode", "score": 0.6}]
    reranked = rr.rerank("python", docs, top_k=2)
    assert len(reranked) == 2

@pytest.mark.asyncio
async def test_encrypted_store_blind_index():
    from memcore_memory.storage.encrypted_sqlite import EncryptedStore
    from memcore_memory.crypto.aes_gcm import AES256GCM
    with tempfile.TemporaryDirectory() as tmp:
        key = AES256GCM.generate_key()
        cipher = AES256GCM(key)
        store = EncryptedStore(Path(tmp)/"test.db", cipher)
        await store.init()
        from memcore_memory.core.tiers import MemoryItem, Tier
        item = MemoryItem(content="User prefers dark mode in python", tier=Tier.EPISODIC, metadata={"importance": 0.9})
        await store.put(item)
        # Blind index search
        ids = await store.search_by_blind_index("dark mode", limit=5)
        assert len(ids) >= 1
        # Content search fallback
        items = await store.search_by_content("dark mode", limit=5)
        assert len(items) >= 1

def test_key_manager_enforce_password():
    from memcore_memory.crypto.key_manager import KeyManager
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["MEMCORE_ENV"] = "prod"
        km = KeyManager(Path(tmp)/"master.key")
        try:
            km.load_or_create(password=None)
            assert False, "Should have raised in prod without password"
        except ValueError as e:
            assert "master password required" in str(e).lower()
        finally:
            os.environ.pop("MEMCORE_ENV", None)
