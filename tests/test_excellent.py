
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
    text = "My email is adam@example.com"
    # warn passes the text through untouched but still reports what it saw.
    allowed, out, found = pf.check_and_act(text)
    assert allowed is True
    assert out == text
    assert "email" in found
    # The old assertion ended in `or pf.action == "warn"`, so redaction was never checked.
    allowed, redacted, _ = PIIFilter(enabled=True, action="redact").check_and_act(text)
    assert allowed is True
    assert "adam@example.com" not in redacted
    assert "REDACTED" in redacted
    allowed, _, _ = PIIFilter(enabled=True, action="block").check_and_act("ssn 123-45-6789")
    assert allowed is False

def test_bm25_cache():
    from memcore_memory.retrieval.retrievers.bm25 import BM25Retriever
    from memcore_memory.core.tiers import MemoryItem, Tier
    import asyncio
    # One fixed list: building fresh MemoryItems per call gave them fresh ids, so the
    # cache key changed every time and caching could not be observed at all.
    items = [MemoryItem(content="User likes python", tier=Tier.EPISODIC),
             MemoryItem(content="User prefers dark mode", tier=Tier.SEMANTIC)]

    class MockStore:
        async def list_all(self):
            return list(items)

    retr = BM25Retriever(MockStore())
    results = asyncio.run(retr.retrieve("python", k=2))
    assert results and results[0]["id"] == items[0].id and results[0]["score"] > 0
    assert items[1].id not in [r["id"] for r in results]
    built = retr._bm25
    asyncio.run(retr.retrieve("python", k=2))
    assert retr._bm25 is built, "unchanged corpus must reuse the index"
    items.append(MemoryItem(content="python again", tier=Tier.EPISODIC))
    results = asyncio.run(retr.retrieve("python", k=5))
    assert retr._bm25 is not built, "a changed corpus must rebuild the index"
    assert items[2].id in [r["id"] for r in results]

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
    # A name no hub can serve: the default would pull ~1.3GB of bge-reranker-large
    # on any machine that has sentence-transformers.
    rr = CrossEncoderReranker(model_name="nonexistent/model-for-test")
    if rr._model is not None:
        pytest.skip("a cross-encoder loaded; this pins down the no-model fallback")
    # Given in the wrong order on purpose: returning the input unchanged used to pass.
    docs = [{"id": "2", "content": "dark mode", "score": 0.6}, {"id": "1", "content": "python is great", "score": 0.8}]
    reranked = rr.rerank("python", docs, top_k=2)
    assert [d["id"] for d in reranked] == ["1", "2"]
    assert [d["score"] for d in reranked] == [0.8, 0.6]
    assert [d["id"] for d in rr.rerank("python", docs, top_k=1)] == ["1"]

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

def test_key_manager_enforce_password(tmp_path, monkeypatch):
    # monkeypatch, not os.environ: the old version set MEMCORE_ENV and then popped it
    # in `finally`, which deleted the variable for every test collected after this one.
    from memcore_memory.crypto.key_manager import KeyManager
    monkeypatch.setenv("MEMCORE_ENV", "prod")
    with pytest.raises(ValueError, match="(?i)master password required"):
        KeyManager(tmp_path / "master.key").load_or_create(password=None)
    assert not (tmp_path / "master.key").exists()
