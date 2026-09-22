"""Regression tests for the bugs found in the 2026-09-22 full-codebase review.

Common theme, again: features that reported success while doing nothing. Vectors
that were never written to disk, weights that could not affect ranking, a rate
limiter that was never attached to an app, a sync endpoint that answered "merged".
"""

import json

import pytest
from fastapi.testclient import TestClient

from memcore_memory.api.rest import create_app
from memcore_memory.config import settings
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.retrieval.hybrid import HybridRetriever
from memcore_memory.storage.encrypted_sqlite import EncryptedStore
from memcore_memory.storage.vector_store import VectorStore
from memcore_memory.sync.crdt import MemoryCRDT


@pytest.fixture
async def store(tmp_path):
    s = EncryptedStore(tmp_path / "review.db", AES256GCM(AES256GCM.generate_key()))
    await s.init()
    return s


# --- vector store ---------------------------------------------------------

async def test_vectors_survive_a_restart_without_hnswlib(tmp_path):
    """The bug: _persist() ran only `if self.index`, so with no hnswlib installed
    - which is the case on every machine here - nothing was ever saved, and the
    vector arm of the retriever started empty after each restart."""
    vs = VectorStore(tmp_path / "v.hnsw", dim=4)
    await vs.add("a", [1.0, 0.0, 0.0, 0.0], {"tier": "episodic"})
    await vs.add("b", [0.0, 1.0, 0.0, 0.0], {"tier": "episodic"})

    reopened = VectorStore(tmp_path / "v.hnsw", dim=4)
    assert reopened.ids == ["a", "b"]
    assert reopened.id_to_meta["a"] == {"tier": "episodic"}
    assert [mid for mid, _ in await reopened.search([1.0, 0.0, 0.0, 0.0], k=1)] == ["a"]


async def test_delete_keeps_ids_and_vectors_aligned(tmp_path):
    """The HNSW labels were list positions, which shift when an earlier id is
    deleted - so search returned the wrong memory for every later vector."""
    vs = VectorStore(tmp_path / "v.hnsw", dim=3)
    await vs.add("first", [1.0, 0.0, 0.0], {})
    await vs.add("second", [0.0, 1.0, 0.0], {})
    await vs.add("third", [0.0, 0.0, 1.0], {})
    await vs.delete("first")

    assert vs.ids == ["second", "third"]
    assert [mid for mid, _ in await vs.search([0.0, 0.0, 1.0], k=1)] == ["third"]
    assert "first" not in vs.id_to_meta


async def test_a_wrong_sized_embedding_is_refused(tmp_path):
    vs = VectorStore(tmp_path / "v.hnsw", dim=4)
    with pytest.raises(ValueError, match="dimensions"):
        await vs.add("bad", [1.0, 2.0], {})


def test_vectors_from_a_different_embedder_are_not_loaded(tmp_path):
    """Mixing 768-dim and 384-dim vectors yields confident nonsense, not an error."""
    path = tmp_path / "v.hnsw"
    VectorStore(path, dim=4)
    path.with_suffix(VectorStore.SIDECAR_SUFFIX).write_text(json.dumps(
        {"dim": 8, "ids": ["x"], "vectors": [[0.0] * 8], "meta": {}}))
    assert VectorStore(path, dim=4).ids == []


def test_a_corrupt_sidecar_does_not_crash_startup(tmp_path):
    path = tmp_path / "v.hnsw"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(VectorStore.SIDECAR_SUFFIX).write_text("{not json")
    assert VectorStore(path, dim=4).ids == []


# --- weighted RRF ---------------------------------------------------------

def _fused(weights, lists):
    r = HybridRetriever.__new__(HybridRetriever)
    r.weights, r.rrf_k = weights, 60
    return r._rrf(lists)


def test_retriever_weights_actually_change_the_ranking():
    """They used to weight only a raw-score term, leaving the documented weights
    (vector 0.35, bm25 0.25, ...) with almost no effect on the fused order."""
    lists = [
        [{"id": "v", "source": "vector", "score": 0.5}],
        [{"id": "b", "source": "bm25", "score": 0.5}],
    ]
    vector_heavy = _fused({"vector": 0.9, "bm25": 0.1}, lists)
    bm25_heavy = _fused({"vector": 0.1, "bm25": 0.9}, lists)
    assert vector_heavy["v"] > vector_heavy["b"]
    assert bm25_heavy["b"] > bm25_heavy["v"]


def test_fusion_ignores_incomparable_raw_scores():
    """BM25 is unbounded while cosine sits in [0,1]; rank is the only comparable
    quantity, which is the entire point of RRF."""
    weights = {"vector": 0.5, "bm25": 0.5}
    modest = _fused(weights, [[{"id": "x", "source": "bm25", "score": 0.01}]])
    huge = _fused(weights, [[{"id": "x", "source": "bm25", "score": 9999.0}]])
    assert modest["x"] == huge["x"]


def test_a_document_found_by_several_retrievers_outranks_one_found_once():
    weights = {"vector": 0.35, "bm25": 0.25, "graph": 0.15}
    fused = _fused(weights, [
        [{"id": "both", "source": "vector", "score": 0.4}],
        [{"id": "both", "source": "bm25", "score": 0.4}],
        [{"id": "one", "source": "vector", "score": 0.9}],
    ])
    assert fused["both"] > fused["one"]


# --- CRDT -----------------------------------------------------------------

def test_a_deleted_memory_does_not_come_back_on_merge():
    """merge() ignored tombstones entirely, so every delete was undone by the
    next sync with a peer that still had the record."""
    a, b = MemoryCRDT("a"), MemoryCRDT("b")
    a.update("m1", {"content": "hello"})
    b.registers["m1"] = a.registers["m1"]
    a.delete("m1")

    merged = a.merge(b)
    assert "m1" not in merged.live_ids()
    assert "m1" not in merged.registers


def test_a_write_after_a_delete_wins():
    a, b = MemoryCRDT("a"), MemoryCRDT("b")
    a.update("m1", {"content": "old"})
    a.delete("m1")
    b.update("m1", {"content": "rewritten"})
    b.registers["m1"].timestamp = a.tombstones.adds["m1"] + 1

    merged = a.merge(b)
    assert merged.registers["m1"].value == {"content": "rewritten"}
    assert "m1" in merged.live_ids()


def test_crdt_round_trips_through_its_wire_format():
    """Without from_dict a received payload could not be merged at all."""
    a = MemoryCRDT("a")
    a.update("m1", {"content": "keep"})
    a.update("m2", {"content": "drop"})
    a.delete("m2")

    restored = MemoryCRDT.from_dict(json.loads(json.dumps(a.to_dict())))
    assert restored.registers["m1"].value == {"content": "keep"}
    assert restored.tombstones.contains("m2")
    assert restored.merge(a).live_ids() == ["m1"]


# --- REST middleware and honest endpoints ---------------------------------

@pytest.fixture
async def client(isolate_data_dir):
    key = KeyManager(settings.key_path).load_or_create()
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    kg = KnowledgeGraph(settings.db_path.with_suffix(".kg.db"), cipher)
    await kg.init()
    mem = MnemosyneMemory(store, VectorStore(settings.vector_path, dim=settings.embedding_dim), kg)
    with TestClient(create_app(mem)) as c:
        yield c


def test_writes_are_rate_limited(client):
    """The middleware was never registered, so no limit had ever applied."""
    from memcore_memory.api.rate_limit import write_limiter
    write_limiter.buckets.clear()

    codes = {client.post("/memory", json={"content": f"m{i}"}).status_code for i in range(25)}
    assert 429 in codes


def test_delete_counts_as_a_write(client):
    """`"/add" in path or "/memory" in path and method == "POST"` binds as
    `or (... and ...)`, so DELETE /memory/{id} was not limited at all."""
    from memcore_memory.api.rate_limit import write_limiter
    write_limiter.buckets.clear()
    for _ in range(20):
        write_limiter.is_allowed("testclient:write")

    r = client.delete("/memory/does-not-matter")
    assert r.status_code == 429
    assert r.headers.get("Retry-After")


def test_reads_are_not_charged_to_the_write_bucket(client):
    from memcore_memory.api.rate_limit import write_limiter
    write_limiter.buckets.clear()
    for _ in range(30):
        assert client.get("/memories").status_code == 200


def test_sync_merge_no_longer_claims_to_have_merged(client):
    """It answered {"status": "merged"} for any payload while merging nothing."""
    r = client.post("/sync/merge", json={"registers": {"a": {"value": 1, "ts": 1, "node": "x"}}})
    assert r.status_code == 501
    assert r.json()["status"] == "not_implemented"


# --- retrieval quality ----------------------------------------------------

from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.retrieval.retrievers.bm25 import BM25Retriever, tokenize


def test_tokenizer_strips_punctuation():
    """`content.lower().split()` made "WireGuard:" and "master.key" whole tokens,
    so no plain query term could match them - BM25 returned nothing for words
    plainly present in the corpus."""
    assert tokenize("WireGuard: two peers, master.key rotated!") == [
        "wireguard", "two", "peers", "master", "key", "rotated"]


def test_tokenizer_keeps_accented_words_whole():
    assert tokenize("Telepítés sikeres") == ["telepítés", "sikeres"]


async def test_bm25_finds_a_term_that_is_followed_by_punctuation(store):
    wanted = MemoryItem(content="NOSTRO HALOZAT - WireGuard: two separate instances.",
                        tier=Tier.EPISODIC)
    other = MemoryItem(content="Something else entirely about printers.", tier=Tier.EPISODIC)
    for item in (wanted, other):
        await store.put(item)

    hits = await BM25Retriever(store).retrieve("WireGuard", k=5)
    assert [h["id"] for h in hits] == [wanted.id]


async def test_query_independent_retrievers_cannot_inject_their_own_candidates(store):
    """TemporalRetriever and ImportanceRetriever rank the whole store identically
    for every query. Fused as equals, the newest high-importance memory came back
    top of every single search."""
    from memcore_memory.storage.vector_store import VectorStore
    from memcore_memory.graph.kg import KnowledgeGraph
    from memcore_memory.retrieval.hybrid import HybridRetriever

    match = MemoryItem(content="the quick brown fox", tier=Tier.EPISODIC,
                       metadata={"importance": 0.1})
    loud = MemoryItem(content="totally unrelated content", tier=Tier.SEMANTIC,
                      metadata={"importance": 0.99})
    for item in (match, loud):
        await store.put(item)

    kg = KnowledgeGraph(store.db_path.with_suffix(".kg2.db"), store.cipher)
    await kg.init()
    retriever = HybridRetriever(store, VectorStore(store.db_path.with_suffix(".vec"), dim=8), kg)
    results = await retriever.search("quick brown fox", k=5)

    assert results, "the lexical match should be found"
    assert results[0]["id"] == match.id
    assert loud.id not in [r["id"] for r in results]
