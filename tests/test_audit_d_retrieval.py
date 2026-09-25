"""Regression tests for the 2026-09-24 audit, group D (retrieval).

F7  graph arm never returned a candidate
F8  tier_filter applied after truncation
F24 BM25 cache ignored content changes
F38 zero-weight vector arm still nominated candidates for the priors
F39 reranker/SPLADE/GraphRAG stubs reported fake success
F41 one recall decrypted the whole store four times plus an N+1 of get()
F42 a failing arm was dropped without a trace
F68 temporal retriever ignores the query (documented, not a search)
F80 metadata arm matched numbers against everything, multi-word queries never
"""

import pytest

from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.retrieval.hybrid import HybridRetriever
from memcore_memory.retrieval.retrievers.bm25 import BM25Retriever
from memcore_memory.retrieval.retrievers.graph import GraphRetriever
from memcore_memory.retrieval.retrievers.metadata import MetadataRetriever
from memcore_memory.retrieval.retrievers.temporal import TemporalRetriever


class _Store:
    """Just enough of EncryptedStore, counting the reads that matter for F41."""

    def __init__(self, items=()):
        self.items = list(items)
        self.list_all_calls = 0
        self.get_calls = 0
        self.get_many_calls = 0

    async def list_all(self):
        self.list_all_calls += 1
        return list(self.items)

    async def get(self, memory_id):
        self.get_calls += 1
        return next((m for m in self.items if m.id == memory_id), None)

    async def get_many(self, ids):
        self.get_many_calls += 1
        wanted = set(ids)
        return {m.id: m for m in self.items if m.id in wanted}


class _KG:
    """Implements the get_related_memories contract: entity names found in the query."""

    def __init__(self, links=None):
        self.links = links or {}
        self.calls = []

    async def get_related_memories(self, query, limit=20):
        self.calls.append((query, limit))
        q = query.lower()
        out = []
        for entity, ids in self.links.items():
            if entity in q:
                out.extend(i for i in ids if i not in out)
        return out[:limit]


class _VectorStore:
    dim = 8

    def __init__(self, ids=()):
        self.ids = list(ids)
        self.searches = 0

    async def search(self, vec, k=10):
        self.searches += 1
        return [(i, 0.5) for i in self.ids[:k]]


def _item(content, tier=Tier.EPISODIC, importance=0.5, **meta):
    return MemoryItem(content=content, tier=tier, metadata={"importance": importance, **meta})


def _hybrid(store, kg=None, vs=None):
    # No embedder: the hash-fallback configuration every SkyNAS deployment runs.
    return HybridRetriever(store, vs or _VectorStore(), kg or _KG())


# --- F7: graph arm ---------------------------------------------------------

async def test_graph_retriever_asks_the_graph_with_the_whole_query():
    kg = _KG({"skynas": ["m1"], "home assistant": ["m2"]})
    hits = await GraphRetriever(kg).retrieve("where does Home Assistant run on SkyNAS,", k=5)
    assert kg.calls == [("where does Home Assistant run on SkyNAS,", 5)]
    assert {h["id"] for h in hits} == {"m1", "m2"}
    assert all(h["source"] == "graph" for h in hits)


async def test_graph_retriever_fails_loudly_without_the_method():
    with pytest.raises(AttributeError):
        await GraphRetriever(object()).retrieve("anything", k=5)


async def test_entity_tagged_memory_is_found_by_hybrid_search_without_the_word_in_content():
    tagged = _item("the box under the desk")
    other = _item("groceries and milk")
    store = _Store([tagged, other])
    results = await _hybrid(store, kg=_KG({"skynas": [tagged.id]})).search("SkyNAS", k=5)
    assert [r["id"] for r in results] == [tagged.id]


# --- F8: tier_filter before truncation ------------------------------------

async def test_tier_filter_finds_a_match_that_ranks_below_the_top_2k():
    filler = " ".join(f"filler{i}" for i in range(200))
    working = _item(f"alpha protocol {filler}", tier=Tier.WORKING)
    episodic = [_item(f"alpha protocol note {i}") for i in range(30)]
    store = _Store([working] + episodic)
    retriever = _hybrid(store)

    unfiltered = await retriever.search("alpha protocol", k=5)
    assert working.id not in [r["id"] for r in unfiltered], "test must exercise the truncation path"

    filtered = await retriever.search("alpha protocol", k=5, tier_filter=["working"])
    assert [r["id"] for r in filtered] == [working.id]


async def test_tier_filter_with_no_member_of_that_tier_returns_nothing():
    store = _Store([_item("alpha protocol")])
    assert await _hybrid(store).search("alpha", k=5, tier_filter=["semantic"]) == []


# --- F24: BM25 cache ------------------------------------------------------

async def test_bm25_index_is_rebuilt_when_content_changes_under_the_same_id():
    item = _item("the old content mentions xyzzy")
    store = _Store([item, _item("unrelated cats")])
    r = BM25Retriever(store)
    assert [x["id"] for x in await r.retrieve("xyzzy")] == [item.id]
    item.content = "the new content mentions plugh"
    assert await r.retrieve("xyzzy") == []
    assert [x["id"] for x in await r.retrieve("plugh")] == [item.id]


# --- F38: zero-weight arm -------------------------------------------------

async def test_zero_weight_vector_arm_neither_runs_nor_nominates():
    zebra = _item("the zebra at the zoo has black and white stripes", importance=0.3)
    diary = [_item(f"diary entry {i} about the weather", importance=0.9) for i in range(30)]
    store = _Store([zebra] + diary)
    vs = _VectorStore([m.id for m in diary])
    retriever = _hybrid(store, vs=vs)
    assert retriever.weights["vector"] == 0.0

    results = await retriever.search("zebra", k=10)
    assert [r["id"] for r in results] == [zebra.id]
    assert vs.searches == 0


async def test_a_query_nothing_matches_returns_nothing_rather_than_priors():
    store = _Store([_item(f"diary entry {i}", importance=0.9) for i in range(10)])
    assert await _hybrid(store).search("unicorn", k=5) == []


# --- F41: one snapshot per search -----------------------------------------

async def test_search_lists_the_store_once_and_resolves_ids_in_one_batch():
    store = _Store([_item(f"kittens are playful {i}") for i in range(20)])
    results = await _hybrid(store).search("kittens", k=5)
    assert len(results) == 5
    assert store.list_all_calls == 1
    assert store.get_many_calls == 1
    assert store.get_calls == 0


# --- F42: failing arms are reported ---------------------------------------

async def test_a_failing_arm_is_logged_to_stderr_and_the_rest_still_answer(capsys):
    match = _item("kittens are playful animals")
    store = _Store([match, _item("grocery list")])
    kg = _KG({"kittens": [match.id]})
    retriever = _hybrid(store, kg=kg)

    async def broken(*a, **kw):
        raise RuntimeError("InvalidTag")
    retriever.bm25.retrieve = broken

    results = await retriever.search("kittens", k=5)
    captured = capsys.readouterr()
    assert "bm25 arm failed" in captured.err
    assert "InvalidTag" in captured.err
    assert captured.out == ""
    assert [r["id"] for r in results] == [match.id]


async def test_all_query_dependent_arms_failing_is_an_error_not_an_empty_success():
    store = _Store([_item("kittens")])
    retriever = _hybrid(store)

    async def broken(*a, **kw):
        raise RuntimeError("boom")
    for arm in (retriever.bm25, retriever.graph, retriever.metadata):
        arm.retrieve = broken

    with pytest.raises(RuntimeError, match="every query-dependent retriever failed"):
        await retriever.search("kittens", k=5)


# --- F68: temporal is a prior ---------------------------------------------

async def test_temporal_retriever_is_documented_as_query_independent():
    store = _Store([_item("a"), _item("b")])
    t = TemporalRetriever(store)
    ids = lambda hits: [h["id"] for h in hits]
    assert ids(await t.retrieve("WireGuard", 5)) == ids(await t.retrieve("grocery", 5))
    assert "ignored" in TemporalRetriever.retrieve.__doc__


# --- F80: metadata tokens -------------------------------------------------

@pytest.mark.parametrize("query,hits", [
    ("0.5", 0),                    # every memory has importance 0.5
    ("5", 0),
    ("0", 0),
    ("claude-export hiking", 3),   # one token matches a value
    ("export", 3),
    ("expo", 0),                   # substring only
    ("", 0),
])
async def test_metadata_matches_query_tokens_against_string_values(query, hits):
    store = _Store([_item(f"m{i}", source="claude-export", flag=True, n=5) for i in range(3)])
    assert len(await MetadataRetriever(store).retrieve(query, k=10)) == hits


async def test_metadata_matches_string_lists_and_exact_filters():
    tagged = _item("x", tags=["WireGuard", "vpn"])
    store = _Store([tagged, _item("y", source="other")])
    r = MetadataRetriever(store)
    assert [h["id"] for h in await r.retrieve("wireguard config", k=5)] == [tagged.id]
    filtered = await r.retrieve("nothing", k=5, filters={"source": "other"})
    assert len(filtered) == 1 and filtered[0]["id"] != tagged.id


# --- F39: experimental stubs don't fake success ---------------------------

async def test_splade_and_graphrag_stubs_raise():
    from memcore_memory.retrieval.reranker import SPLADERetriever, GraphRAG
    with pytest.raises(NotImplementedError):
        await SPLADERetriever().retrieve("q")
    with pytest.raises(NotImplementedError):
        await GraphRAG(None).community_summarize(["a", "b"])


def test_rerank_does_not_mutate_the_callers_results():
    from memcore_memory.retrieval.reranker import CrossEncoderReranker

    class _Model:
        def predict(self, pairs):
            return [0.1, 0.9]

    rr = CrossEncoderReranker.__new__(CrossEncoderReranker)
    rr.model_name, rr.device, rr._model = "fake", None, _Model()
    docs = [{"id": "1", "content": "a", "score": 0.8}, {"id": "2", "content": "b", "score": 0.6}]
    out = rr.rerank("q", docs, top_k=2)
    assert [d["id"] for d in out] == ["2", "1"]
    assert docs == [{"id": "1", "content": "a", "score": 0.8}, {"id": "2", "content": "b", "score": 0.6}]


def test_experimental_modules_say_so():
    from memcore_memory.retrieval import reranker
    from memcore_memory.embeddings import matryoshka
    assert "NOT WIRED IN" in reranker.__doc__
    assert "NOT WIRED IN" in matryoshka.__doc__


# --- integration through the real store, graph and vector store ------------

async def test_real_components_end_to_end(tmp_path):
    """Exercises the cross-group contracts: EncryptedStore.get_many and
    KnowledgeGraph.get_related_memories, with the hash embedder."""
    from memcore_memory.core.memory import MnemosyneMemory
    from memcore_memory.crypto.aes_gcm import AES256GCM
    from memcore_memory.embeddings.factory import get_embedder
    from memcore_memory.graph.kg import KnowledgeGraph
    from memcore_memory.storage.encrypted_sqlite import EncryptedStore
    from memcore_memory.storage.vector_store import VectorStore

    cipher = AES256GCM(AES256GCM.generate_key())
    store = EncryptedStore(tmp_path / "m.db", cipher)
    await store.init()
    kg = KnowledgeGraph(tmp_path / "m.kg.db", cipher)
    await kg.init()
    mem = MnemosyneMemory(store, VectorStore(tmp_path / "m.vec", dim=8), kg,
                          embedder=get_embedder("local", dim=8))

    tagged = await mem.add("the box under the desk", entities=["SkyNAS"])
    zebra = await mem.add("the zebra at the zoo", importance=0.2)
    for i in range(20):
        await mem.add(f"diary entry {i}", importance=0.9)

    assert [r["id"] for r in await mem.retriever.search("SkyNAS", k=5)] == [tagged.id]
    assert [r["id"] for r in await mem.retriever.search("zebra", k=10)] == [zebra.id]
    assert await mem.retriever.search("unicorn", k=10) == []
    stored_tier = (await store.get(zebra.id)).tier.value
    other_tier = "semantic" if stored_tier != "semantic" else "episodic"
    assert [r["id"] for r in await mem.retriever.search("zebra", k=5, tier_filter=[stored_tier])] == [zebra.id]
    assert await mem.retriever.search("zebra", k=5, tier_filter=[other_tier]) == []
