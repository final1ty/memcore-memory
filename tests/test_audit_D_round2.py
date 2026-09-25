"""Round 2 of the 2026-09-24 audit, group D (retrieval, embeddings).

F80-regress  a bookkeeping metadata value nominated every imported memory
priors-depth priors were cut to the store-wide top 2k before being fused
F79          search results say whether a query-dependent arm matched them
F23          BGEEmbedder let a configured dim override the model's real one
F68          the temporal prior can be called without a query
"""

import sys
import types

import pytest

from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.retrieval.hybrid import HybridRetriever
from memcore_memory.retrieval.retrievers.metadata import MetadataRetriever
from memcore_memory.retrieval.retrievers.temporal import TemporalRetriever
from memcore_memory.retrieval.retrievers.importance import ImportanceRetriever


class _Store:
    def __init__(self, items=()):
        self.items = list(items)

    async def list_all(self):
        return list(self.items)

    async def get(self, memory_id):
        return next((m for m in self.items if m.id == memory_id), None)

    async def get_many(self, ids):
        wanted = set(ids)
        return {m.id: m for m in self.items if m.id in wanted}


class _KG:
    def __init__(self, links=None):
        self.links = links or {}

    async def get_related_memories(self, query, limit=20):
        q = query.lower()
        out = []
        for entity, ids in self.links.items():
            if entity in q:
                out.extend(i for i in ids if i not in out)
        return out[:limit]


class _VectorStore:
    dim = 8

    async def search(self, vec, k=10):
        return []


def _item(content, tier=Tier.EPISODIC, importance=0.5, **meta):
    return MemoryItem(content=content, tier=tier, metadata={"importance": importance, **meta})


def _imported(content, i):
    # What scripts/sync-rest-to-local.py writes on every record it copies.
    return _item(content, imported_from="docker-rest-store", origin_id=f"origin-{i}")


# --- F80-regress: provenance and store-wide tokens ------------------------

async def test_import_marker_does_not_make_every_imported_memory_a_candidate():
    items = [_imported(f"diary entry {i} about nothing", i) for i in range(20)]
    zebra = _imported("the zebra at the zoo", 99)
    store = _Store(items + [zebra])
    assert await MetadataRetriever(store).retrieve("docker store setup", k=10) == []

    hybrid = HybridRetriever(store, _VectorStore(), _KG())
    assert await hybrid.search("docker store setup", k=10) == []
    assert [r["id"] for r in await hybrid.search("zebra docker", k=10)] == [zebra.id]


async def test_any_key_ending_in_id_is_provenance():
    store = _Store([_item("x", session_id="alpha-beta")])
    assert await MetadataRetriever(store).retrieve("alpha", k=5) == []


async def test_a_token_most_of_the_store_carries_nominates_nothing():
    # Every live memory carries source="claude-code-session".
    items = [_item(f"note {i}", source="claude-code-session") for i in range(20)]
    vpn = _item("tunnel config", source="claude-code-session", tags=["wireguard"])
    store = _Store(items + [vpn])
    r = MetadataRetriever(store)
    assert [h["id"] for h in await r.retrieve("claude session wireguard", k=10)] == [vpn.id]
    assert await r.retrieve("claude session", k=10) == []


async def test_a_shared_tag_on_a_small_store_still_matches():
    store = _Store([_item(f"m{i}", source="claude-export") for i in range(3)])
    assert len(await MetadataRetriever(store).retrieve("export", k=10)) == 3


async def test_a_tag_on_a_minority_of_a_large_store_still_matches():
    tagged = [_item(f"t{i}", project="skynas") for i in range(5)]
    store = _Store(tagged + [_item(f"u{i}", project="other") for i in range(15)])
    hits = await MetadataRetriever(store).retrieve("skynas", k=10)
    assert {h["id"] for h in hits} == {m.id for m in tagged}


# --- priors-depth: priors rank the candidates, not the store --------------

def _router_store():
    diary = [_item(f"diary entry {i} about the day", importance=0.9) for i in range(20)]
    note = _item("note about the router", importance=0.5)
    return diary, note, _KG({"home assistant": [note.id]})


async def test_a_graph_and_bm25_top_hit_is_not_buried_by_important_stopword_matches():
    diary, note, kg = _router_store()
    hybrid = HybridRetriever(_Store(diary + [note]), _VectorStore(), kg)
    results = await hybrid.search("what about Home Assistant?", k=5)
    assert results[0]["id"] == note.id


async def test_a_tier_filter_covering_everything_does_not_change_the_order():
    diary, note, kg = _router_store()
    hybrid = HybridRetriever(_Store(diary + [note]), _VectorStore(), kg)
    plain = await hybrid.search("what about Home Assistant?", k=10)
    filtered = await hybrid.search("what about Home Assistant?", k=10,
                                   tier_filter=[t.value for t in Tier])
    assert [r["id"] for r in plain] == [r["id"] for r in filtered]


async def test_priors_only_see_the_candidates():
    diary, note, kg = _router_store()
    hybrid = HybridRetriever(_Store(diary + [note]), _VectorStore(), kg)
    seen = []
    real = hybrid.temporal.retrieve

    async def spy(query="", k=20, items=None):
        seen.append({m.id for m in items})
        return await real(query, k, items=items)

    hybrid.temporal.retrieve = spy
    await hybrid.search("router", k=5)
    assert seen == [{note.id}]


async def test_a_failing_prior_is_logged_and_the_search_still_answers(capsys):
    diary, note, kg = _router_store()
    hybrid = HybridRetriever(_Store(diary + [note]), _VectorStore(), kg)

    async def boom(*a, **kw):
        raise RuntimeError("prior broke")

    hybrid.importance.retrieve = boom
    assert [r["id"] for r in await hybrid.search("router", k=5)] == [note.id]
    assert "importance arm failed" in capsys.readouterr().err


# --- F79: matched flag -----------------------------------------------------

async def test_every_result_says_it_was_matched():
    diary, note, kg = _router_store()
    hybrid = HybridRetriever(_Store(diary + [note]), _VectorStore(), kg)
    results = await hybrid.search("diary", k=5)
    assert results and all(r["matched"] is True for r in results)


# --- F68: the temporal prior needs no query --------------------------------

async def test_priors_can_be_called_without_a_query():
    store = _Store([_item("a"), _item("b")])
    assert len(await TemporalRetriever(store).retrieve(k=5)) == 2
    assert len(await ImportanceRetriever(store).retrieve(k=5)) == 2


# --- F23: the model's own width wins, a mismatch is an error ---------------

class _FakeModel:
    native = 768

    def __init__(self, name, device=None):
        if name == "broken/model":
            raise OSError("no such model")
        self.name = name

    def get_sentence_embedding_dimension(self):
        return self.native

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np
        return np.zeros((len(texts), self.native))


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = _FakeModel
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "torch", torch)


def test_configured_dim_that_disagrees_with_the_model_raises(fake_sentence_transformers):
    from memcore_memory.embeddings.bge import BGEEmbedder
    with pytest.raises(ValueError, match="MNEM_EMBEDDING_DIM=768"):
        BGEEmbedder("BAAI/bge-base-en-v1.5", dim=384)


def test_without_a_configured_dim_the_model_decides(fake_sentence_transformers):
    from memcore_memory.embeddings.bge import BGEEmbedder
    e = BGEEmbedder("BAAI/bge-base-en-v1.5", dim=None)
    assert e.dim == 768 and e._model is not None
    assert len(e.embed(["x"])[0]) == 768


def test_matching_dim_loads_the_model(fake_sentence_transformers):
    from memcore_memory.embeddings.bge import BGEEmbedder
    e = BGEEmbedder("BAAI/bge-base-en-v1.5", dim=768)
    assert e.dim == 768 and e._model is not None


def test_a_model_that_fails_to_load_keeps_the_configured_dim(fake_sentence_transformers):
    # The live store holds 384-dim hash vectors; the fallback must keep that width.
    from memcore_memory.embeddings.bge import BGEEmbedder
    e = BGEEmbedder("broken/model", dim=384)
    assert e.dim == 384 and e._model is None
    assert len(e.embed(["x"])[0]) == 384


def test_missing_sentence_transformers_keeps_the_configured_dim(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from memcore_memory.embeddings.bge import BGEEmbedder
    e = BGEEmbedder("BAAI/bge-small-en-v1.5", dim=384)
    assert e.dim == 384 and e._model is None
    assert len(e.embed(["x"])[0]) == 384
