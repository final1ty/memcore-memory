"""Audit group C: tier lifecycle, MnemosyneMemory core behaviour and Settings.

Every test runs against the per-test data dir from conftest's autouse
``isolate_data_dir``. Rows are aged by rewriting ``timestamp`` and
``forgetting.last_access`` and putting them back, which is the only thing the
lifecycle rules look at.
"""

import json
import math
import os
import subprocess
import sys
import textwrap
import time

import aiosqlite
import pytest

from memcore_memory import create_memory_system
from memcore_memory.config import Settings, settings
from memcore_memory.core.ebbinghaus import ForgettingCurve, POWER_LAW
from memcore_memory.core.memory import MnemosyneMemory
from memcore_memory.core.tiers import MemoryItem, Tier, TierManager, TIER_BASE_STRENGTH

DAY = 86400.0


@pytest.fixture
async def mem():
    return await create_memory_system()


async def age(mem, item_id, seconds):
    item = await mem.store.get(item_id)
    item.timestamp -= seconds
    item.forgetting.last_access -= seconds
    await mem.store.put(item)


async def tier_of(mem, item_id):
    item = await mem.store.get(item_id)
    return item.tier if item else None


# --- lifecycle: forget_expired (F12, F13, F37, F78) -------------------------

@pytest.mark.parametrize("importance", [0.5, 1.0])
async def test_expired_explicit_working_is_demoted_not_deleted(mem, importance):
    item = await mem.add("short-lived note", tier="working", importance=importance)
    await age(mem, item.id, 4200)
    assert await mem.forget_expired() == 0
    loaded = await mem.store.get(item.id)
    assert loaded is not None and loaded.tier == Tier.EPISODIC
    assert loaded.forgetting.strength >= TIER_BASE_STRENGTH[Tier.EPISODIC]


async def test_expired_sensory_is_still_deleted(mem):
    item = await mem.add("flash", tier="sensory")
    await age(mem, item.id, 60)
    assert await mem.forget_expired() == 1
    assert await mem.store.get(item.id) is None


async def test_promoted_item_is_not_deleted_in_the_same_pass(mem):
    rehearsed = await mem.add("recalled once", tier="working")
    loaded = await mem.store.get(rehearsed.id)
    loaded.touch()
    await mem.store.put(loaded)
    await age(mem, rehearsed.id, 3 * DAY)

    report = await mem.lifecycle_pass()
    assert report["promoted"] == 1 and report["forgotten"] == 0
    assert await tier_of(mem, rehearsed.id) == Tier.EPISODIC


async def test_semantic_is_never_deleted_automatically(mem):
    item = await mem.add("core fact", tier="semantic")
    await age(mem, item.id, 100 * 365 * DAY)
    assert (await mem.store.get(item.id)).forgetting.retention() < 0.01
    assert await mem.forget_expired() == 0
    assert await tier_of(mem, item.id) == Tier.SEMANTIC


async def test_stale_episodic_is_forgotten(mem):
    item = await mem.add("old episode", tier="episodic")
    await age(mem, item.id, 400 * DAY)
    assert await mem.forget_expired() == 1
    assert await mem.store.get(item.id) is None


async def _rehearsed_episodic(mem, importance):
    item = await mem.add(f"episode {importance}", tier="episodic", importance=importance)
    loaded = await mem.store.get(item.id)
    for _ in range(settings.semantic_consolidation_threshold):
        loaded.touch()
    await mem.store.put(loaded)
    return item.id


async def test_semantic_promotion_needs_importance(mem):
    low = await _rehearsed_episodic(mem, 0.0)
    high = await _rehearsed_episodic(mem, 0.9)
    assert await mem.forget_expired() == 0
    assert await tier_of(mem, low) == Tier.EPISODIC
    assert await tier_of(mem, high) == Tier.SEMANTIC


async def test_consolidate_returns_its_count_and_uses_the_same_rule(mem):
    await _rehearsed_episodic(mem, 0.0)
    high = await _rehearsed_episodic(mem, 0.9)
    assert await mem.consolidate() == 1
    assert await tier_of(mem, high) == Tier.SEMANTIC
    assert await mem.consolidate() == 0


async def test_one_bad_row_does_not_abort_the_pass(mem, monkeypatch):
    bad = await mem.add("will fail", tier="working")
    sensory = [await mem.add(f"flash {i}", tier="sensory") for i in range(3)]
    # Aged after the adds: an add demotes expired working memories itself.
    await age(mem, bad.id, 4200)
    for s in sensory:
        await age(mem, s.id, 60)

    real = mem.store.update_tier

    async def flaky(memory_id, tier):
        if memory_id == bad.id:
            raise RuntimeError("boom")
        return await real(memory_id, tier)

    monkeypatch.setattr(mem.store, "update_tier", flaky)
    report = await mem.lifecycle_pass()
    assert report["forgotten"] == 3
    assert [e["id"] for e in report["errors"]] == [bad.id]
    assert await mem.store.get(bad.id) is not None


# --- TierManager rules, one case per branch ---------------------------------

def _item(tier, age_s=0.0, rehearsals=0, importance=0.5, **meta):
    now = time.time()
    curve = ForgettingCurve(strength=TIER_BASE_STRENGTH[tier], last_access=now - age_s,
                            rehearsals=rehearsals, importance=importance)
    return MemoryItem(content="x", tier=tier, timestamp=now - age_s,
                      metadata={"importance": importance, **meta}, forgetting=curve)


def test_tier_manager_branches():
    tm = TierManager(settings)
    assert tm.assign_tier(_item(Tier.EPISODIC)) == Tier.WORKING
    assert tm.assign_tier(_item(Tier.EPISODIC, sensory=True)) == Tier.SENSORY
    assert tm.assign_tier(_item(Tier.EPISODIC), {"tier": "semantic"}) == Tier.SEMANTIC

    assert tm.should_promote(_item(Tier.SENSORY, attended=True)) == Tier.WORKING
    assert tm.should_promote(_item(Tier.SENSORY)) is None
    assert tm.should_promote(_item(Tier.WORKING, rehearsals=1)) == Tier.EPISODIC
    assert tm.should_promote(_item(Tier.EPISODIC, rehearsals=3, importance=0.9)) == Tier.SEMANTIC
    assert tm.should_promote(_item(Tier.EPISODIC, rehearsals=3, importance=0.5)) is None
    assert tm.should_promote(_item(Tier.EPISODIC, age_s=60 * DAY, rehearsals=3, importance=0.9)) is None

    assert tm.should_demote_or_forget(_item(Tier.SENSORY, age_s=60)) == "forget"
    assert tm.should_demote_or_forget(_item(Tier.SENSORY)) is None
    assert tm.should_demote_or_forget(_item(Tier.WORKING, age_s=4200)) == "demote"
    assert tm.should_demote_or_forget(_item(Tier.WORKING, age_s=10 * DAY)) == "demote"
    assert tm.should_demote_or_forget(_item(Tier.EPISODIC, age_s=400 * DAY)) == "forget"
    assert tm.should_demote_or_forget(_item(Tier.EPISODIC)) is None
    assert tm.should_demote_or_forget(_item(Tier.SEMANTIC, age_s=1e5 * DAY)) is None


# --- working capacity (F22, F35) ---------------------------------------------

async def test_working_cap_holds_across_instances(mem):
    other = await create_memory_system()
    for i in range(10):
        await (mem if i % 2 else other).add(f"note {i}")
    working = await mem.store.list_by_tier(Tier.WORKING)
    assert len(working) == settings.working_capacity
    # Newest stay; every auto-assigned working memory starts at the working baseline.
    assert {w.content for w in working} == {f"note {i}" for i in range(3, 10)}
    assert all(w.forgetting.strength == TIER_BASE_STRENGTH[Tier.WORKING] for w in working)
    assert len(await mem.store.list_by_tier(Tier.EPISODIC)) == 3


async def test_eviction_never_demotes_a_promoted_memory(mem):
    core = await mem.add("core fact")
    assert core.tier == Tier.WORKING
    await mem.store.update_tier(core.id, Tier.SEMANTIC)
    for i in range(settings.working_capacity + 1):
        await mem.add(f"scratch {i}", tier="working")
    assert await tier_of(mem, core.id) == Tier.SEMANTIC


async def test_deleted_working_memories_free_their_slots(mem):
    ids = [(await mem.add(f"w {i}")).id for i in range(settings.working_capacity)]
    for i in ids:
        assert await mem.delete(i) is True
    assert (await mem.add("fresh")).tier == Tier.WORKING


# --- add() (F23, F36, F76, F77) ---------------------------------------------

@pytest.mark.parametrize("bad", [-1.0, 1.5, 5.0, float("nan"), None, "high"])
async def test_importance_out_of_range_is_rejected(mem, bad):
    with pytest.raises(ValueError):
        await mem.add("x", importance=bad)
    assert await mem.store.list_all() == []


def test_curve_clamps_out_of_range_importance_on_read():
    assert ForgettingCurve(strength=7.0, importance=-1.0).effective_strength() > 0
    assert ForgettingCurve(strength=7.0, importance=5.0).effective_strength() == pytest.approx(14.0)


async def test_add_does_not_mutate_or_share_the_callers_metadata(mem):
    md = {"source": "x"}
    a = await mem.add("m1", metadata=md, importance=0.9)
    b = await mem.add("m2", metadata=md, importance=0.1)
    assert md == {"source": "x"}
    assert a.metadata["importance"] == 0.9
    assert a.metadata is not b.metadata


async def test_add_embeds_as_a_document(mem, monkeypatch):
    calls = []
    real = mem.embedder

    class Recording:
        dim = real.dim

        def embed(self, texts):
            return real.embed(texts)

        def embed_query(self, text):
            calls.append(("query", text))
            return real.embed([text])[0]

        def embed_documents(self, texts):
            calls.append(("documents", texts))
            return real.embed(texts)

    monkeypatch.setattr(mem, "embedder", Recording())
    await mem.add("SkyNAS runs Docker")
    assert calls == [("documents", ["SkyNAS runs Docker"])]


async def test_vector_meta_carries_no_content(mem):
    item = await mem.add("secret content")
    assert mem.vectors.id_to_meta[item.id] == {"tier": "working"}


async def test_failed_vector_write_leaves_no_orphan_row(mem, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("vector store down")

    monkeypatch.setattr(mem.vectors, "add", boom)
    with pytest.raises(RuntimeError):
        await mem.add("first note")
    assert await mem.store.list_all() == []


async def test_wrong_dimension_is_refused_before_anything_is_stored(mem, monkeypatch):
    monkeypatch.setattr(mem.embedder, "embed_documents", lambda texts: [[0.1] * 768 for _ in texts])
    with pytest.raises(ValueError, match="768"):
        await mem.add("first note")
    assert await mem.store.list_all() == []


async def test_mismatched_embedder_and_vector_store_dims_are_refused(mem):
    class Wide:
        dim = mem.vectors.dim * 2

        def embed(self, texts):
            return [[0.0] * self.dim for _ in texts]

    with pytest.raises(ValueError, match="dim"):
        MnemosyneMemory(mem.store, mem.vectors, mem.kg, embedder=Wide())


# --- delete, get, recall (contract 3, F79) ----------------------------------

async def test_delete_removes_row_and_vector(mem):
    item = await mem.add("to delete")
    assert await mem.delete(item.id) is True
    assert await mem.store.get(item.id) is None
    assert item.id not in mem.vectors.ids
    assert await mem.delete(item.id) is False


async def test_get_rehearses_without_rewriting_the_row(mem, monkeypatch):
    item = await mem.add("read me")

    async def no_put(*a, **k):
        raise AssertionError("get() must not put the whole row")

    monkeypatch.setattr(mem.store, "put", no_put)
    await mem.get(item.id)
    assert (await mem.store.get(item.id)).forgetting.rehearsals == 1


async def test_recall_rehearses_only_matched_results_and_reports_retention_after(mem, monkeypatch):
    items = [await mem.add(f"alpha fact {i}", tier="episodic") for i in range(4)]
    for it in items:
        await age(mem, it.id, 2 * DAY)
    before = {it.id: (await mem.store.get(it.id)).forgetting.retention() for it in items}

    async def fake_search(query, k=10, tier_filter=None):
        return [{"id": it.id, "retention": before[it.id], "matched": i != 0}
                for i, it in enumerate(items)]

    monkeypatch.setattr(mem.retriever, "search", fake_search)
    results = await mem.recall("alpha")
    rehearsals = [(await mem.store.get(it.id)).forgetting.rehearsals for it in items]
    # Only the best matched hit adds to the count promotion looks at.
    assert rehearsals == [0, 1, 0, 0]
    # Lower ranks count for less.
    strengths = [(await mem.store.get(it.id)).forgetting.strength for it in items[1:]]
    assert strengths[0] > strengths[1] > strengths[2]
    assert results[1]["retention"] == pytest.approx(1.0, abs=1e-3)
    assert results[1]["retention"] > before[items[1].id]

    await mem.recall("alpha", rehearse=False)
    assert (await mem.store.get(items[1].id)).forgetting.rehearsals == 1


# --- importance kept in one place (F64) -------------------------------------

async def test_metadata_importance_drives_decay_after_an_update(mem):
    item = await mem.add("x", importance=0.1)
    loaded = await mem.store.get(item.id)
    loaded.metadata.update({"importance": 1.0})  # what MCP memory_update does
    await mem.store.put(loaded)
    reloaded = await mem.store.get(item.id)
    assert reloaded.forgetting.importance == 1.0
    fresh = MemoryItem(content="y", tier=reloaded.tier, metadata={"importance": 1.0})
    fresh.forgetting.strength = reloaded.forgetting.strength
    assert reloaded.forgetting.effective_strength() == pytest.approx(fresh.forgetting.effective_strength())


def test_set_importance_updates_both_copies_and_validates():
    item = MemoryItem(content="x", metadata={"importance": 0.1})
    item.set_importance(0.9)
    assert item.metadata["importance"] == item.forgetting.importance == 0.9
    with pytest.raises(ValueError):
        item.set_importance(2)


# --- forgetting model and stored-format compatibility (F65, R25) ------------

async def test_configured_forgetting_model_reaches_new_memories(mem, monkeypatch):
    monkeypatch.setattr(settings, "forgetting_model", POWER_LAW)
    item = await mem.add("power law memory")
    assert (await mem.store.get(item.id)).forgetting.decay_model == POWER_LAW


async def test_rows_written_by_the_previous_code_still_load(mem):
    item = await mem.add("legacy row", tier="episodic")
    legacy = [
        # exactly what the previous to_dict wrote
        {"strength": 7.0, "last_access": 1.7e9, "rehearsals": 2, "importance": 0.5,
         "decay_model": "exponential", "power_d": 0.3},
        # rows from before decay_model/power_d existed
        {"strength": 7.0, "last_access": 1.7e9, "rehearsals": 0, "importance": 0.5},
        # integer JSON numbers
        {"strength": 7, "last_access": 1700000000, "rehearsals": 1, "importance": 1},
    ]
    for data in legacy:
        async with aiosqlite.connect(settings.db_path) as db:
            await db.execute("UPDATE memories SET forgetting_json=? WHERE id=?", (json.dumps(data), item.id))
            await db.commit()
        loaded = await mem.store.get(item.id)
        assert loaded.forgetting.strength == float(data["strength"])
        assert loaded.forgetting.rehearsals == data["rehearsals"]
        assert loaded.forgetting.decay_model == "exponential"
        assert 0.0 <= loaded.forgetting.retention() <= 1.0


def test_from_dict_coerces_types_and_names_the_bad_field():
    assert ForgettingCurve.from_dict({"strength": "7.0"}).strength == 7.0
    with pytest.raises(ValueError, match="strength"):
        ForgettingCurve.from_dict({"strength": "bad"})
    with pytest.raises(ValueError, match="rehearsals"):
        ForgettingCurve.from_dict({"rehearsals": [1]})
    with pytest.raises(ValueError, match="decay_model"):
        ForgettingCurve.from_dict({"decay_model": "linear"})
    assert ForgettingCurve.from_dict({"strength": None}, default_last_access=5.0).last_access == 5.0


# --- Settings (R14, R15, R28, contract 4) -----------------------------------

def test_backend_is_a_closed_set_with_aliases(tmp_path):
    assert Settings(data_dir=tmp_path, backend="postgresql").backend == "postgres"
    assert Settings(data_dir=tmp_path, backend=" SQLite ").backend == "sqlite"
    with pytest.raises(Exception):
        Settings(data_dir=tmp_path, backend="mysql")


async def test_create_memory_system_rejects_unknown_backend():
    with pytest.raises(ValueError, match="backend"):
        await create_memory_system(backend="mysql")


def test_unknown_kwargs_and_forgetting_models_are_rejected(tmp_path):
    with pytest.raises(Exception):
        Settings(data_dir=tmp_path, working_capcity=3)
    with pytest.raises(Exception):
        Settings(data_dir=tmp_path, forgetting_model="powerlaw")


def test_unknown_mnem_env_var_warns_on_stderr_only(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MNEM_DATADIR", "/x")
    Settings(data_dir=tmp_path)
    out, err = capsys.readouterr()
    assert "MNEM_DATADIR" in err
    assert out == ""


def test_api_key_setting(tmp_path, monkeypatch):
    assert Settings(data_dir=tmp_path).api_key is None
    monkeypatch.setenv("MNEM_API_KEY", "s3cret")
    s = Settings(data_dir=tmp_path)
    assert s.api_key == "s3cret"
    assert "s3cret" not in repr(s)


def test_tilde_in_data_dir_is_expanded(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = Settings(data_dir="~/x")
    assert s.data_dir == tmp_path / "x"
    assert s.key_path == tmp_path / "x" / "master.key"


def test_p2p_peers_accepts_a_comma_list_and_json(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEM_P2P_PEERS", "http://a:1, http://b:2")
    assert Settings(data_dir=tmp_path).p2p_peers == ["http://a:1", "http://b:2"]
    monkeypatch.setenv("MNEM_P2P_PEERS", '["http://a:1"]')
    assert Settings(data_dir=tmp_path).p2p_peers == ["http://a:1"]


def test_import_creates_no_data_dir(tmp_path):
    target = tmp_path / "missing"
    script = "import memcore_memory, memcore_memory.cli.main"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True,
                   env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"),
                        "MNEM_DATA_DIR": str(target)})
    assert not target.exists()
    assert not (tmp_path / "home" / ".memcore").exists()


def test_ensure_data_dir_creates_it_private(tmp_path):
    s = Settings(data_dir=tmp_path / "fresh")
    old = os.umask(0o022)
    try:
        s.ensure_data_dir()
    finally:
        os.umask(old)
    assert (tmp_path / "fresh").stat().st_mode & 0o777 == 0o700


async def test_create_memory_system_creates_the_data_dir(tmp_path, monkeypatch):
    # The key lives elsewhere, so nothing but create_memory_system can make the dir.
    fresh = tmp_path / "new-store"
    monkeypatch.setattr(settings, "data_dir", fresh)
    for field, name in {"db_path": "memory.db", "vector_path": "vectors.hnsw"}.items():
        monkeypatch.setattr(settings, field, fresh / name)
    monkeypatch.setattr(settings, "key_path", tmp_path / "master.key")
    mem = await create_memory_system()
    await mem.add("hello")
    assert fresh.is_dir()
