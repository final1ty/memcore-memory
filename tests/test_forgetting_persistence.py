"""Regression tests for rehearsal persistence.

``MemoryItem.__post_init__`` used to overwrite ``forgetting.strength`` from the tier
on *every* construction, including the one inside ``EncryptedStore._row_to_item``.
Rehearsals were written to SQLite correctly and then discarded on the next read, so
the Ebbinghaus curve - the core feature of the system - never strengthened with use.
A memory recalled a hundred times decayed on exactly the same schedule as one never
touched again.
"""

import pytest

from memcore_memory.core.ebbinghaus import ForgettingCurve
from memcore_memory.core.tiers import MemoryItem, Tier, TIER_BASE_STRENGTH
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.storage.encrypted_sqlite import EncryptedStore


@pytest.fixture
async def store(tmp_path):
    s = EncryptedStore(tmp_path / "memory.db", AES256GCM(AES256GCM.generate_key()))
    await s.init()
    return s


def test_new_item_gets_its_tier_baseline():
    for tier, expected in TIER_BASE_STRENGTH.items():
        assert MemoryItem(content="x", tier=tier).forgetting.strength == expected


def test_supplied_curve_is_not_overwritten():
    curve = ForgettingCurve(strength=99.0, last_access=1.0, rehearsals=7)
    item = MemoryItem(content="x", tier=Tier.WORKING, forgetting=curve)
    assert item.forgetting.strength == 99.0
    assert item.forgetting.rehearsals == 7


async def test_rehearsal_survives_a_roundtrip(store):
    item = MemoryItem(content="remember me", tier=Tier.WORKING)
    baseline = item.forgetting.strength
    item.touch()
    strengthened = item.forgetting.strength
    assert strengthened > baseline
    await store.put(item)

    loaded = await store.get(item.id)
    assert loaded.forgetting.strength == pytest.approx(strengthened)
    assert loaded.forgetting.rehearsals == 1


async def test_repeated_rehearsal_compounds_across_reads(store):
    """The failing case: each touch must build on the last, not restart from baseline."""
    item = MemoryItem(content="rehearse me", tier=Tier.WORKING)
    await store.put(item)
    strengths = []
    for _ in range(4):
        loaded = await store.get(item.id)
        loaded.touch()
        await store.put(loaded)
        strengths.append(loaded.forgetting.strength)
    assert strengths == sorted(strengths)
    assert len(set(strengths)) == len(strengths), f"strength stopped growing: {strengths}"
    assert (await store.get(item.id)).forgetting.rehearsals == 4


async def test_rehearsed_memory_retains_better_than_untouched(store):
    """The user-visible point of all this."""
    import time
    old = time.time() - 3 * 86400  # three days ago

    untouched = MemoryItem(content="never recalled", tier=Tier.EPISODIC, timestamp=old)
    untouched.forgetting.last_access = old
    await store.put(untouched)

    rehearsed = MemoryItem(content="often recalled", tier=Tier.EPISODIC, timestamp=old)
    rehearsed.forgetting.last_access = old
    for _ in range(5):
        rehearsed.forgetting.strength = rehearsed.forgetting.strength * 1.6 + 0.5
        rehearsed.forgetting.rehearsals += 1
    await store.put(rehearsed)

    a = await store.get(untouched.id)
    b = await store.get(rehearsed.id)
    assert b.forgetting.retention() > a.forgetting.retention()


async def test_promotion_raises_strength_to_the_new_tier(store):
    item = MemoryItem(content="promote me", tier=Tier.WORKING)
    await store.put(item)
    await store.update_tier(item.id, Tier.SEMANTIC)
    loaded = await store.get(item.id)
    assert loaded.tier == Tier.SEMANTIC
    assert loaded.forgetting.strength >= TIER_BASE_STRENGTH[Tier.SEMANTIC]


async def test_demotion_does_not_discard_earned_strength(store):
    item = MemoryItem(content="hard won", tier=Tier.SEMANTIC)
    for _ in range(3):
        item.touch()
    earned = item.forgetting.strength
    await store.put(item)
    await store.update_tier(item.id, Tier.WORKING)
    loaded = await store.get(item.id)
    assert loaded.tier == Tier.WORKING
    assert loaded.forgetting.strength == pytest.approx(earned)
