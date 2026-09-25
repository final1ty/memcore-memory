"""Regression tests for four bugs that made features report success while doing nothing.

The blind index one is the sharpest: the tokenizer's pattern was written as
'\\b[a-z0-9]{3,}\\b' without the r prefix, so Python turned each \\b into a backspace
(0x08) and the regex hunted for a literal control character. It matched nothing on
any input, so every blind index ever computed was an empty list and encrypted search
returned nothing - with no error anywhere. `cat` renders 0x08 invisibly, so the line
looked correct in every review.
"""

import json
import os
import sqlite3

import pytest

from memcore_memory.core.ebbinghaus import EXPONENTIAL, POWER_LAW, ForgettingCurve
from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.blind_index import BlindIndex
from memcore_memory.crypto.key_manager import ENV_VAR, KeyManager
from memcore_memory.storage.encrypted_sqlite import EncryptedStore


@pytest.fixture
async def store(tmp_path):
    s = EncryptedStore(tmp_path / "t.db", AES256GCM(AES256GCM.generate_key()))
    await s.init()
    return s


# --- blind index ----------------------------------------------------------

def test_tokenizer_actually_produces_tokens():
    """The bug: this returned an empty set for every possible input."""
    bi = BlindIndex(AES256GCM.generate_key())
    assert bi._tokenize("User prefers dark mode") == {"user", "prefers", "dark", "mode"}


def test_tokenizer_splits_on_punctuation_and_underscores():
    bi = BlindIndex(AES256GCM.generate_key())
    assert {"user", "name", "dark", "mode"} <= bi._tokenize("user_name: dark-mode!")


def test_index_and_query_overlap_on_a_shared_term():
    bi = BlindIndex(AES256GCM.generate_key())
    idx = bi.compute_index("User prefers dark mode and likes python", {"importance": 0.9})
    assert set(idx) & set(bi.search_query_hmacs("dark mode"))


def test_blind_index_key_is_not_the_content_key():
    """Reusing the encryption key for search HMACs makes one compromise into two."""
    key = AES256GCM.generate_key()
    cipher = AES256GCM(key)
    assert cipher.derive_subkey(b"blind-index-v1") != key


async def test_search_by_blind_index_finds_the_memory(store):
    item = MemoryItem(content="User prefers dark mode in python", tier=Tier.EPISODIC)
    await store.put(item)
    assert item.id in await store.search_by_blind_index("dark mode")


async def test_blind_index_search_does_not_decrypt_anything(store):
    """The whole point: keyword search over rows the searcher cannot read."""
    item = MemoryItem(content="the passphrase is hunter2", tier=Tier.EPISODIC)
    await store.put(item)

    blind = store.blind
    store.cipher = None  # any decryption attempt now raises
    assert item.id in await store.search_by_blind_index("passphrase")

    # And the stored index must not leak the words themselves.
    raw = sqlite3.connect(store.db_path).execute(
        "SELECT blind_index_json FROM memories WHERE id=?", (item.id,)).fetchone()[0]
    assert "hunter2" not in raw and "passphrase" not in raw
    assert blind.compute_token_hmac("hunter2") in json.loads(raw)


async def test_rows_written_before_the_column_existed_are_migrated(tmp_path):
    """An older store must open, not crash, and be repairable."""
    db = sqlite3.connect(tmp_path / "old.db")
    db.execute("""CREATE TABLE memories (
        id TEXT PRIMARY KEY, tier TEXT, timestamp REAL, content_enc BLOB, nonce BLOB,
        metadata_enc BLOB, meta_nonce BLOB, forgetting_json TEXT, entities_json TEXT,
        embedding BLOB)""")
    db.commit()
    db.close()

    store = EncryptedStore(tmp_path / "old.db", AES256GCM(AES256GCM.generate_key()))
    await store.init()
    item = MemoryItem(content="written after the migration", tier=Tier.EPISODIC)
    await store.put(item)
    assert item.id in await store.search_by_blind_index("migration")


async def test_rebuild_blind_index_repairs_null_rows(store):
    item = MemoryItem(content="indexed later", tier=Tier.EPISODIC)
    await store.put(item)
    db = sqlite3.connect(store.db_path)
    db.execute("UPDATE memories SET blind_index_json=NULL")
    db.commit()
    db.close()

    assert await store.search_by_blind_index("indexed") == []
    assert await store.rebuild_blind_index() == 1
    assert item.id in await store.search_by_blind_index("indexed")


# --- forgetting curve -----------------------------------------------------

def test_power_law_keeps_a_tail_where_exponential_collapses():
    now = 1_000_000.0
    args = dict(strength=1.0, last_access=now, importance=0.5)
    exp = ForgettingCurve(decay_model=EXPONENTIAL, **args)
    power = ForgettingCurve(decay_model=POWER_LAW, power_d=0.3, **args)
    far = now + 365 * 86400
    assert exp.retention(far) < 1e-6
    assert power.retention(far) > 0.1


def test_both_models_decrease_monotonically():
    now = 1_000_000.0
    for model in (EXPONENTIAL, POWER_LAW):
        fc = ForgettingCurve(strength=1.0, last_access=now, decay_model=model)
        seq = [fc.retention(now + d * 86400) for d in range(0, 40, 5)]
        assert seq == sorted(seq, reverse=True), model


def test_feedback_scales_how_much_a_rehearsal_counts():
    weak, strong = ForgettingCurve(strength=1.0), ForgettingCurve(strength=1.0)
    weak.rehearse(feedback=0.2)
    strong.rehearse(feedback=1.0)
    assert 1.0 < weak.strength < strong.strength


def test_default_feedback_reproduces_the_original_rule():
    fc = ForgettingCurve(strength=1.0)
    fc.rehearse()
    assert fc.strength == pytest.approx(1.0 * 1.6 + 0.5)


async def test_decay_model_survives_a_storage_round_trip(store):
    """Leaving it out of to_dict() would silently turn a power-law memory exponential."""
    item = MemoryItem(content="slow forgetter", tier=Tier.SEMANTIC)
    item.forgetting.decay_model = POWER_LAW
    item.forgetting.power_d = 0.42
    await store.put(item)

    loaded = await store.get(item.id)
    assert loaded.forgetting.decay_model == POWER_LAW
    assert loaded.forgetting.power_d == pytest.approx(0.42)


async def test_rows_without_the_new_fields_load_with_defaults(store):
    item = MemoryItem(content="written by an older version", tier=Tier.EPISODIC)
    await store.put(item)
    db = sqlite3.connect(store.db_path)
    db.execute("UPDATE memories SET forgetting_json=?",
               (json.dumps({"strength": 3.0, "last_access": 1.0, "rehearsals": 2,
                            "importance": 0.7}),))
    db.commit()
    db.close()

    loaded = await store.get(item.id)
    assert loaded.forgetting.strength == 3.0
    assert loaded.forgetting.rehearsals == 2
    assert loaded.forgetting.decay_model == EXPONENTIAL


# --- production key policy ------------------------------------------------

def test_prod_refuses_to_create_an_unprotected_key(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "prod")
    monkeypatch.delenv("MNEM_MASTER_PASSWORD", raising=False)
    with pytest.raises(ValueError, match="master password required"):
        KeyManager(tmp_path / "master.key").load_or_create(password=None)
    assert not (tmp_path / "master.key").exists()


def test_prod_still_allows_a_password_protected_key(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "prod")
    key = KeyManager(tmp_path / "master.key").load_or_create(password="hunter2")
    assert len(key) == 32


def test_prod_does_not_lock_an_existing_unprotected_store_out(tmp_path, monkeypatch):
    """Gating loads too would strand a running deployment from its own data."""
    monkeypatch.delenv(ENV_VAR, raising=False)
    key = KeyManager(tmp_path / "master.key").load_or_create()

    monkeypatch.setenv(ENV_VAR, "prod")
    assert KeyManager(tmp_path / "master.key").load_or_create() == key


def test_non_prod_creates_unprotected_keys_as_before(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_VAR, "dev")
    assert len(KeyManager(tmp_path / "master.key").load_or_create()) == 32
