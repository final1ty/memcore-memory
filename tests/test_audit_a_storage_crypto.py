"""Regression tests for the storage/crypto findings of the 2026-09-24 audit.

Everything runs in tmp dirs; nothing here opens ~/.memcore or a live store. The
"old format" fixtures build databases exactly the way the code before this change
wrote them (no aad_v column, no store_meta table, ciphertexts without AAD), because
the live stores hold 120+ rows written that way and must keep reading.
"""

import asyncio
import errno
import json
import multiprocessing
import os
import sqlite3

import pytest

from memcore_memory import config
from memcore_memory.core.ebbinghaus import ForgettingCurve
from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.blind_index import BlindIndex
from memcore_memory.crypto import key_manager as km_mod
from memcore_memory.crypto.key_manager import (
    KeyManager, MasterKeyError, MasterKeyMismatch, MasterKeyMissing, MasterPasswordRequired,
    UnprotectedKeyRefused, WrongMasterPassword,
)
from memcore_memory.security.pii_filter import PIIFilter
from memcore_memory.storage.audit_log import AuditLog
from memcore_memory.storage.encrypted_sqlite import (
    BLIND_INDEX_INFO, CorruptRow, EncryptedStore,
)


@pytest.fixture(autouse=True)
def _no_deployment_env(monkeypatch):
    for var in ("MNEM_ENV", "MEMCORE_ENV", "MNEM_MASTER_PASSWORD"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def cipher():
    return AES256GCM(AES256GCM.generate_key())


@pytest.fixture
async def store(tmp_path, cipher):
    s = EncryptedStore(tmp_path / "memory.db", cipher)
    await s.init()
    return s


def _rows(db_path, sql="SELECT COUNT(*) FROM memories"):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --- old-format store, as written by the code before this change ------------

OLD_SCHEMA = '''CREATE TABLE memories (
    id TEXT PRIMARY KEY, tier TEXT, timestamp REAL, content_enc BLOB, nonce BLOB,
    metadata_enc BLOB, meta_nonce BLOB, forgetting_json TEXT, entities_json TEXT,
    embedding BLOB, blind_index_json TEXT)'''


def _write_old_store(db_path, cipher, items):
    """Rows exactly as the pre-AAD EncryptedStore.put wrote them."""
    blind = BlindIndex(cipher.derive_subkey(BLIND_INDEX_INFO))
    con = sqlite3.connect(db_path)
    con.execute(OLD_SCHEMA)
    for item in items:
        c_nonce, c_ct = cipher.encrypt(item.content.encode())
        m_nonce, m_ct = cipher.encrypt(json.dumps(item.metadata).encode())
        # The ASCII-only tokenizer the old index was built with.
        old_index = sorted(blind.compute_token_hmac(t) for t in BlindIndex._tokenize_v1(item.content))
        con.execute('INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (item.id, item.tier.value, item.timestamp, c_ct, c_nonce, m_ct, m_nonce,
                     json.dumps(item.forgetting.to_dict()), json.dumps(item.entities),
                     json.dumps(item.embedding).encode() if item.embedding else None,
                     json.dumps(old_index)))
    con.commit()
    con.close()


async def test_old_format_store_still_reads_and_is_upgraded(tmp_path, cipher):
    db = tmp_path / "memory.db"
    items = [MemoryItem(content=f"SkyNAS Docker-only telepítés {i}", tier=Tier.EPISODIC,
                        metadata={"n": i}, embedding=[0.1, 0.2]) for i in range(5)]
    _write_old_store(db, cipher, items)

    store = EncryptedStore(db, cipher)
    await store.init()
    loaded = {i.id: i for i in await store.list_all(strict=True)}
    assert {i.id for i in items} == set(loaded)
    for item in items:
        assert loaded[item.id].content == item.content
        assert loaded[item.id].metadata == item.metadata
    # The fingerprint is stamped only after the key proved itself on real rows.
    assert _rows(db, "SELECT v FROM store_meta WHERE k='key_id'") == [(store.key_id,)]
    assert _rows(db, "SELECT COUNT(*) FROM memories WHERE aad_v IS NULL") == [(5,)]

    assert await store.reencrypt_legacy_rows() == 5
    assert _rows(db, "SELECT COUNT(*) FROM memories WHERE aad_v IS NULL") == [(0,)]
    again = EncryptedStore(db, cipher)
    await again.init()
    assert {i.id: i.content for i in await again.list_all(strict=True)} == \
        {i.id: i.content for i in items}


# --- F74: ciphertexts are bound to their row --------------------------------

async def test_swapped_ciphertexts_no_longer_decrypt(store):
    a = MemoryItem(content="Transfer approved: pay Alice 10 EUR", tier=Tier.WORKING)
    b = MemoryItem(content="Never pay Mallory anything", tier=Tier.SEMANTIC)
    await store.put(a)
    await store.put(b)
    con = sqlite3.connect(store.db_path)
    ra = con.execute("SELECT content_enc, nonce FROM memories WHERE id=?", (a.id,)).fetchone()
    rb = con.execute("SELECT content_enc, nonce FROM memories WHERE id=?", (b.id,)).fetchone()
    con.execute("UPDATE memories SET content_enc=?, nonce=? WHERE id=?", (*rb, a.id))
    con.execute("UPDATE memories SET content_enc=?, nonce=? WHERE id=?", (*ra, b.id))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow) as err:
        await store.get(a.id)
    assert err.value.column == "content_enc" and a.id in str(err.value)
    assert "Mallory" not in str(err.value)


async def test_content_cannot_be_moved_into_the_metadata_slot(store):
    a = MemoryItem(content="{}", tier=Tier.WORKING)
    await store.put(a)
    con = sqlite3.connect(store.db_path)
    ct, nonce = con.execute("SELECT content_enc, nonce FROM memories").fetchone()
    con.execute("UPDATE memories SET metadata_enc=?, meta_nonce=?", (ct, nonce))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow) as err:
        await store.get(a.id)
    assert err.value.column == "metadata_enc"


# --- R2: one bad row no longer takes the whole store down --------------------

def _corrupt(db_path, memory_id, sql):
    con = sqlite3.connect(db_path)
    if sql.startswith("flip "):
        # One flipped bit in a ciphertext, the way bit rot or a bad hand edit looks.
        column = sql.split()[1]
        blob = bytearray(con.execute(f"SELECT {column} FROM memories WHERE id=?",
                                     (memory_id,)).fetchone()[0])
        blob[3] ^= 0x01
        con.execute(f"UPDATE memories SET {column}=? WHERE id=?", (bytes(blob), memory_id))
    else:
        con.execute(sql, (memory_id,))
    con.commit()
    con.close()


@pytest.mark.parametrize("fault", [
    ("content", "flip content_enc"),
    ("metadata", "flip metadata_enc"),
    ("forgetting", "UPDATE memories SET forgetting_json = '{\"strength\": ' WHERE id=?"),
    ("forgetting-type", "UPDATE memories SET forgetting_json = '{\"strength\": \"abc\"}' WHERE id=?"),
    ("entities", "UPDATE memories SET entities_json = '[\"a\", ' WHERE id=?"),
    ("tier", "UPDATE memories SET tier = 'archival' WHERE id=?"),
])
async def test_one_bad_row_is_skipped_and_named(store, fault):
    name, sql = fault
    items = [MemoryItem(content=f"memory {i}", tier=Tier.EPISODIC) for i in range(20)]
    for item in items:
        await store.put(item)
    bad = items[7]
    _corrupt(store.db_path, bad.id, sql)

    listed = await store.list_all()
    assert len(listed) == 19 and bad.id not in {i.id for i in listed}
    assert len(await store.list_by_tier(Tier.EPISODIC)) == 19
    readable, unreadable = await store.scan()
    assert len(readable) == 19 and list(unreadable) == [bad.id]
    assert (await store.verify())["unreadable"].keys() == {bad.id}
    with pytest.raises(CorruptRow):
        await store.list_all(strict=True)
    with pytest.raises(CorruptRow) as err:
        await store.get(bad.id)
    assert bad.id in str(err.value)
    # A bad row is reported, never silently removed.
    assert _rows(store.db_path) == [(20,)]
    assert bad.id not in await store.get_many([i.id for i in items])
    assert await store.rebuild_blind_index() == 19


# --- F19: targeted updates cannot resurrect or revert ------------------------

async def test_delete_reports_whether_a_row_went(store):
    item = MemoryItem(content="x", tier=Tier.EPISODIC)
    await store.put(item)
    assert await store.delete(item.id) is True
    assert await store.delete(item.id) is False


async def test_rehearsal_after_a_delete_does_not_resurrect(store):
    item = MemoryItem(content="gone soon", tier=Tier.EPISODIC)
    await store.put(item)
    snapshot = await store.get(item.id)
    await store.delete(item.id)
    snapshot.touch()
    assert await store.update_forgetting(item.id, snapshot.forgetting) is False
    assert await store.get(item.id) is None


async def test_concurrent_get_touch_and_delete_or_promote(store):
    async def rehearse(mid):
        loaded = await store.get(mid)
        if loaded is None:
            return
        await asyncio.sleep(0)
        loaded.touch()
        await store.update_forgetting(mid, loaded.forgetting)

    for _ in range(20):
        a = MemoryItem(content="a", tier=Tier.EPISODIC)
        b = MemoryItem(content="b", tier=Tier.EPISODIC)
        await store.put(a)
        await store.put(b)
        await asyncio.gather(rehearse(a.id), store.delete(a.id))
        await asyncio.gather(rehearse(b.id), store.update_tier(b.id, Tier.SEMANTIC))
        assert await store.get(a.id) is None
        assert (await store.get(b.id)).tier == Tier.SEMANTIC


async def test_update_forgetting_persists_only_the_curve(store):
    item = MemoryItem(content="x", tier=Tier.EPISODIC)
    await store.put(item)
    await store.update_tier(item.id, Tier.SEMANTIC)
    curve = ForgettingCurve(strength=999.0, last_access=1.0, rehearsals=4, importance=0.9)
    assert await store.update_forgetting(item.id, curve) is True
    loaded = await store.get(item.id)
    assert loaded.tier == Tier.SEMANTIC
    assert loaded.forgetting.strength == 999.0 and loaded.forgetting.rehearsals == 4


async def test_update_content_keeps_tier_and_never_inserts(store):
    item = MemoryItem(content="old", tier=Tier.EPISODIC)
    await store.put(item)
    await store.update_tier(item.id, Tier.SEMANTIC)
    item.content = "new"
    assert await store.update_content(item) is True
    loaded = await store.get(item.id)
    assert (loaded.content, loaded.tier) == ("new", Tier.SEMANTIC)
    ghost = MemoryItem(content="ghost", tier=Tier.WORKING)
    assert await store.update_content(ghost) is False
    assert await store.get(ghost.id) is None


async def test_update_tier_on_a_missing_row(store):
    assert await store.update_tier("nope", Tier.SEMANTIC) is False


async def test_get_many(store):
    items = [MemoryItem(content=str(i), tier=Tier.WORKING) for i in range(3)]
    for item in items:
        await store.put(item)
    got = await store.get_many([items[0].id, "missing", items[2].id, items[0].id])
    assert set(got) == {items[0].id, items[2].id}
    assert got[items[2].id].content == "2"
    assert await store.get_many([]) == {}


# --- F75: WAL journal and a busy timeout ------------------------------------

async def test_store_runs_in_wal_mode_with_a_busy_timeout(store):
    assert _rows(store.db_path, "PRAGMA journal_mode") == [("wal",)]
    async with store._connect() as db:
        async with db.execute("PRAGMA busy_timeout") as cur:
            assert (await cur.fetchone())[0] == 10000


# --- R3: a missing or foreign key next to a populated store -----------------

async def _populated_store(settings_like_dir, cipher):
    s = EncryptedStore(settings_like_dir / "memory.db", cipher)
    await s.init()
    await s.put(MemoryItem(content="precious", tier=Tier.EPISODIC))
    return s


async def test_no_new_key_next_to_a_populated_store(tmp_path, cipher):
    await _populated_store(tmp_path, cipher)
    key = tmp_path / "master.key"
    with pytest.raises(MasterKeyMissing, match="holds 1 encrypted"):
        KeyManager(key).load_or_create(db_path=tmp_path / "memory.db")
    assert not key.exists(), "a refused start must not leave a key behind"
    assert not list(tmp_path.glob(".master.key.*"))
    # Deliberately starting over is still possible.
    assert len(KeyManager(key).load_or_create(db_path=tmp_path / "memory.db",
                                              allow_new_key=True)) == 32


async def test_configured_key_is_checked_against_the_configured_store(isolate_data_dir):
    settings = config.settings
    key = KeyManager(settings.key_path).load_or_create()
    s = EncryptedStore(settings.db_path, AES256GCM(key))
    await s.init()
    await s.put(MemoryItem(content="precious", tier=Tier.EPISODIC))
    settings.key_path.unlink()
    with pytest.raises(MasterKeyMissing):
        KeyManager(settings.key_path).load_or_create()
    assert not settings.key_path.exists()


async def test_kg_rows_also_block_a_new_key(tmp_path):
    con = sqlite3.connect(tmp_path / "memory.kg.db")
    con.execute("CREATE TABLE kg_nodes (id TEXT)")
    con.execute("INSERT INTO kg_nodes VALUES ('n')")
    con.commit()
    con.close()
    with pytest.raises(MasterKeyMissing):
        KeyManager(tmp_path / "master.key").load_or_create(db_path=tmp_path / "memory.db")


def test_empty_dir_still_creates_a_key(tmp_path):
    assert len(KeyManager(tmp_path / "master.key").load_or_create(db_path=tmp_path / "memory.db")) == 32


async def test_empty_schema_only_store_still_gets_a_key(tmp_path, cipher):
    s = EncryptedStore(tmp_path / "memory.db", cipher)
    await s.init()
    assert len(KeyManager(tmp_path / "master.key").load_or_create(db_path=tmp_path / "memory.db")) == 32


async def test_foreign_key_is_refused_before_any_write(tmp_path, cipher):
    await _populated_store(tmp_path, cipher)
    wrong = EncryptedStore(tmp_path / "memory.db", AES256GCM(os.urandom(32)))
    with pytest.raises(MasterKeyMismatch):
        await wrong.init()
    assert _rows(tmp_path / "memory.db") == [(1,)]


async def test_foreign_key_is_refused_on_an_unstamped_legacy_store(tmp_path, cipher):
    _write_old_store(tmp_path / "memory.db", cipher, [MemoryItem(content="old", tier=Tier.EPISODIC)])
    wrong = EncryptedStore(tmp_path / "memory.db", AES256GCM(os.urandom(32)))
    with pytest.raises(MasterKeyMismatch):
        await wrong.init()
    # Nothing was stamped with the wrong key's fingerprint.
    assert _rows(tmp_path / "memory.db", "SELECT COUNT(*) FROM store_meta WHERE k='key_id'") == [(0,)]
    right = EncryptedStore(tmp_path / "memory.db", cipher)
    await right.init()
    assert [i.content for i in await right.list_all(strict=True)] == ["old"]


# --- F11: concurrent creation leaves every process on the key on disk -------

def _race_worker(path, password, barrier, out):
    barrier.wait()
    key = KeyManager(path).load_or_create(password=password)
    out.put(key.hex())


@pytest.mark.parametrize("password", [None, "s3cret"])
def test_concurrent_creators_all_end_up_with_the_key_on_disk(tmp_path, password):
    ctx = multiprocessing.get_context("fork")
    for run in range(5):
        path = tmp_path / f"run{run}" / "master.key"
        path.parent.mkdir()
        barrier, out = ctx.Barrier(3), ctx.Queue()
        procs = [ctx.Process(target=_race_worker, args=(path, password, barrier, out))
                 for _ in range(3)]
        for p in procs:
            p.start()
        held = {out.get(timeout=60) for _ in procs}
        for p in procs:
            p.join(timeout=60)
        on_disk = KeyManager(path).load_or_create(password=password).hex()
        assert held == {on_disk}, f"run {run}: a process kept a key that is not on disk"
        assert path.stat().st_mode & 0o777 == 0o600
        assert not list(path.parent.glob(".master.key.*")), "temp files left behind"


def test_losing_creator_adopts_the_winners_key(tmp_path):
    path = tmp_path / "master.key"
    winner = KeyManager(path).load_or_create()
    # A creator that passed the exists() check just before the winner wrote.
    assert KeyManager(path)._create(None) == winner


def test_creation_without_hard_links_still_has_one_winner(tmp_path, monkeypatch):
    def no_link(*a, **k):
        raise OSError(errno.EPERM, "no hard links here")

    monkeypatch.setattr(km_mod.os, "link", no_link)
    path = tmp_path / "master.key"
    first = KeyManager(path).load_or_create()
    assert KeyManager(path)._create(None) == first
    assert path.stat().st_mode & 0o777 == 0o600


# --- F20 / R13: env password and deployment marker ---------------------------

def test_env_password_protects_a_new_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEM_MASTER_PASSWORD", "correct horse")
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create()
    assert json.loads(path.read_text()).keys() >= {"salt", "nonce", "ct"}
    assert KeyManager(path).load_or_create() == key


@pytest.mark.parametrize("var", ["MNEM_ENV", "MEMCORE_ENV"])
def test_prod_marker_refuses_unprotected_creation(tmp_path, monkeypatch, var):
    monkeypatch.setenv(var, "prod")
    with pytest.raises(UnprotectedKeyRefused, match=var):
        KeyManager(tmp_path / "master.key").load_or_create()
    assert not (tmp_path / "master.key").exists()


def test_prod_with_env_password_creates(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEM_ENV", "prod")
    monkeypatch.setenv("MNEM_MASTER_PASSWORD", "pw")
    assert len(KeyManager(tmp_path / "master.key").load_or_create()) == 32


def test_refusal_is_catchable_as_password_required_and_value_error(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEM_ENV", "production")
    with pytest.raises(MasterPasswordRequired):
        KeyManager(tmp_path / "a.key").load_or_create()
    with pytest.raises(ValueError):
        KeyManager(tmp_path / "b.key").load_or_create()


# --- F21: a wrong password is named as such ----------------------------------

def test_wrong_password_is_diagnosed_and_the_file_untouched(tmp_path):
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create(password="A")
    before = path.read_bytes()
    with pytest.raises(WrongMasterPassword, match=str(path)) as err:
        KeyManager(path).load_or_create(password="B")
    assert isinstance(err.value, MasterPasswordRequired)
    assert path.read_bytes() == before
    assert KeyManager(path).load_or_create(password="A") == key


def test_literal_unexpanded_placeholder_is_a_wrong_password(tmp_path, monkeypatch):
    path = tmp_path / "master.key"
    KeyManager(path).load_or_create(password="real")
    monkeypatch.setenv("MNEM_MASTER_PASSWORD", "${MNEM_MASTER_PASSWORD}")
    with pytest.raises(WrongMasterPassword):
        KeyManager(path).load_or_create()


# --- F62: a password offered to a raw key is not silently ignored -----------

def test_password_on_a_raw_key_warns_and_rewrites_nothing(tmp_path, capsys):
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create()
    before = path.read_bytes()
    assert KeyManager(path).load_or_create(password="i-think-this-protects-it") == key
    assert path.read_bytes() == before
    captured = capsys.readouterr()
    assert "UNPROTECTED" in captured.err and captured.out == ""


def test_protect_wraps_the_same_key(tmp_path):
    path = tmp_path / "master.key"
    key = KeyManager(path).load_or_create()
    km = KeyManager(path)
    assert km.protect("hunter2") == key
    assert km.is_protected()
    assert path.stat().st_mode & 0o777 == 0o600
    assert KeyManager(path).load_or_create(password="hunter2") == key
    assert not list(tmp_path.glob(".master.key.*"))
    with pytest.raises(MasterKeyError, match="already"):
        KeyManager(path).protect("again")


def test_protect_refuses_an_empty_password_and_a_non_key(tmp_path):
    path = tmp_path / "master.key"
    KeyManager(path).load_or_create()
    with pytest.raises(ValueError):
        KeyManager(path).protect("")
    junk = tmp_path / "junk.key"
    junk.write_bytes(b"x" * 40)
    with pytest.raises(MasterKeyError):
        KeyManager(junk).protect("pw")
    assert junk.read_bytes() == b"x" * 40


# --- F34 / F71: audit log survives restarts and detects truncation -----------

def test_audit_chain_survives_a_reopen(tmp_path):
    path = tmp_path / "audit.log"
    first = AuditLog(path, b"k" * 32)
    first.log("add", "m1")
    reopened = AuditLog(path, b"k" * 32)
    assert reopened._last_hash == first._last_hash
    reopened.log("get", "m1")
    AuditLog(path, b"k" * 32).log("delete", "m1")
    assert AuditLog(path, b"k" * 32).verify_chain() is True


def test_audit_corrupt_tail_raises_instead_of_restarting(tmp_path):
    path = tmp_path / "audit.log"
    AuditLog(path, b"k" * 32).log("add", "m1")
    with open(path, "a") as f:
        f.write("{not json\n")
    with pytest.raises(ValueError):
        AuditLog(path, b"k" * 32)


def test_audit_truncation_and_deletion_are_detected(tmp_path):
    path = tmp_path / "audit.log"
    log = AuditLog(path, b"k" * 32)
    for action in ("add", "get", "delete"):
        log.log(action, "m1")
    assert log.verify_chain() is True
    lines = path.read_text().splitlines(keepends=True)
    path.write_text(lines[0])
    assert log.verify_chain() is False
    path.unlink()
    assert log.verify_chain() is False
    log.head_path.unlink()
    assert log.verify_chain() is True


def test_audit_log_without_a_head_still_verifies_and_gains_one(tmp_path):
    # A log from before heads existed (no head-aware marker). A log written with a
    # head must not verify once the head is gone - see test_audit_A_round2.py.
    path = tmp_path / "audit.log"
    log = AuditLog(path, b"k" * 32)
    data = {"ts": 1.0, "actor": "system", "action": "add", "memory_id": "m1", "metadata": {}}
    chain = log._compute_chain_hash("0" * 64, data)
    path.write_text(json.dumps({**data, "prev_hash": "0" * 64, "chain_hash": chain}) + "\n")
    assert log.verify_chain() is True
    log.log("get", "m1")
    assert json.loads(log.head_path.read_text())["count"] == 2
    assert log.verify_chain() is True


# --- F72: PII filter --------------------------------------------------------

def test_redact_covers_every_match_not_just_five():
    text = " ".join(f"u{i}@example.com" for i in range(7))
    allowed, out, findings = PIIFilter(action="redact").check_and_act(text)
    assert "@" not in out and out.count("[EMAIL_REDACTED]") == 7
    assert len(findings["email"]) == 5


def test_redact_does_not_leak_part_of_a_longer_match():
    out, _ = PIIFilter().redact("mail a@b.co and aa@b.co")
    assert out == "mail [EMAIL_REDACTED] and [EMAIL_REDACTED]"


@pytest.mark.parametrize("text", [
    "card 4111 1111 1111 1111",
    "card 4111-1111-1111-1111",
    "sk-ant-api03-" + "a" * 40,
    "sk-proj-" + "b" * 40,
    "API_KEY=abcdefghijklmnop1234",
    "AKIAIOSFODNN7EXAMPLE",
    "ghp_" + "c" * 36,
])
def test_block_mode_catches_modern_secrets_and_spaced_cards(text):
    allowed, _, findings = PIIFilter(action="block").check_and_act(text)
    assert allowed is False, findings


def test_non_luhn_digit_runs_are_not_cards():
    assert "credit_card" not in PIIFilter().scan("order 1234 5678 9012 3456")


def test_international_phone_is_detected():
    assert "phone" in PIIFilter().scan("hívj: +36 30 123 4567")


# --- F73: blind index handles accented words ---------------------------------

def test_blind_tokenizer_keeps_accented_words():
    bi = BlindIndex(AES256GCM.generate_key())
    assert bi._tokenize("tűzfal beállítás kész") == {"tűzfal", "beállítás", "kész"}
    assert bi._tokenize("Kovács-Dobos Ádám") == {"kovács", "dobos", "ádám"}
    assert {"user", "name"} <= bi._tokenize("user_name")
    # NFC and NFD spellings of the same word index alike.
    assert bi._tokenize("árvíz") == bi._tokenize("árvíz")


async def test_blind_search_finds_hungarian_words(store):
    item = MemoryItem(content="tűzfal beállítás kész", tier=Tier.EPISODIC)
    await store.put(item)
    assert await store.search_by_blind_index("tűzfal") == [item.id]
    assert await store.search_by_blind_index("KÉSZ") == [item.id]


async def test_old_blind_index_rows_match_until_rebuilt(tmp_path, cipher):
    old = MemoryItem(content="SkyNAS Docker-only telepítés sikeres", tier=Tier.EPISODIC)
    _write_old_store(tmp_path / "memory.db", cipher, [old])
    store = EncryptedStore(tmp_path / "memory.db", cipher)
    await store.init()
    # Old index holds "telep"; the query tokenizes to "telepítés" - legacy tokens bridge it.
    assert await store.search_by_blind_index("telepítés") == [old.id]
    assert await store.search_by_blind_index("docker") == [old.id]
    assert await store.rebuild_blind_index() == 1
    assert await store.search_by_blind_index("telepítés") == [old.id]
    assert _rows(tmp_path / "memory.db", "SELECT v FROM store_meta WHERE k='blind_index_v'") == [("2",)]
