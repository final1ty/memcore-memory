"""Round-2 regression tests for the storage/crypto findings of the 2026-09-24 audit.

Everything runs in tmp dirs. Old-format databases are built by hand, exactly as
earlier code wrote them, because the live stores hold rows in those formats.
"""

import json
import os
import sqlite3
import stat
import time

import pytest

from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.crypto.key_manager import KeyManager
from memcore_memory.graph.kg import KnowledgeGraph
from memcore_memory.security.pii_filter import PIIFilter
from memcore_memory.storage.audit_log import AuditLog, AuditLogCorrupt, TAMPER_ACTION
from memcore_memory.storage.encrypted_sqlite import CorruptRow, EncryptedStore


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


def _q(db_path, sql, params=()):
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def _file_bytes(db_path):
    data = b""
    for suffix in ("", "-wal"):
        p = db_path.parent / (db_path.name + suffix)
        if p.exists():
            data += p.read_bytes()
    return data


# --- F4: entity names are no longer stored in clear -------------------------

# The schema the live stores have: pre-AAD HEAD code plus the blind-index column.
LEGACY_SCHEMA = '''CREATE TABLE memories (
    id TEXT PRIMARY KEY, tier TEXT, timestamp REAL, content_enc BLOB, nonce BLOB,
    metadata_enc BLOB, meta_nonce BLOB, forgetting_json TEXT, entities_json TEXT,
    embedding BLOB, blind_index_json TEXT)'''

NAMES = ["Kovacs-Dobos", "WireGuardHost", "JellyfinBox"]


def _write_legacy_store(db_path, cipher, items, entities_json=None):
    con = sqlite3.connect(db_path)
    con.execute(LEGACY_SCHEMA)
    for item in items:
        c_nonce, c_ct = cipher.encrypt(item.content.encode())
        m_nonce, m_ct = cipher.encrypt(json.dumps(item.metadata).encode())
        ent = entities_json.get(item.id) if entities_json else None
        con.execute('INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (item.id, item.tier.value, item.timestamp, c_ct, c_nonce, m_ct, m_nonce,
                     json.dumps(item.forgetting.to_dict()),
                     ent if ent is not None else json.dumps(item.entities), None, None))
    con.commit()
    con.close()


async def test_legacy_plaintext_entities_are_read_and_migrated(tmp_path, cipher):
    db = tmp_path / "memory.db"
    items = [MemoryItem(content=f"note {i}", tier=Tier.EPISODIC, entities=NAMES[: i + 1])
             for i in range(3)]
    broken = MemoryItem(content="broken entities", tier=Tier.EPISODIC)
    _write_legacy_store(db, cipher, items + [broken], {broken.id: '["half'})
    assert NAMES[0].encode() in _file_bytes(db)

    s = EncryptedStore(db, cipher)
    await s.init()
    got = {i.id: i for i in await s.list_all()}
    for item in items:
        assert got[item.id].entities == item.entities
        assert got[item.id].content == item.content
    # Plaintext gone from the column and from the file, freed pages and -wal included.
    assert _q(db, "SELECT COUNT(*) FROM memories WHERE entities_json IS NOT NULL") == [(1,)]
    assert _q(db, "SELECT entities_json FROM memories WHERE id=?", (broken.id,)) == [('["half',)]
    raw = _file_bytes(db)
    for name in NAMES:
        assert name.encode() not in raw
    # The unparseable list is reported, never destroyed.
    with pytest.raises(CorruptRow, match="entities_json"):
        await s.get(broken.id)
    # A second open finds nothing left to do and reads the same.
    again = EncryptedStore(db, cipher)
    await again.init()
    assert (await again.get(items[2].id)).entities == NAMES


async def test_two_processes_opening_a_legacy_store_together(tmp_path, cipher):
    import asyncio
    db = tmp_path / "memory.db"
    items = [MemoryItem(content=f"n{i}", tier=Tier.EPISODIC, entities=[f"Ent{i}"]) for i in range(20)]
    _write_legacy_store(db, cipher, items)
    a, b = EncryptedStore(db, cipher), EncryptedStore(db, cipher)
    await asyncio.gather(a.init(), b.init())
    assert _q(db, "SELECT COUNT(*) FROM memories WHERE entities_json IS NOT NULL") == [(0,)]
    assert {i.id: i.entities for i in await a.list_all(strict=True)} == \
        {i.id: i.entities for i in items}


async def test_new_writes_never_store_entities_in_clear(store):
    item = MemoryItem(content="x", tier=Tier.EPISODIC, entities=["SecretHostName"])
    await store.put(item)
    item.entities = ["OtherSecretName"]
    item.content = "y"
    assert await store.update_content(item) is True
    assert (await store.get(item.id)).entities == ["OtherSecretName"]
    assert _q(store.db_path, "SELECT entities_json FROM memories") == [(None,)]
    raw = _file_bytes(store.db_path)
    assert b"SecretHostName" not in raw and b"OtherSecretName" not in raw


async def test_entities_cannot_be_moved_between_rows(store):
    a = MemoryItem(content="a", tier=Tier.EPISODIC, entities=["Alice"])
    b = MemoryItem(content="b", tier=Tier.EPISODIC, entities=["Bob"])
    await store.put(a)
    await store.put(b)
    con = sqlite3.connect(store.db_path)
    enc = con.execute("SELECT entities_enc, entities_nonce FROM memories WHERE id=?", (a.id,)).fetchone()
    con.execute("UPDATE memories SET entities_enc=?, entities_nonce=? WHERE id=?", (*enc, b.id))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow, match="entities_enc"):
        await store.get(b.id)


async def test_plaintext_written_by_older_code_after_migration_wins(store):
    # A process still running the previous code rewrites entities_json only; its
    # value is newer than the encrypted one it didn't know about.
    item = MemoryItem(content="x", tier=Tier.EPISODIC, entities=["Old"])
    await store.put(item)
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE memories SET entities_json=? WHERE id=?", (json.dumps(["New"]), item.id))
    con.commit()
    con.close()
    assert (await store.get(item.id)).entities == ["New"]


# --- F69: a tier listing is newest first -------------------------------------

async def test_list_by_tier_is_newest_first_even_after_rewrites(store):
    items = [MemoryItem(content=f"m{i}", tier=Tier.EPISODIC, timestamp=1000.0 + i) for i in range(5)]
    for item in items:
        await store.put(item)
    await store.put(items[0])  # INSERT OR REPLACE moves it to the end of rowid order
    listed = [i.id for i in await store.list_by_tier(Tier.EPISODIC)]
    assert listed == [i.id for i in reversed(items)]


# --- R25: update_tier parses the curve like every read does ------------------

async def test_update_tier_with_null_or_string_typed_curve(store):
    a = MemoryItem(content="a", tier=Tier.WORKING)
    b = MemoryItem(content="b", tier=Tier.WORKING)
    await store.put(a)
    await store.put(b)
    con = sqlite3.connect(store.db_path)
    con.execute("UPDATE memories SET forgetting_json=NULL WHERE id=?", (a.id,))
    con.execute("""UPDATE memories SET forgetting_json='{"strength": "0.5"}' WHERE id=?""", (b.id,))
    con.commit()
    con.close()
    assert await store.update_tier(a.id, Tier.EPISODIC) is True
    assert await store.update_tier(b.id, Tier.EPISODIC) is True
    for mid in (a.id, b.id):
        stored = json.loads(_q(store.db_path, "SELECT forgetting_json FROM memories WHERE id=?", (mid,))[0][0])
        assert isinstance(stored["strength"], float) and stored["strength"] >= 7.0
        assert set(stored) >= {"strength", "last_access", "rehearsals", "importance", "decay_model"}
        assert (await store.get(mid)).tier == Tier.EPISODIC


async def test_update_tier_on_an_unparseable_curve_names_the_row_and_changes_nothing(store):
    item = MemoryItem(content="a", tier=Tier.WORKING)
    await store.put(item)
    con = sqlite3.connect(store.db_path)
    con.execute("""UPDATE memories SET forgetting_json='{"strength": "abc"}' WHERE id=?""", (item.id,))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow, match=item.id):
        await store.update_tier(item.id, Tier.EPISODIC)
    assert _q(store.db_path, "SELECT tier, forgetting_json FROM memories") == \
        [("working", '{"strength": "abc"}')]


# --- R28: a new database file is private -------------------------------------

async def test_new_database_file_is_created_0600(tmp_path, cipher):
    old = os.umask(0o022)
    try:
        s = EncryptedStore(tmp_path / "memory.db", cipher)
        await s.init()
        await s.put(MemoryItem(content="x", tier=Tier.EPISODIC))
    finally:
        os.umask(old)
    assert stat.S_IMODE((tmp_path / "memory.db").stat().st_mode) == 0o600
    for side in ("memory.db-wal", "memory.db-shm"):
        p = tmp_path / side
        if p.exists():
            assert stat.S_IMODE(p.stat().st_mode) == 0o600


async def test_an_existing_database_keeps_its_mode(tmp_path, cipher):
    db = tmp_path / "memory.db"
    s = EncryptedStore(db, cipher)
    await s.init()
    db.chmod(0o640)
    await EncryptedStore(db, cipher).init()
    assert stat.S_IMODE(db.stat().st_mode) == 0o640


# --- R3: an initialised, empty knowledge graph is not data -------------------

async def test_empty_initialised_graph_does_not_block_the_first_key(tmp_path, cipher):
    kg = KnowledgeGraph(tmp_path / "memory.kg.db", cipher)
    await kg.init()
    assert _q(tmp_path / "memory.kg.db", "SELECT COUNT(*) FROM kg_meta")[0][0] >= 1
    key = KeyManager(tmp_path / "master.key").load_or_create(db_path=tmp_path / "memory.db")
    assert len(key) == 32


# --- F11: stale key temp files are removed -----------------------------------

def test_stale_key_temp_files_are_removed_and_young_ones_kept(tmp_path):
    key_path = tmp_path / "master.key"
    KeyManager(key_path).load_or_create(db_path=tmp_path / "memory.db")
    stale = tmp_path / ".master.key.abc123.tmp"
    young = tmp_path / ".master.key.def456.tmp"
    other = tmp_path / ".other.key.abc.tmp"
    for p in (stale, young, other):
        p.write_bytes(os.urandom(32))
    past = time.time() - 3600
    os.utime(stale, (past, past))
    os.utime(other, (past, past))
    before = key_path.read_bytes()
    assert KeyManager(key_path).load_or_create() == before
    assert not stale.exists()
    assert young.exists() and other.exists()
    assert key_path.read_bytes() == before


# --- F71: truncation stays detected, and a head can't just be deleted --------

def _fill(path, n=5):
    log = AuditLog(path, b"k" * 32)
    for i in range(n):
        log.log("add", f"m{i}")
    return log


def _truncate_to(path, n):
    lines = path.read_text().splitlines(keepends=True)
    path.write_text("".join(lines[:n]))


def test_truncation_is_still_detected_after_the_next_append(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path)
    _truncate_to(path, 3)
    assert log.verify_chain() is False
    log.log("get", "m9")  # a legitimate append must not launder the truncation
    assert log.verify_chain() is False
    entries = log.tail()
    tamper = [e for e in entries if e["action"] == TAMPER_ACTION]
    assert len(tamper) == 1
    assert tamper[0]["metadata"]["head_as_found"]["count"] == 5
    assert tamper[0]["metadata"]["found_entries"] == 3
    assert AuditLog(path, b"k" * 32).verify_chain() is False


def test_removing_the_tamper_record_is_caught_again(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path)
    _truncate_to(path, 3)
    log.log("get", "m9")
    _truncate_to(path, 3)
    assert log.verify_chain() is False


def test_deleting_the_head_with_the_tail_is_detected(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path)
    _truncate_to(path, 2)
    log.head_path.unlink()
    assert log.verify_chain() is False
    log.log("get", "m9")
    assert log.verify_chain() is False
    assert any(e["action"] == TAMPER_ACTION for e in log.tail())


def test_forged_head_is_detected_and_recorded(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path)
    head = json.loads(log.head_path.read_text())
    head["count"] = 3
    log.head_path.write_text(json.dumps(head))
    assert log.verify_chain() is False
    log.log("get", "m9")
    assert log.verify_chain() is False


def test_head_lagging_after_a_crash_is_not_tampering(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path, 3)
    saved_head = log.head_path.read_text()
    log.log("add", "m3")
    # Crash between the fsync'd append and the head update: the head is one behind.
    log.head_path.write_text(saved_head)
    assert log.verify_chain() is True
    log.log("add", "m4")
    assert log.verify_chain() is True
    assert not any(e["action"] == TAMPER_ACTION for e in log.tail())
    assert json.loads(log.head_path.read_text())["count"] == 5


def test_untouched_log_verifies(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path)
    assert log.verify_chain() is True
    assert AuditLog(path, b"k" * 32).verify_chain() is True


# --- F34: a torn last line is typed, located and repairable ------------------

def test_torn_tail_is_typed_and_repairable(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path, 3)
    size = path.stat().st_size
    with open(path, "ab") as f:
        f.write(b'{"ts": 1.0, "actor": "sys')
    with pytest.raises(AuditLogCorrupt) as err:
        log.log("add", "m9")
    assert err.value.torn and err.value.offset == size and str(path) in str(err.value)
    with pytest.raises(ValueError):
        AuditLog(path, b"k" * 32)
    assert log.repair_torn_tail() == 25
    assert path.stat().st_size == size
    log.log("add", "m9")
    assert log.verify_chain() is True


def test_a_complete_corrupt_line_is_not_repaired(tmp_path):
    path = tmp_path / "audit.log"
    log = _fill(path, 2)
    with open(path, "ab") as f:
        f.write(b"{not json\n")
    before = path.read_bytes()
    with pytest.raises(AuditLogCorrupt) as err:
        log.repair_torn_tail()
    assert not err.value.torn
    assert path.read_bytes() == before


# --- F72: a card behind or before other digit groups -------------------------

@pytest.mark.parametrize("text, card", [
    ("qty 2 4111 1111 1111 1111", "4111 1111 1111 1111"),
    ("ref 12 4111-1111-1111-1111 ok", "4111-1111-1111-1111"),
    ("4111 1111 1111 1111 22", "4111 1111 1111 1111"),
    ("a 1 2 3 5500 0000 0000 0004 b", "5500 0000 0000 0004"),
])
def test_card_with_neighbouring_digits_is_blocked_and_redacted(text, card):
    allowed, _, findings = PIIFilter(action="block").check_and_act(text)
    assert allowed is False and findings["credit_card"] == [card]
    _, out, _ = PIIFilter(action="redact").check_and_act(text)
    assert card not in out and "[CREDIT_CARD_REDACTED]" in out
    assert out.replace("[CREDIT_CARD_REDACTED]", card) == text


def test_long_non_card_numbers_are_left_alone():
    for text in ("id 12345678901234567890123", "order 1234 5678 9012 3456 7890",
                 "2026-09-25 12 34 56"):
        assert "credit_card" not in PIIFilter().scan(text), text


def test_pathological_digit_runs_stay_fast():
    text = "1 " * 20000 + "x"
    start = time.time()
    PIIFilter(action="redact").check_and_act(text)
    assert time.time() - start < 5
