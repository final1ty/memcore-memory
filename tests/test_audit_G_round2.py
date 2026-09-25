"""Round-2 regression tests for the graph / Postgres fixes of the 2026-09-24 audit.

Everything runs in tmp dirs. The "old format" fixtures write rows statement for
statement the way earlier code did: the committed HEAD graph code (plaintext
lowercased ids, relation in a clear column, memory id inside the edge props), the
schema-v2 graph (hashed ids, ciphertexts without AAD) and the previous Postgres
row format (no aad_v, no blind_index_json). Stores written by each exist, and
must open.

PostgresStore runs on SQLite through SQLAlchemy with a stand-in vector type, as
in test_audit_g_graph_sync_postgres.py: the ORM paths are covered, the
Postgres-only DDL is exercised only through a recording fake connection.
"""

import hashlib
import json
import os
import sqlite3
import stat
import sys
import time
import types
import uuid

import pytest

from memcore_memory.core.tiers import MemoryItem, Tier
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.graph.kg import SCHEMA_VERSION, KnowledgeGraph, KnowledgeGraphUnreadable
from memcore_memory.storage.encrypted_sqlite import CorruptRow


@pytest.fixture
def cipher():
    return AES256GCM(AES256GCM.generate_key())


@pytest.fixture
async def kg(tmp_path, cipher):
    g = KnowledgeGraph(tmp_path / "memory.kg.db", cipher)
    await g.init()
    return g


def _item(entities, content="x"):
    return MemoryItem(content=content, tier=Tier.EPISODIC, entities=list(entities))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _q(path, sql, args=()):
    con = sqlite3.connect(path)
    try:
        return con.execute(sql, args).fetchall()
    finally:
        con.close()


# --- remove_memory keeps what was added by hand ----------------------------------

async def test_hand_added_entity_without_props_survives_memory_removal(kg):
    await kg.add_entity("Alice", "person")
    m1 = _item(["Alice"])
    await kg.add_memory_entities(m1)
    assert await kg.remove_memory(m1.id) == 1  # the link, not the node
    assert [e["label"] for e in await kg.list_entities()] == ["Alice"]
    # Default type too: the flag, not the type, is what protects it.
    await kg.add_entity("Bob")
    m2 = _item(["Bob"])
    await kg.add_memory_entities(m2)
    await kg.remove_memory(m2.id)
    assert await kg.has_entity("bob")


async def test_relation_endpoint_becomes_hand_added(kg):
    m = _item(["SkyNAS"])
    await kg.add_memory_entities(m)
    await kg.add_relation("SkyNAS", "Ubuntu", relation="runs")
    await kg.delete_entity("Ubuntu")  # drops the edge too
    await kg.remove_memory(m.id)
    assert await kg.has_entity("skynas")


async def test_memory_created_orphans_are_still_collected(kg):
    m = _item(["Docker", "Portainer"])
    await kg.add_memory_entities(m)
    assert await kg.remove_memory(m.id) == 2 + 1 + 2
    assert await kg.list_entities() == []


# --- AAD binds ciphertexts to their row ------------------------------------------

async def test_label_moved_to_another_node_fails_authentication(tmp_path, kg):
    await kg.add_entity("Alice")
    await kg.add_entity("Mallory")
    path = tmp_path / "memory.kg.db"
    rows = _q(path, "SELECT id, label_enc, nonce FROM kg_nodes ORDER BY rowid")
    con = sqlite3.connect(path)
    (a, a_ct, a_n), (m, m_ct, m_n) = rows
    con.execute("UPDATE kg_nodes SET label_enc=?, nonce=? WHERE id=?", (m_ct, m_n, a))
    con.commit()
    con.close()
    fresh = KnowledgeGraph(path, kg.cipher)
    await fresh.init()
    # Alice's row now carries Mallory's blob: skipped, never shown as "Mallory".
    assert [e["label"] for e in await fresh.list_entities()] == ["Mallory"]


async def test_edge_props_moved_to_another_edge_are_rejected(tmp_path, kg):
    await kg.add_relation("a", "b", relation="trusts")
    await kg.add_relation("c", "d", relation="distrusts")
    path = tmp_path / "memory.kg.db"
    (e1, ct1, n1), (e2, ct2, n2) = _q(path, "SELECT id, props_enc, props_nonce FROM kg_edges ORDER BY rowid")
    con = sqlite3.connect(path)
    con.execute("UPDATE kg_edges SET props_enc=?, props_nonce=? WHERE id=?", (ct2, n2, e1))
    con.commit()
    con.close()
    edges = await kg.traverse("a", depth=1)
    assert [e["relation"] for e in edges] == [None]  # not "distrusts"


async def test_every_row_written_is_bound(tmp_path, kg):
    await kg.add_memory_entities(_item(["SkyNAS", "Docker"]))
    await kg.add_entity("WireGuard", props={"port": 51820})
    await kg.add_relation("SkyNAS", "Ubuntu")
    path = tmp_path / "memory.kg.db"
    assert _q(path, "SELECT COUNT(*) FROM kg_nodes WHERE aad_v IS NULL") == [(0,)]
    assert _q(path, "SELECT COUNT(*) FROM kg_edges WHERE aad_v IS NULL") == [(0,)]
    assert _q(path, "SELECT v FROM kg_meta WHERE k='schema_v'") == [(str(SCHEMA_VERSION),)]


# --- old formats open --------------------------------------------------------------

def _write_v2_graph(path, cipher, nid, memories, relations=()):
    """What the schema-v2 graph code wrote: hashed ids, no AAD, no aad_v/auto."""
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, nonce BLOB, '
                'props_enc BLOB, props_nonce BLOB)')
    con.execute('CREATE TABLE kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT, weight REAL, '
                'timestamp REAL, props_enc BLOB, props_nonce BLOB, memory_id TEXT)')
    con.execute('CREATE TABLE memory_entities (memory_id TEXT NOT NULL, node_id TEXT NOT NULL, '
                'UNIQUE (memory_id, node_id))')
    con.execute('CREATE TABLE kg_meta (k TEXT PRIMARY KEY, v TEXT)')

    def node(name):
        n, ct = cipher.encrypt(name.encode())
        pn, pct = cipher.encrypt(b'{}')
        con.execute('INSERT OR IGNORE INTO kg_nodes VALUES (?,?,?,?,?,?)', (nid(name), 'entity', ct, n, pct, pn))

    for mid, ents in memories:
        for ent in ents:
            node(ent)
            con.execute('INSERT OR IGNORE INTO memory_entities VALUES (?, ?)', (mid, nid(ent)))
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                n, ct = cipher.encrypt(json.dumps({'relation': 'co_occurs'}).encode())
                con.execute('INSERT INTO kg_edges VALUES (?,?,?,?,?,?,?,?,?)',
                            (str(uuid.uuid4()), nid(ents[i]), nid(ents[j]), None, 1.0, time.time(), ct, n, mid))
    for src, dst, rel in relations:
        node(src)
        node(dst)
        n, ct = cipher.encrypt(json.dumps({'relation': rel}).encode())
        con.execute('INSERT INTO kg_edges VALUES (?,?,?,?,?,?,?,?,?)',
                    (str(uuid.uuid4()), nid(src), nid(dst), None, 1.0, time.time(), ct, n, None))
    con.execute("INSERT INTO kg_meta VALUES ('schema_v', '2')")
    con.execute("INSERT INTO kg_meta VALUES ('links_complete', '1')")
    con.commit()
    con.close()


async def test_v2_graph_is_rebound_in_place(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    probe = KnowledgeGraph(tmp_path / "probe.db", cipher)
    _write_v2_graph(path, cipher, probe._nid, [("m1", ["SkyNAS", "Docker"]), ("solo", ["Jellyfin"])],
                    relations=[("SkyNAS", "Ubuntu", "runs")])
    g = KnowledgeGraph(path, cipher)
    await g.init()
    assert not g.needs_link_backfill  # v2 already had complete links
    assert _q(path, "SELECT COUNT(*) FROM kg_nodes WHERE aad_v IS NULL") == [(0,)]
    assert _q(path, "SELECT COUNT(*) FROM kg_edges WHERE aad_v IS NULL") == [(0,)]
    assert _q(path, "SELECT v FROM kg_meta WHERE k='schema_v'") == [(str(SCHEMA_VERSION),)]
    assert {e["label"] for e in await g.list_entities()} == {"SkyNAS", "Docker", "Jellyfin", "Ubuntu"}
    rels = {(e["src"], e["dst"], e["relation"]) for e in await g.traverse("skynas", depth=1)}
    assert rels == {("skynas", "docker", "co_occurs"), ("skynas", "ubuntu", "runs")}
    assert await g.get_related_memories("jellyfin") == ["solo"]
    # Origin unknown for every pre-existing node: none is collected as an orphan.
    assert await g.remove_memory("solo") == 1
    assert await g.has_entity("Jellyfin")


def _head_writes(path, cipher, mid, entities):
    """add_memory_entities of the committed HEAD code, statement for statement."""
    con = sqlite3.connect(path)
    for ent in entities:
        n, ct = cipher.encrypt(ent.encode())
        pn, pct = cipher.encrypt(b'{}')
        con.execute('INSERT OR IGNORE INTO kg_nodes (id, type, label_enc, nonce, props_enc, props_nonce) '
                    'VALUES (?,?,?,?,?,?)', (ent.lower(), 'entity', ct, n, pct, pn))
    for i in range(len(entities)):
        for j in range(i + 1, len(entities)):
            n, ct = cipher.encrypt(json.dumps({'memory_id': mid}).encode())
            con.execute('INSERT INTO kg_edges (id, src, dst, relation, weight, timestamp, props_enc, props_nonce) '
                        'VALUES (?,?,?,?,?,?,?,?)',
                        (str(uuid.uuid4()), entities[i].lower(), entities[j].lower(), 'co_occurs', 1.0,
                         time.time(), ct, n))
    con.commit()
    con.close()


async def test_rows_an_older_release_wrote_into_an_upgraded_file_are_migrated(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    g = KnowledgeGraph(path, cipher)
    await g.init()
    first = _item(["SkyNAS"])
    await g.add_memory_entities(first)
    await g.rebuild_links([first])
    # A rollback to the old image writes into the upgraded file.
    _head_writes(path, cipher, "old-1", ["SkyNAS", "WireGuard"])
    assert _q(path, "SELECT COUNT(*) FROM kg_edges WHERE relation IS NOT NULL") == [(1,)]

    g2 = KnowledgeGraph(path, cipher)
    await g2.init()
    raw = path.read_bytes().lower()
    for name in (b"skynas", b"wireguard", b"co_occurs"):
        assert name not in raw, name
    # The stray "skynas" row merged into the existing node, not a duplicate.
    assert sorted(e["label"] for e in await g2.list_entities()) == ["SkyNAS", "WireGuard"]
    assert set(await g2.get_related_memories("wireguard on skynas")) == {"old-1", first.id}
    # The old writer left no links for single-entity memories: rebuild them.
    assert g2.needs_link_backfill


async def test_refused_newer_schema_leaves_the_file_byte_for_byte(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE kg_meta (k TEXT PRIMARY KEY, v TEXT)")
    con.execute("INSERT INTO kg_meta VALUES ('schema_v', '99')")
    con.execute("CREATE TABLE kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, nonce BLOB, "
                "props_enc BLOB, props_nonce BLOB, future_column BLOB)")
    con.commit()
    con.close()
    before = _sha(path)
    with pytest.raises(KnowledgeGraphUnreadable, match="v99"):
        await KnowledgeGraph(path, cipher).init()
    assert _sha(path) == before
    tables = {r[0] for r in _q(path, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"kg_meta", "kg_nodes"}


async def test_aborted_v1_migration_leaves_the_file_byte_for_byte(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    _head_writes(path, cipher, "m1", [])  # nothing yet: create the v1 tables first
    con = sqlite3.connect(path)
    con.execute('CREATE TABLE IF NOT EXISTS kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB, '
                'nonce BLOB, props_enc BLOB, props_nonce BLOB)')
    con.execute('CREATE TABLE IF NOT EXISTS kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT, '
                'weight REAL, timestamp REAL, props_enc BLOB, props_nonce BLOB)')
    con.commit()
    con.close()
    _head_writes(path, cipher, "m1", ["SkyNAS", "Docker"])
    before = _sha(path)
    with pytest.raises(KnowledgeGraphUnreadable, match="file left unchanged"):
        await KnowledgeGraph(path, AES256GCM(AES256GCM.generate_key())).init()
    assert _sha(path) == before
    columns = {r[1] for r in _q(path, "PRAGMA table_info(kg_edges)")}
    assert "memory_id" not in columns and "aad_v" not in columns


async def test_graph_runs_in_wal_mode_and_is_private(tmp_path, kg):
    path = tmp_path / "memory.kg.db"
    assert _q(path, "PRAGMA journal_mode") == [("wal",)]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


# --- Postgres (on SQLite, stand-in vector type) ----------------------------------

def _install_vector_stub():
    try:
        import pgvector.sqlalchemy  # noqa: F401
        return
    except ImportError:
        pass
    from sqlalchemy.types import UserDefinedType

    class Vector(UserDefinedType):
        cache_ok = True

        def __init__(self, dim=None):
            self.dim = dim

        def get_col_spec(self, **kw):
            return f"VECTOR({self.dim})" if self.dim else "VECTOR"

        def bind_processor(self, dialect):
            return lambda v: None if v is None else json.dumps(list(v))

        def result_processor(self, dialect, coltype):
            return lambda v: None if v is None else json.loads(v)

    pkg = types.ModuleType("pgvector")
    sub = types.ModuleType("pgvector.sqlalchemy")
    sub.Vector = Vector
    pkg.sqlalchemy = sub
    sys.modules.setdefault("pgvector", pkg)
    sys.modules.setdefault("pgvector.sqlalchemy", sub)


@pytest.fixture
def pg_mod():
    _install_vector_stub()
    from memcore_memory.storage import postgres
    return postgres


@pytest.fixture
async def pg(tmp_path, cipher, pg_mod):
    store = pg_mod.PostgresStore(f"sqlite+aiosqlite:///{tmp_path / 'pg.db'}", cipher, embedding_dim=4)
    await store.init()
    yield store
    await store.engine.dispose()


def _mem(content="SkyNAS runs WireGuard", tier=Tier.EPISODIC, ts=None):
    item = MemoryItem(content=content, tier=tier, embedding=[0.1, 0.2, 0.3, 0.4],
                      metadata={"importance": 0.5}, entities=["SkyNAS"])
    if ts is not None:
        item.timestamp = ts
    return item


async def test_pg_content_moved_between_rows_is_rejected(tmp_path, pg):
    a, b = _mem("alpha secret"), _mem("bravo secret")
    await pg.put(a)
    await pg.put(b)
    path = tmp_path / "pg.db"
    assert _q(path, "SELECT aad_v FROM memories") == [(1,), (1,)]
    ct, n = _q(path, "SELECT content_enc, nonce FROM memories WHERE id=?", (b.id,))[0]
    con = sqlite3.connect(path)
    con.execute("UPDATE memories SET content_enc=?, nonce=? WHERE id=?", (ct, n, a.id))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow, match="content_enc"):
        await pg.get(a.id)
    assert list(await pg.get_many([a.id, b.id])) == [b.id]
    assert [i.id for i in await pg.list_all()] == [b.id]
    with pytest.raises(CorruptRow):
        await pg.list_all(strict=True)
    report = await pg.verify()
    assert report["total"] == 2 and report["readable"] == 1 and a.id in report["unreadable"]


def _old_pg_store(path, cipher, items):
    """The previous PostgresStore's rows: no aad_v, no blind index, no AAD."""
    con = sqlite3.connect(path)
    con.execute('''CREATE TABLE memories (id VARCHAR PRIMARY KEY, tier VARCHAR, timestamp FLOAT,
                   content_enc BLOB NOT NULL, nonce BLOB NOT NULL, metadata_enc BLOB NOT NULL,
                   meta_nonce BLOB NOT NULL, forgetting_json TEXT, entities_json TEXT,
                   embedding VECTOR(4), importance FLOAT)''')
    for item in items:
        cn, cc = cipher.encrypt(item.content.encode())
        mn, mc = cipher.encrypt(json.dumps(item.metadata).encode())
        con.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (item.id, item.tier.value, item.timestamp, cc, cn, mc, mn,
                     json.dumps(item.forgetting.to_dict()), json.dumps(item.entities),
                     json.dumps(item.embedding), 0.5))
    con.commit()
    con.close()


async def test_pg_old_rows_read_get_indexed_and_rebind(tmp_path, cipher, pg_mod):
    path = tmp_path / "old.db"
    old = [_mem("WireGuard tunnel to SkyNAS", ts=100.0), _mem("Jellyfin library", ts=200.0)]
    _old_pg_store(path, cipher, old)
    store = pg_mod.PostgresStore(f"sqlite+aiosqlite:///{path}", cipher, embedding_dim=4)
    await store.init()
    try:
        assert {i.content for i in await store.list_all()} == {i.content for i in old}
        # Backfilled at init: these rows used to be invisible to encrypted search.
        assert await store.search_by_blind_index("wireguard") == [old[0].id]
        assert _q(path, "SELECT COUNT(*) FROM memories WHERE blind_index_json IS NULL") == [(0,)]
        assert _q(path, "SELECT COUNT(*) FROM memories WHERE aad_v IS NULL") == [(2,)]
        assert await store.reencrypt_legacy_rows() == 2
        assert await store.reencrypt_legacy_rows() == 0
        assert _q(path, "SELECT COUNT(*) FROM memories WHERE aad_v IS NULL") == [(0,)]
        assert (await store.get(old[1].id)).content == "Jellyfin library"
    finally:
        await store.engine.dispose()


async def test_pg_list_by_tier_is_newest_first(pg):
    for ts in (10.0, 30.0, 20.0):
        await pg.put(_mem(f"m{ts}", tier=Tier.WORKING, ts=ts))
    assert [i.timestamp for i in await pg.list_by_tier(Tier.WORKING)] == [30.0, 20.0, 10.0]


async def test_pg_update_tier_names_a_corrupt_curve(tmp_path, pg):
    item = _mem()
    await pg.put(item)
    con = sqlite3.connect(tmp_path / "pg.db")
    con.execute("UPDATE memories SET forgetting_json=? WHERE id=?", (json.dumps({"strength": "strong"}), item.id))
    con.commit()
    con.close()
    with pytest.raises(CorruptRow, match="forgetting_json"):
        await pg.update_tier(item.id, Tier.SEMANTIC)


async def test_pg_old_ivfflat_index_is_dropped_even_without_hnsw(pg_mod, cipher):
    store = pg_mod.PostgresStore("sqlite+aiosqlite:///:memory:", cipher, embedding_dim=4)
    ran = []

    class Conn:
        async def execute(self, stmt):
            sql = str(stmt)
            if "hnsw" in sql:
                raise RuntimeError('access method "hnsw" does not exist')
            ran.append(sql)

    class Begin:
        async def __aenter__(self):
            return Conn()

        async def __aexit__(self, *exc):
            return False

    class Engine:
        def begin(self):
            return Begin()

    real = store.engine
    store.engine = Engine()
    try:
        await store._ensure_hnsw()
    finally:
        store.engine = real
        await real.dispose()
    assert ran == ["DROP INDEX IF EXISTS idx_memories_embedding"]
