"""Regression tests for the graph / sync / Postgres findings of the 2026-09-24 audit.

Everything runs in tmp dirs. The "v1" knowledge-graph fixture writes rows exactly
the way the graph code before this change did (plaintext lowercased names as ids
and edge endpoints, relation in a clear column, memory_id inside the edge props),
because the live stores hold graphs written that way and must migrate cleanly.

No Postgres server and no pgvector exist here, so PostgresStore is exercised on
SQLite through SQLAlchemy with a stand-in vector type. That covers the ORM code
(put/get/update_tier/...), not the Postgres-only DDL in init().
"""

import json
import sqlite3
import sys
import time
import types
import uuid

import httpx
import pytest

from memcore_memory.core.ebbinghaus import ForgettingCurve
from memcore_memory.core.tiers import MemoryItem, Tier, TIER_BASE_STRENGTH
from memcore_memory.crypto.aes_gcm import AES256GCM
from memcore_memory.graph.extractor import extract_entities
from memcore_memory.graph.kg import KnowledgeGraph, KnowledgeGraphUnreadable
from memcore_memory.sync import gossip as gossip_mod
from memcore_memory.sync.crdt import LWWElementSet, MemoryCRDT, ORSet
from memcore_memory.sync.gossip import GossipProtocol
from memcore_memory.sync.p2p import P2PNode


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


def _all_cells(path):
    con = sqlite3.connect(path)
    out = []
    for (table,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        for row in con.execute(f"SELECT * FROM {table}"):
            out.extend(row)
    con.close()
    return out


# --- F4: no entity name or relation in clear -------------------------------

async def test_kg_file_holds_no_entity_name_or_relation(tmp_path, kg):
    await kg.add_memory_entities(_item(["Kovács-Dobos Ádám", "SkyNAS"]))
    await kg.add_entity("WireGuard", props={"port": 51820})
    await kg.add_relation("SkyNAS", "Ubuntu", relation="runs-secretly-on")

    for cell in _all_cells(tmp_path / "memory.kg.db"):
        if isinstance(cell, str):
            low = cell.lower()
            for secret in ("skynas", "kovács", "wireguard", "ubuntu", "runs-secretly-on"):
                assert secret not in low, (secret, cell)
    raw = (tmp_path / "memory.kg.db").read_bytes().lower()
    for secret in ("skynas", "kov", "wireguard", "ubuntu", "runs-secretly"):
        assert secret.encode() not in raw

    # ... while every read still speaks names.
    names = {e["label"] for e in await kg.list_entities()}
    assert names == {"Kovács-Dobos Ádám", "SkyNAS", "WireGuard", "Ubuntu"}
    edges = await kg.traverse("skynas", depth=1)
    assert {(e["src"], e["dst"], e["relation"]) for e in edges} == {
        ("kovács-dobos ádám", "skynas", "co_occurs"), ("skynas", "ubuntu", "runs-secretly-on")}


async def test_node_ids_are_keyed_not_plain_hashes(tmp_path):
    a = KnowledgeGraph(tmp_path / "a.db", AES256GCM(AES256GCM.generate_key()))
    b = KnowledgeGraph(tmp_path / "b.db", AES256GCM(AES256GCM.generate_key()))
    assert a._nid("SkyNAS") == a._nid("skynas")
    assert a._nid("SkyNAS") != b._nid("SkyNAS")  # unguessable without the key
    # NFC: decomposed and precomposed spellings are one entity.
    assert a._nid("Ádám") == a._nid("Ádám")


# --- v1 migration -------------------------------------------------------------

def _write_v1_graph(path, cipher, memories, relations=()):
    """What the graph code before 2026-09-24 wrote, statement for statement."""
    con = sqlite3.connect(path)
    con.execute('''CREATE TABLE IF NOT EXISTS kg_nodes (id TEXT PRIMARY KEY, type TEXT, label_enc BLOB,
                   nonce BLOB, props_enc BLOB, props_nonce BLOB)''')
    con.execute('''CREATE TABLE IF NOT EXISTS kg_edges (id TEXT PRIMARY KEY, src TEXT, dst TEXT, relation TEXT,
                   weight REAL, timestamp REAL, props_enc BLOB, props_nonce BLOB)''')
    for mid, ents in memories:
        for ent in ents:
            n, ct = cipher.encrypt(ent.encode())
            pn, pct = cipher.encrypt(b'{}')
            con.execute('INSERT OR IGNORE INTO kg_nodes VALUES (?,?,?,?,?,?)', (ent.lower(), 'entity', ct, n, pct, pn))
        for i in range(len(ents)):
            for j in range(i + 1, len(ents)):
                n, ct = cipher.encrypt(json.dumps({'memory_id': mid}).encode())
                con.execute('INSERT INTO kg_edges VALUES (?,?,?,?,?,?,?,?)',
                            (str(uuid.uuid4()), ents[i].lower(), ents[j].lower(), 'co_occurs', 1.0, time.time(), ct, n))
    for src, dst, rel in relations:
        src, dst = src.lower(), dst.lower()
        for node in (src, dst):
            n, ct = cipher.encrypt(node.encode())
            pn, pct = cipher.encrypt(b'{}')
            con.execute('INSERT OR IGNORE INTO kg_nodes VALUES (?,?,?,?,?,?)', (node, 'entity', ct, n, pct, pn))
        n, ct = cipher.encrypt(b'{}')
        con.execute('INSERT INTO kg_edges VALUES (?,?,?,?,?,?,?,?)',
                    (str(uuid.uuid4()), src, dst, rel, 1.0, time.time(), ct, n))
    con.commit()
    con.close()


async def test_v1_graph_is_migrated_in_place(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    _write_v1_graph(path, cipher, [("m1", ["SkyNAS", "Docker"]), ("m2", ["SkyNAS", "WireGuard"])],
                    relations=[("SkyNAS", "Ubuntu", "runs")])
    before = sqlite3.connect(path).execute("SELECT id FROM kg_nodes").fetchall()
    assert ("skynas",) in before

    g = KnowledgeGraph(path, cipher)
    await g.init()
    assert not g.created_fresh
    assert g.needs_link_backfill

    raw = path.read_bytes().lower()
    for name in (b"skynas", b"docker", b"wireguard", b"ubuntu", b"runs"):
        assert name not in raw, name
    assert {e["label"] for e in await g.list_entities()} == {"SkyNAS", "Docker", "WireGuard", "ubuntu"}
    rels = {(e["src"], e["dst"], e["relation"]) for e in await g.traverse("SkyNAS", depth=1)}
    assert rels == {("skynas", "docker", "co_occurs"), ("skynas", "wireguard", "co_occurs"),
                    ("skynas", "ubuntu", "runs")}
    # Links recovered from the v1 edges' encrypted memory_id.
    assert set(await g.get_related_memories("what runs on SkyNAS")) == {"m1", "m2"}
    assert await g.get_related_memories("docker") == ["m1"]

    # Reopening does not migrate again.
    g2 = KnowledgeGraph(path, cipher)
    await g2.init()
    assert len(await g2.list_entities()) == 4


async def test_v1_single_entity_memories_come_back_through_rebuild_links(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    # A memory with one entity had no edge, so v1 kept no trace of it.
    _write_v1_graph(path, cipher, [("solo", ["Jellyfin"])])
    g = KnowledgeGraph(path, cipher)
    await g.init()
    assert await g.get_related_memories("jellyfin") == []
    solo = _item(["Jellyfin"])
    solo.id = "solo"
    assert await g.rebuild_links([solo, _item([])]) == 1
    assert await g.get_related_memories("jellyfin") == ["solo"]
    assert not g.needs_link_backfill
    g2 = KnowledgeGraph(path, cipher)
    await g2.init()
    assert not g2.needs_link_backfill


async def test_v1_migration_under_a_foreign_key_changes_nothing(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    _write_v1_graph(path, cipher, [("m1", ["SkyNAS", "Docker"])])
    snapshot = sorted(_all_cells(path), key=repr)
    g = KnowledgeGraph(path, AES256GCM(AES256GCM.generate_key()))
    with pytest.raises(KnowledgeGraphUnreadable, match=str(path)):
        await g.init()
    con = sqlite3.connect(path)
    assert con.execute("SELECT id FROM kg_nodes ORDER BY id").fetchall() == [("docker",), ("skynas",)]
    assert con.execute("SELECT relation FROM kg_edges").fetchall() == [("co_occurs",)]
    con.close()
    # The right key still migrates it.
    g = KnowledgeGraph(path, cipher)
    await g.init()
    assert await g.get_related_memories("docker") == ["m1"]


# --- get_related_memories / remove_memory contract -------------------------

async def test_get_related_memories_matches_whole_names_and_ranks(kg):
    a, b, c = _item(["SkyNAS", "Home Assistant"]), _item(["SkyNAS"]), _item(["NAS"])
    for it in (a, b, c):
        await kg.add_memory_entities(it)
    # Both entities named -> a first; "nas" is not found inside "SkyNAS".
    assert await kg.get_related_memories("does Home Assistant run on SkyNAS?") == [a.id, b.id]
    assert await kg.get_related_memories("home  assistant") == [a.id]
    assert await kg.get_related_memories("the NAS box") == [c.id]
    assert await kg.get_related_memories("nothing here") == []
    assert await kg.get_related_memories("") == []
    assert await kg.get_related_memories("skynas", limit=1) == [b.id]  # rarer first? no: tie -> newest
    # Idempotent: replaying a memory adds no duplicate links or edges.
    await kg.add_memory_entities(a)
    assert len(await kg.traverse("home assistant", depth=1)) == 1


async def test_remove_memory_drops_links_edges_and_orphans(tmp_path, kg):
    a = _item(["SkyNAS", "Docker"])
    b = _item(["SkyNAS", "WireGuard"])
    await kg.add_memory_entities(a)
    await kg.add_memory_entities(b)
    await kg.add_entity("Pinned", props={"keep": True})
    await kg.add_memory_entities(_item(["Pinned"]))
    # 2 links + 1 edge + orphaned Docker; SkyNAS is still used by b.
    assert await kg.remove_memory(a.id) == 4
    assert await kg.get_related_memories("docker skynas") == [b.id]
    labels = {e["label"] for e in await kg.list_entities()}
    assert labels == {"SkyNAS", "WireGuard", "Pinned"}
    assert await kg.remove_memory(a.id) == 0
    assert await kg.remove_memory("never-existed") == 0


async def test_manual_relation_survives_memory_removal(kg):
    a = _item(["SkyNAS"])
    await kg.add_memory_entities(a)
    await kg.add_relation("SkyNAS", "Ubuntu", relation="runs")
    assert await kg.remove_memory(a.id) == 1  # the link only; the node still has an edge
    assert [e["dst"] for e in await kg.traverse("SkyNAS", depth=1)] == ["ubuntu"]


# --- F31: traverse -------------------------------------------------------------

async def test_traverse_depth_dedupe_and_limit(kg):
    for s, d in (("a", "b"), ("b", "c"), ("c", "d")):
        await kg.add_relation(s, d)
    one = await kg.traverse("a", depth=1, limit=10)
    assert [(e["src"], e["dst"]) for e in one] == [("a", "b")]
    two = await kg.traverse("a", depth=2, limit=10)
    assert [(e["src"], e["dst"]) for e in two] == [("a", "b"), ("b", "c")]
    assert await kg.traverse("a", depth=0) == []
    for i in range(5):
        await kg.add_relation("hub", f"leaf{i}")
    assert len(await kg.traverse("hub", depth=1, limit=2)) == 2


async def test_direct_neighbours_only_at_depth_one(kg):
    await kg.add_relation("a", "b")
    await kg.add_relation("c", "b")
    related = {e["dst"] if e["src"] == "a" else e["src"] for e in await kg.traverse("a", depth=1)} - {"a"}
    assert related == {"b"}  # c is two hops away, and used to be reported


# --- F87 / F88 -----------------------------------------------------------------

async def test_relation_keeps_label_case_and_does_not_override(kg):
    await kg.add_entity("Ubuntu")
    await kg.add_relation("SkyNAS", "UBUNTU")
    labels = {e["id"]: e["label"] for e in await kg.list_entities()}
    assert labels == {"skynas": "SkyNAS", "ubuntu": "Ubuntu"}


async def test_graph_state_is_the_file_not_process_memory(tmp_path, cipher, kg):
    assert not hasattr(kg, "graph")
    await kg.add_relation("alice", "bob", relation="knows")
    other = KnowledgeGraph(tmp_path / "memory.kg.db", cipher)
    await other.init()
    assert [(e["src"], e["dst"], e["relation"]) for e in await other.traverse("alice", depth=1)] == [
        ("alice", "bob", "knows")]


async def test_delete_entity_shape_and_links(kg):
    m = _item(["SkyNAS", "Docker"])
    await kg.add_memory_entities(m)
    out = await kg.delete_entity("SkyNAS")
    assert out == {"entity": "skynas", "nodes_deleted": 1, "edges_deleted": 1}
    assert await kg.get_related_memories("skynas") == []
    assert await kg.get_related_memories("docker") == [m.id]


# --- R12: a bad or missing kg.db ------------------------------------------------

async def test_garbage_kg_file_names_itself_and_is_left_alone(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    junk = bytes(range(256)) * 16
    path.write_bytes(junk)
    with pytest.raises(KnowledgeGraphUnreadable) as exc:
        await KnowledgeGraph(path, cipher).init()
    assert str(path) in str(exc.value)
    assert path.read_bytes() == junk


async def test_fresh_graph_is_flagged(tmp_path, cipher):
    g = KnowledgeGraph(tmp_path / "new.kg.db", cipher)
    await g.init()
    assert g.created_fresh and g.needs_link_backfill
    await g.rebuild_links([])
    g2 = KnowledgeGraph(tmp_path / "new.kg.db", cipher)
    await g2.init()
    assert not g2.created_fresh and not g2.needs_link_backfill


async def test_newer_schema_is_refused(tmp_path, cipher):
    path = tmp_path / "memory.kg.db"
    g = KnowledgeGraph(path, cipher)
    await g.init()
    con = sqlite3.connect(path)
    con.execute("UPDATE kg_meta SET v='99' WHERE k='schema_v'")
    con.commit()
    con.close()
    with pytest.raises(KnowledgeGraphUnreadable, match="v99"):
        await KnowledgeGraph(path, cipher).init()


# --- F89: extractor -------------------------------------------------------------

def test_extractor_handles_this_projects_names():
    assert extract_entities("SkyNAS runs Docker") == ["SkyNAS", "Docker"]
    assert extract_entities("Kovács-Dobos Ádám wrote it") == ["Kovács-Dobos Ádám"]
    assert extract_entities("WireGuard on SkyNAS") == ["WireGuard", "SkyNAS"]
    assert extract_entities("I met Alice Smith in Budapest") == ["Alice Smith", "Budapest"]
    assert extract_entities("The SkyNAS. The skynas, SKYNAS") == ["SkyNAS"]
    assert extract_entities("") == []
    assert len(extract_entities(" ".join(f"Name{i}, " for i in range(30)))) == 10


# --- F90: CRDT naming, in-place merge --------------------------------------------

def test_tombstone_set_is_lww_and_old_name_still_imports():
    assert ORSet is LWWElementSet
    s = LWWElementSet()
    s.adds["x"] = 100.0
    s.removes["x"] = 100.0
    assert not s.contains("x")  # remove wins ties: LWW, not add-wins


def test_merge_from_updates_the_shared_object():
    node = P2PNode(node_id="n1")
    assert node.gossip.crdt is node.crdt
    remote = MemoryCRDT("n2")
    remote.update("m1", {"c": 1})
    node.gossip.crdt.merge_from(remote)
    assert node.crdt.live_ids() == ["m1"]


# --- F32 / F48: P2P and gossip fail loudly ----------------------------------------

async def test_p2p_entry_points_refuse():
    node = P2PNode(node_id="n1")
    assert node.host == "127.0.0.1"
    with pytest.raises(NotImplementedError):
        await node.start()
    with pytest.raises(NotImplementedError):
        await node.broadcast_memory("m1", {"content": "x"})
    await node.stop()  # nothing started, nothing to clean up


async def test_p2p_handler_never_acks():
    sent = []

    class WS:
        def __init__(self, msgs):
            self.msgs = msgs

        def __aiter__(self):
            async def gen():
                for m in self.msgs:
                    yield m
            return gen()

        async def send(self, m):
            sent.append(json.loads(m))

    await P2PNode(node_id="n1")._handler(WS([json.dumps({"type": "sync"}), json.dumps({"type": "memory"}),
                                             json.dumps({"type": "bogus"}), "not json"]))
    assert [m["type"] for m in sent] == ["error"] * 4
    assert "not implemented" in sent[0]["error"]
    assert "bogus" in sent[2]["error"]


async def test_gossip_loop_refuses_and_surfaces_http_errors(monkeypatch):
    g = GossipProtocol(MemoryCRDT("n1"), ["http://peer"])
    with pytest.raises(NotImplementedError):
        await g.gossip_loop()

    status = {"code": 501}
    real_client = httpx.AsyncClient

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(lambda req: httpx.Response(status["code"], json={}))
        return real_client(*a, **kw)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    with pytest.raises(httpx.HTTPStatusError):
        await g._sync_with_peer("http://peer")
    status["code"] = 200
    with pytest.raises(NotImplementedError):
        await g._sync_with_peer("http://peer")


# --- Postgres store (F6, F15, F49, F91, contract 1) --------------------------------

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


def test_embedding_column_follows_the_requested_dim(pg_mod, cipher):
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.schema import CreateIndex, CreateTable
    for dim in (384, 1024):
        store = pg_mod.PostgresStore("sqlite+aiosqlite:///:memory:", cipher, embedding_dim=dim)
        table = store.MemoryRow.__table__
        ddl = str(CreateTable(table).compile(dialect=postgresql.dialect()))
        assert f"VECTOR({dim})" in ddl
        indexes = " ".join(str(CreateIndex(i).compile(dialect=postgresql.dialect())) for i in table.indexes)
        assert "ivfflat" not in indexes
    assert not hasattr(pg_mod, "KGNodeRow") and not hasattr(pg_mod, "KGEdgeRow")


def _mem(tier=Tier.WORKING, strength=None, entities=("SkyNAS",), content="SkyNAS runs WireGuard"):
    item = MemoryItem(content=content, tier=tier, embedding=[0.1, 0.2, 0.3, 0.4],
                      metadata={"importance": 0.5}, entities=list(entities))
    if strength is not None:
        item.forgetting.strength = strength
    return item


async def test_pg_update_tier_raises_strength_never_lowers(pg):
    item = _mem()
    await pg.put(item)
    assert (await pg.get(item.id)).forgetting.strength == TIER_BASE_STRENGTH[Tier.WORKING]
    assert await pg.update_tier(item.id, Tier.EPISODIC) is True
    got = await pg.get(item.id)
    assert got.tier == Tier.EPISODIC and got.forgetting.strength == TIER_BASE_STRENGTH[Tier.EPISODIC]
    strong = _mem(tier=Tier.SEMANTIC, strength=900.0)
    await pg.put(strong)
    await pg.update_tier(strong.id, Tier.EPISODIC)
    assert (await pg.get(strong.id)).forgetting.strength == 900.0
    assert await pg.update_tier("missing", Tier.EPISODIC) is False


async def test_pg_contract_methods(pg):
    a, b = _mem(), _mem(content="other")
    await pg.put(a)
    await pg.put(b)
    assert set(await pg.get_many([a.id, b.id, "nope", a.id])) == {a.id, b.id}
    curve = ForgettingCurve(strength=42.0, importance=0.5)
    assert await pg.update_forgetting(a.id, curve) is True
    assert (await pg.get(a.id)).forgetting.strength == 42.0
    assert await pg.delete(a.id) is True
    assert await pg.delete(a.id) is False
    # never inserts: a rehearsal racing a delete must not bring the row back
    assert await pg.update_forgetting(a.id, curve) is False
    assert await pg.get(a.id) is None
    b.content = "edited"
    assert await pg.update_content(b) is True
    assert (await pg.get(b.id)).content == "edited"


async def test_pg_entities_are_encrypted_and_blind_index_works(tmp_path, pg):
    item = _mem(entities=["Kovács-Dobos Ádám"])
    await pg.put(item)
    assert (await pg.get(item.id)).entities == ["Kovács-Dobos Ádám"]
    con = sqlite3.connect(tmp_path / "pg.db")
    row = con.execute("SELECT entities_json, blind_index_json FROM memories").fetchone()
    con.close()
    assert row[0] is None and "Kov" not in (row[1] or "")
    assert await pg.search_by_blind_index("wireguard") == [item.id]
    assert await pg.search_by_blind_index("zebra") == []
    assert await pg.rebuild_blind_index() == 1


async def test_pg_reads_and_migrates_rows_written_by_the_old_schema(tmp_path, cipher, pg_mod):
    """Rows from the previous PostgresStore: no entities_enc/blind_index_json
    columns, entity list in clear."""
    path = tmp_path / "old.db"
    con = sqlite3.connect(path)
    con.execute('''CREATE TABLE memories (id VARCHAR PRIMARY KEY, tier VARCHAR, timestamp FLOAT,
                   content_enc BLOB NOT NULL, nonce BLOB NOT NULL, metadata_enc BLOB NOT NULL,
                   meta_nonce BLOB NOT NULL, forgetting_json TEXT, entities_json TEXT,
                   embedding VECTOR(4), importance FLOAT)''')
    item = _mem(tier=Tier.EPISODIC, entities=["SkyNAS", "Docker"])
    cn, cc = cipher.encrypt(item.content.encode())
    mn, mc = cipher.encrypt(json.dumps(item.metadata).encode())
    con.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, "episodic", item.timestamp, cc, cn, mc, mn, json.dumps(item.forgetting.to_dict()),
                 json.dumps(item.entities), json.dumps(item.embedding), 0.5))
    con.commit()
    con.close()

    store = pg_mod.PostgresStore(f"sqlite+aiosqlite:///{path}", cipher, embedding_dim=4)
    await store.init()
    got = await store.get(item.id)
    assert got.content == item.content and got.entities == ["SkyNAS", "Docker"]
    con = sqlite3.connect(path)
    assert con.execute("SELECT entities_json FROM memories").fetchone() == (None,)
    con.close()
    await store.engine.dispose()
