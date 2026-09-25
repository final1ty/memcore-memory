"""Encrypted knowledge graph over a local SQLite file (memory.kg.db).

What is and is not hidden from someone holding the file without the key:

- Entity names are not stored in clear. A node's id is an HMAC of its normalised
  name under a subkey of the content key (like the blind index), and the name
  itself only exists encrypted in label_enc. Edge relations live inside the
  encrypted props, not in a column.
- Every ciphertext is bound to its row (AES-GCM associated data): a node's label
  and props to its id, an edge's props to its id and endpoints. Moving a blob to
  another row fails authentication instead of renaming an entity or a relation.
- The shape is visible: node and edge counts, which nodes are connected, edge
  weights and timestamps, which memory ids share an entity (memory_entities
  holds plaintext memory ids against node HMACs), and which nodes were created
  by memory linking rather than by hand. Names are protected, structure is not -
  the same tradeoff the blind index makes.

Before schema v2 the id was the lowercased name itself, so the "encryption" hid
only letter case; v2 had hashed ids but unbound ciphertexts. init() upgrades
either in place, and also any row an older version wrote into an upgraded file
(see _migrate_legacy). The upgrade is one-way: to go back to an older release,
move memory.kg.db aside - it is rebuilt from the memories at the next start.
"""

import hashlib, hmac, json, os, re, sqlite3, sys, time, unicodedata, uuid
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import aiosqlite
from ..crypto.aes_gcm import AES256GCM

NODE_KEY_INFO = b"kg-node-v1"
SCHEMA_VERSION = 3
# aad_v on a row: its ciphertexts are bound to it. NULL: written without AAD, by
# schema v1/v2 code or by an older release writing into an upgraded file.
AAD_VERSION = 1

# Same as the memory store: the host graph is opened by the MCP server and the
# CLI at once, and sqlite3's 5 s default gave up too early.
BUSY_TIMEOUT_S = 10.0

_HMAC_ID = re.compile(r'[0-9a-f]{64}')


class KnowledgeGraphUnreadable(Exception):
    """memory.kg.db cannot be opened or migrated. Always names the file."""


def _log(msg: str):
    # stderr only: stdout is the JSON-RPC stream under the stdio MCP server.
    print(f"[kg] {msg}", file=sys.stderr)


def _norm(name: str) -> str:
    # NFC so a precomposed and a decomposed "á" are one entity. lower(), not
    # casefold(): ids used to be name.lower(), and casefold would merge names
    # (straße/strasse) that were distinct nodes before the migration.
    return unicodedata.normalize('NFC', name).lower()


def _node_aad(nid: str, column: str) -> bytes:
    return f"memcore:v{AAD_VERSION}:kg:node:{nid}:{column}".encode()


def _edge_aad(eid: str, src: str, dst: str) -> bytes:
    # The endpoints too: they are plaintext columns, and a relation is only
    # meaningful between the two nodes it was written for.
    return f"memcore:v{AAD_VERSION}:kg:edge:{eid}:{src}:{dst}:props".encode()


class KnowledgeGraph:
    def __init__(self, db_path: Path, cipher: AES256GCM):
        self.db_path = db_path
        self.cipher = cipher
        self._nk = cipher.derive_subkey(NODE_KEY_INFO)
        # node id -> display name. A node's name never changes except in case, so
        # entries never go stale; a deleted node just stops being looked up.
        self._names: Dict[str, str] = {}
        self._patterns: Dict[str, re.Pattern] = {}
        # Set by init(): the file did not exist before this process opened it.
        self.created_fresh = False
        # Set by init(): memory links may be missing (a migrated v1 graph, a new
        # file). rebuild_links() from the store's memories fills them in.
        self.needs_link_backfill = False
        self._vacuum_after_init = False

    def _connect(self):
        return aiosqlite.connect(self.db_path, timeout=BUSY_TIMEOUT_S)

    def _nid(self, name: str) -> str:
        return hmac.new(self._nk, _norm(name).encode('utf-8'), hashlib.sha256).hexdigest()

    def _enc(self, data: bytes, aad: bytes = b""):
        return self.cipher.encrypt(data, aad)

    def _dec_json(self, nonce, ct, aad: bytes = b"") -> dict:
        if ct is None:
            return {}
        value = json.loads(self.cipher.decrypt(nonce, ct, aad).decode())
        return value if isinstance(value, dict) else {}

    def _label(self, nid, label_enc, nonce, aad_v) -> str:
        return self.cipher.decrypt(nonce, label_enc, _node_aad(nid, 'label') if aad_v else b"").decode()

    def _edge_props(self, eid, src, dst, props_enc, props_nonce, aad_v) -> dict:
        return self._dec_json(props_nonce, props_enc, _edge_aad(eid, src, dst) if aad_v else b"")

    async def _put_node(self, db, nid: str, name: str, type_: str = 'entity', props: Dict = None,
                        auto: bool = False, replace: bool = False):
        n, ct = self._enc(name.encode(), _node_aad(nid, 'label'))
        pn, pct = self._enc(json.dumps(props or {}).encode(), _node_aad(nid, 'props'))
        verb = 'INSERT OR REPLACE' if replace else 'INSERT OR IGNORE'
        await db.execute(f'{verb} INTO kg_nodes (id, type, label_enc, nonce, props_enc, props_nonce, aad_v, auto) '
                         'VALUES (?,?,?,?,?,?,?,?)',
                         (nid, type_, ct, n, pct, pn, AAD_VERSION, 1 if auto else None))

    async def _put_edge(self, db, eid: str, src: str, dst: str, props: Dict, weight: float,
                        memory_id: Optional[str]):
        n, ct = self._enc(json.dumps(props).encode(), _edge_aad(eid, src, dst))
        await db.execute('INSERT INTO kg_edges (id, src, dst, relation, weight, timestamp, props_enc, props_nonce, '
                         'memory_id, aad_v) VALUES (?,?,?,?,?,?,?,?,?,?)',
                         (eid, src, dst, None, weight, time.time(), ct, n, memory_id, AAD_VERSION))

    # --- schema -----------------------------------------------------------

    def _create_private(self):
        # 0600 like memory.db: the structure (who shares an entity with whom) is
        # plaintext, and sqlite3 would create the file with the umask default.
        try:
            fd = os.open(self.db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except (FileExistsError, OSError):
            return
        os.close(fd)

    async def init(self):
        existed = Path(self.db_path).exists()
        if not existed:
            self._create_private()
        try:
            async with self._connect() as db:
                await self._upgrade(db)
                links_done = await self._meta(db, 'links_complete')
        except sqlite3.OperationalError as e:
            # Another process holding the write lock past the busy timeout is not a
            # damaged file. Degrading would skip graph writes for the whole life of
            # this process on a graph still marked complete, so those links would
            # never be backfilled: fail the start instead and let it be retried.
            if 'locked' in str(e).lower() or 'busy' in str(e).lower():
                raise
            raise KnowledgeGraphUnreadable(f"knowledge graph at {self.db_path} could not be opened: {e}") from e
        except sqlite3.DatabaseError as e:
            # Every entry point opens the graph, so without the path a damaged
            # kg.db read as a broken memory.db. Never delete or rewrite the file.
            raise KnowledgeGraphUnreadable(
                f"knowledge graph at {self.db_path} is not a readable SQLite database ({e}); "
                f"move it aside to start an empty graph - memory.db is not affected") from e
        self.created_fresh = not existed
        self.needs_link_backfill = links_done != '1'
        await self._set_wal()
        if self._vacuum_after_init:
            await self._vacuum()

    async def _set_wal(self):
        # Only once the file is known good and upgraded: a refused file is left
        # byte for byte as it was. Persistent, so later connections inherit it.
        try:
            async with self._connect() as db:
                await db.execute('PRAGMA journal_mode=WAL')
        except sqlite3.Error as e:
            _log(f"{self.db_path} stays in rollback-journal mode ({e}); readers will block writers")

    async def _create_schema(self, db):
        await db.execute('''CREATE TABLE IF NOT EXISTS kg_nodes (
            id TEXT PRIMARY KEY,
            type TEXT,
            label_enc BLOB,
            nonce BLOB,
            props_enc BLOB,
            props_nonce BLOB,
            aad_v INTEGER,
            auto INTEGER
        )''')
        await db.execute('''CREATE TABLE IF NOT EXISTS kg_edges (
            id TEXT PRIMARY KEY,
            src TEXT,
            dst TEXT,
            relation TEXT,
            weight REAL,
            timestamp REAL,
            props_enc BLOB,
            props_nonce BLOB,
            memory_id TEXT,
            aad_v INTEGER
        )''')
        # auto: 1 when memory linking created the node, so remove_memory may drop
        # it once unused. NULL for nodes made by hand, and for every node older
        # than the column, whose origin is unknown - those are never dropped.
        for table, name in (('kg_nodes', 'aad_v'), ('kg_nodes', 'auto'),
                            ('kg_edges', 'memory_id'), ('kg_edges', 'aad_v')):
            async with db.execute(f"PRAGMA table_info({table})") as cur:
                columns = {row[1] async for row in cur}
            if name not in columns:
                decl = 'TEXT' if name == 'memory_id' else 'INTEGER'
                await db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {decl}')
        # Which memory names which entity. Edges only existed between pairs, so a
        # memory tagged with a single entity used to leave no trace in the graph.
        await db.execute('''CREATE TABLE IF NOT EXISTS memory_entities (
            memory_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            UNIQUE (memory_id, node_id)
        )''')
        await db.execute('CREATE TABLE IF NOT EXISTS kg_meta (k TEXT PRIMARY KEY, v TEXT)')
        # Every traversal step and every delete used to scan the whole edge table.
        await db.execute('CREATE INDEX IF NOT EXISTS idx_kg_edges_src ON kg_edges(src)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_kg_edges_dst ON kg_edges(dst)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_kg_edges_memory ON kg_edges(memory_id)')
        await db.execute('CREATE INDEX IF NOT EXISTS idx_memory_entities_node ON memory_entities(node_id)')

    async def _meta(self, db, k: str) -> Optional[str]:
        async with db.execute('SELECT v FROM kg_meta WHERE k=?', (k,)) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def _set_meta(self, db, k: str, v: str):
        await db.execute('INSERT OR REPLACE INTO kg_meta (k, v) VALUES (?, ?)', (k, v))

    async def _upgrade(self, db):
        # One IMMEDIATE transaction around the version check, the DDL and the
        # migration: two processes opening an old file at once must not both
        # migrate it, and a refused or failed upgrade must leave the file as it
        # was - the schema changes included, which used to be committed first.
        await db.execute('BEGIN IMMEDIATE')
        try:
            async with db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='kg_meta'") as cur:
                has_meta = await cur.fetchone() is not None
            version = await self._meta(db, 'schema_v') if has_meta else None
            try:
                newer = version is not None and int(version) > SCHEMA_VERSION
            except ValueError:
                raise KnowledgeGraphUnreadable(
                    f"knowledge graph at {self.db_path} has an unrecognised schema version {version!r}")
            if newer:
                raise KnowledgeGraphUnreadable(
                    f"knowledge graph at {self.db_path} has schema v{version}, newer than this "
                    f"version understands (v{SCHEMA_VERSION}); refusing to write to it")
            await self._create_schema(db)
            async with db.execute('SELECT EXISTS(SELECT 1 FROM kg_nodes WHERE aad_v IS NULL) '
                                  'OR EXISTS(SELECT 1 FROM kg_edges WHERE aad_v IS NULL)') as cur:
                legacy = (await cur.fetchone())[0]
            if legacy:
                plaintext = await self._migrate_legacy(db)
                self._vacuum_after_init = plaintext
                if version == str(SCHEMA_VERSION):
                    # Rows without AAD in an upgraded file: an older release wrote
                    # here after the upgrade (a rollback, an image not rebuilt).
                    # It left no memory links for single-entity memories.
                    _log(f"{self.db_path}: upgraded rows written by an older version; "
                         f"memory links will be rebuilt from the store")
                    await self._set_meta(db, 'links_complete', '0')
            await self._set_meta(db, 'schema_v', str(SCHEMA_VERSION))
            await db.commit()
        except BaseException:
            try:
                await db.rollback()
            except sqlite3.Error:
                pass
            raise

    async def _migrate_legacy(self, db) -> bool:
        """Bring every row without AAD up to the current format, in place.

        That is a whole v1 file (plaintext-name ids, relation in a clear column,
        memory id inside the edge props), a v2 file (hashed ids, unbound
        ciphertexts), or such rows written into an upgraded file by an older
        release. Every name is recovered from label_enc, never from the id, so a
        node whose id and label disagree ends up under the name it displays.
        Runs inside the caller's transaction: a ciphertext that fails to decrypt
        (a foreign key) aborts it all and leaves the file exactly as it was.
        Returns whether names or relations were found in clear.
        """
        # Overwrite freed pages, so the plaintext names do not survive in the file
        # (VACUUM afterwards compacts it; the rollback journal is deleted on commit).
        await db.execute('PRAGMA secure_delete=ON')

        def unreadable(what, e):
            return KnowledgeGraphUnreadable(
                f"knowledge graph at {self.db_path}: {what} unreadable under the loaded key "
                f"({type(e).__name__}); not migrated, file left unchanged")

        async with db.execute('SELECT id, type, label_enc, nonce, props_enc, props_nonce, auto FROM kg_nodes '
                              'WHERE aad_v IS NULL ORDER BY rowid') as cur:
            nodes = await cur.fetchall()
        decoded = []
        for old_id, type_, label_enc, nonce, props_enc, props_nonce, auto in nodes:
            try:
                name = self.cipher.decrypt(nonce, label_enc).decode()
                props = self._dec_json(props_nonce, props_enc)
            except Exception as e:
                raise unreadable('node label', e) from e
            decoded.append((old_id, self._nid(name), type_, name, props, auto))
        mapping = {}
        plaintext = False
        for old_id, nid, *_ in decoded:
            mapping[old_id] = nid
            plaintext = plaintext or old_id != nid
            await db.execute('DELETE FROM kg_nodes WHERE id=?', (old_id,))
        for old_id, nid, type_, name, props, auto in decoded:
            # Two ids that normalise alike (NFC vs NFD, or a stray v1 row next to
            # its migrated node) become one node; the row already there, or else
            # the first-written, keeps its label.
            await self._put_node(db, nid, name, type_, props, auto=bool(auto))

        def remap(x):
            x = x or ''
            return mapping.get(x) or (x if _HMAC_ID.fullmatch(x) else self._nid(x))

        async with db.execute('SELECT id, src, dst, relation, props_enc, props_nonce, memory_id FROM kg_edges '
                              'WHERE aad_v IS NULL') as cur:
            edges = await cur.fetchall()
        for eid, src, dst, relation, props_enc, props_nonce, memory_col in edges:
            try:
                props = self._dec_json(props_nonce, props_enc)
            except Exception as e:
                raise unreadable('edge props', e) from e
            memory_id = props.pop('memory_id', None) or memory_col
            if relation is not None:
                props['relation'] = relation
                plaintext = True
            s, d = remap(src), remap(dst)
            n, ct = self._enc(json.dumps(props).encode(), _edge_aad(eid, s, d))
            await db.execute('UPDATE kg_edges SET src=?, dst=?, relation=NULL, props_enc=?, props_nonce=?, '
                             'memory_id=?, aad_v=? WHERE id=?', (s, d, ct, n, memory_id, AAD_VERSION, eid))
            if memory_id:
                for node in (s, d):
                    await db.execute('INSERT OR IGNORE INTO memory_entities (memory_id, node_id) VALUES (?, ?)',
                                     (memory_id, node))
        _log(f"upgraded {len(nodes)} nodes and {len(edges)} edges in {self.db_path} "
             f"to hashed ids and row-bound encryption")
        return plaintext

    async def _vacuum(self):
        self._vacuum_after_init = False
        try:
            async with self._connect() as db:
                await db.execute('VACUUM')
                # In WAL mode the pre-VACUUM pages live on in -wal until a checkpoint.
                await db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.Error as e:
            _log(f"VACUUM after migration failed ({e}); freed pages were zeroed but the file is not compacted")

    # --- names ------------------------------------------------------------

    def _remember(self, nid: str, name: str):
        self._names[nid] = name
        self._patterns.pop(nid, None)

    async def _names_for(self, db, ids: Iterable[str]) -> Dict[str, str]:
        """Display names for node ids, decrypting only the ones not seen yet."""
        ids = set(ids)
        missing = [i for i in ids if i not in self._names]
        for start in range(0, len(missing), 500):
            chunk = missing[start:start + 500]
            marks = ','.join('?' * len(chunk))
            async with db.execute(f'SELECT id, label_enc, nonce, aad_v FROM kg_nodes WHERE id IN ({marks})',
                                  chunk) as cur:
                async for nid, label_enc, nonce, aad_v in cur:
                    try:
                        self._remember(nid, self._label(nid, label_enc, nonce, aad_v))
                    except Exception as e:
                        _log(f"node {nid[:12]}: label unreadable ({type(e).__name__}); skipped")
        return {i: self._names[i] for i in ids if i in self._names}

    def _pattern(self, nid: str) -> re.Pattern:
        pat = self._patterns.get(nid)
        if pat is None:
            words = [re.escape(w) for w in _norm(self._names[nid]).split()]
            # Whole words only, so "nas" does not match inside "SkyNAS".
            pat = re.compile(r'(?<!\w)' + r'\s+'.join(words) + r'(?!\w)') if words else re.compile(r'(?!x)x')
            self._patterns[nid] = pat
        return pat

    # --- writes -----------------------------------------------------------

    async def add_memory_entities(self, mem_item):
        """Link a memory to its entities and connect the entities pairwise.

        Idempotent per memory, so rebuild_links() can replay it over a graph that
        already holds some of the links.
        """
        entities = []
        seen = set()
        for ent in mem_item.entities or []:
            if not isinstance(ent, str) or not ent.strip():
                continue
            nid = self._nid(ent)
            if nid not in seen:
                seen.add(nid)
                entities.append((nid, ent))
        if not entities:
            return
        async with self._connect() as db:
            for nid, ent in entities:
                await self._put_node(db, nid, ent, auto=True)
                await db.execute('INSERT OR IGNORE INTO memory_entities (memory_id, node_id) VALUES (?, ?)',
                                 (mem_item.id, nid))
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    src, dst = entities[i][0], entities[j][0]
                    async with db.execute('SELECT 1 FROM kg_edges WHERE memory_id=? AND src=? AND dst=?',
                                          (mem_item.id, src, dst)) as cur:
                        if await cur.fetchone():
                            continue
                    await self._put_edge(db, str(uuid.uuid4()), src, dst, {'relation': 'co_occurs'},
                                         1.0, mem_item.id)
            await db.commit()

    async def add_entity(self, entity: str, type: str = 'entity', props: Dict = None) -> str:
        nid = self._nid(entity)
        async with self._connect() as db:
            # REPLACE, and not auto: an entity named by hand is never garbage.
            await self._put_node(db, nid, entity, type, props, replace=True)
            await db.commit()
        self._remember(nid, entity)
        return _norm(entity)

    async def has_entity(self, entity: str) -> bool:
        async with self._connect() as db:
            async with db.execute('SELECT 1 FROM kg_nodes WHERE id=?', (self._nid(entity),)) as cur:
                return await cur.fetchone() is not None

    async def add_relation(self, src: str, dst: str, relation: str = 'related_to', weight: float = 1.0) -> str:
        eid = str(uuid.uuid4())
        ends = [(self._nid(src), src), (self._nid(dst), dst)]
        async with self._connect() as db:
            for nid, name in ends:
                # The caller's spelling, not the lowercased id: labels used to lose
                # their case when a relation created the node. OR IGNORE keeps a
                # label add_entity already set.
                await self._put_node(db, nid, name)
                # Named by hand now, whoever created it: remove_memory keeps it.
                await db.execute('UPDATE kg_nodes SET auto=NULL WHERE id=?', (nid,))
            await self._put_edge(db, eid, ends[0][0], ends[1][0], {'relation': relation}, weight, None)
            await db.commit()
        return eid

    async def remove_memory(self, memory_id: str) -> int:
        """Drop a memory's links and co-occurrence edges, and nodes left with nothing.

        A node is only removed if memory linking created it, this memory referenced
        it, and nothing else does. An entity added or related by hand stays, as
        does every node older than that distinction. Returns the number of rows
        removed.
        """
        removed = 0
        async with self._connect() as db:
            await db.execute('BEGIN IMMEDIATE')
            try:
                async with db.execute('SELECT node_id FROM memory_entities WHERE memory_id=? '
                                      'UNION SELECT src FROM kg_edges WHERE memory_id=? '
                                      'UNION SELECT dst FROM kg_edges WHERE memory_id=?',
                                      (memory_id, memory_id, memory_id)) as cur:
                    candidates = [r[0] async for r in cur]
                cur = await db.execute('DELETE FROM memory_entities WHERE memory_id=?', (memory_id,))
                removed += cur.rowcount
                cur = await db.execute('DELETE FROM kg_edges WHERE memory_id=?', (memory_id,))
                removed += cur.rowcount
                for nid in candidates:
                    cur = await db.execute(
                        'DELETE FROM kg_nodes WHERE id=? AND auto=1 '
                        'AND NOT EXISTS(SELECT 1 FROM memory_entities WHERE node_id=?) '
                        'AND NOT EXISTS(SELECT 1 FROM kg_edges WHERE src=? OR dst=?)', (nid, nid, nid, nid))
                    removed += cur.rowcount
                await db.commit()
            except BaseException:
                await db.rollback()
                raise
        return removed

    async def rebuild_links(self, items) -> int:
        """Replay add_memory_entities over stored memories; returns how many had entities.

        Fills in what a migrated v1 graph cannot know (memories with a single
        entity had no edge) and rebuilds a lost or deleted kg.db from the store.
        Entities added by hand without a memory cannot be recovered this way.
        """
        n = 0
        for item in items:
            if item.entities:
                await self.add_memory_entities(item)
                n += 1
        async with self._connect() as db:
            await self._set_meta(db, 'links_complete', '1')
            await db.commit()
        self.needs_link_backfill = False
        return n

    # --- reads ------------------------------------------------------------

    async def list_entities(self, limit: int = 100) -> List[Dict]:
        out = []
        async with self._connect() as db:
            async with db.execute('SELECT id, type, label_enc, nonce, aad_v FROM kg_nodes ORDER BY rowid LIMIT ?',
                                  (limit,)) as cur:
                async for nid, type_, label_enc, nonce, aad_v in cur:
                    try:
                        label = self._label(nid, label_enc, nonce, aad_v)
                    except Exception as e:
                        # One tampered or foreign row used to fail the whole list.
                        _log(f"node {nid[:12]}: label unreadable ({type(e).__name__}); skipped")
                        continue
                    self._remember(nid, label)
                    out.append({'id': _norm(label), 'type': type_, 'label': label})
        return out

    async def delete_entity(self, entity: str) -> Dict:
        nid = self._nid(entity)
        async with self._connect() as db:
            cur = await db.execute('DELETE FROM kg_nodes WHERE id=?', (nid,))
            nodes = cur.rowcount
            cur = await db.execute('DELETE FROM kg_edges WHERE src=? OR dst=?', (nid, nid))
            edges = cur.rowcount
            await db.execute('DELETE FROM memory_entities WHERE node_id=?', (nid,))
            await db.commit()
        self._names.pop(nid, None)
        self._patterns.pop(nid, None)
        return {'entity': _norm(entity), 'nodes_deleted': nodes, 'edges_deleted': edges}

    async def traverse(self, entity: str, depth: int = 2, limit: int = 20) -> List[Dict]:
        """Edges within `depth` hops of `entity`, each once, at most `limit` of them.

        depth=1 is the edges touching the entity itself. This used to expand the
        nodes at the last level too (depth+1 hops), return every edge once from
        each end, and overshoot the limit by a whole node's degree.
        """
        start = self._nid(entity)
        visited = {start}
        queue = deque([(start, 0)])
        seen_edges = set()
        raw = []
        async with self._connect() as db:
            while queue and len(raw) < limit:
                curr, d = queue.popleft()
                if d >= depth:
                    continue
                async with db.execute('SELECT id, src, dst, relation, weight, props_enc, props_nonce, aad_v '
                                      'FROM kg_edges WHERE src=? OR dst=? ORDER BY rowid', (curr, curr)) as cur:
                    rows = await cur.fetchall()
                for row in rows:
                    if row[0] in seen_edges:
                        continue
                    if len(raw) >= limit:
                        break
                    seen_edges.add(row[0])
                    raw.append(row)
                    other = row[2] if row[1] == curr else row[1]
                    if other not in visited:
                        visited.add(other)
                        queue.append((other, d + 1))
            names = await self._names_for(db, {n for r in raw for n in (r[1], r[2])})
        results = []
        for eid, src, dst, relation, weight, props_enc, props_nonce, aad_v in raw:
            try:
                relation = self._edge_props(eid, src, dst, props_enc, props_nonce, aad_v).get('relation', relation)
            except Exception as e:
                _log(f"edge {eid}: props unreadable ({type(e).__name__})")
            results.append({'src': _norm(names.get(src, src)), 'dst': _norm(names.get(dst, dst)),
                            'relation': relation, 'weight': weight})
        return results

    async def get_related_memories(self, query: str, limit: int = 20) -> List[str]:
        """Ids of memories linked to entities named in `query`, best first.

        Ranked by how many distinct matched entities a memory carries, then by how
        rare those entities are (one naming a handful of memories says more than
        one on every memory), then newest link first.
        """
        q = _norm(query or '')
        if not q.strip():
            return []
        async with self._connect() as db:
            async with db.execute('SELECT DISTINCT node_id FROM memory_entities') as cur:
                linked = [r[0] async for r in cur]
            names = await self._names_for(db, linked)
            matched = [nid for nid in names if self._pattern(nid).search(q)]
            if not matched:
                return []
            marks = ','.join('?' * len(matched))
            async with db.execute(f'SELECT memory_id, node_id, rowid FROM memory_entities WHERE node_id IN ({marks})',
                                  matched) as cur:
                links = await cur.fetchall()
        df: Dict[str, int] = {}
        for _, nid, _ in links:
            df[nid] = df.get(nid, 0) + 1
        score: Dict[str, list] = {}
        for mid, nid, rowid in links:
            s = score.setdefault(mid, [0, 0.0, 0])
            s[0] += 1
            s[1] += 1.0 / df[nid]
            s[2] = max(s[2], rowid)
        ranked = sorted(score.items(), key=lambda kv: (kv[1][0], kv[1][1], kv[1][2]), reverse=True)
        return [mid for mid, _ in ranked[:limit]]
