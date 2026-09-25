
import json, os, sqlite3, sys, time
from typing import Dict, List, Optional, Tuple
from ..core.tiers import MemoryItem, Tier
from ..core.ebbinghaus import ForgettingCurve
from ..crypto.aes_gcm import AES256GCM
from ..crypto.blind_index import BlindIndex, TOKENIZER_VERSION
from ..crypto.key_manager import MasterKeyMismatch
from cryptography.exceptions import InvalidTag
import aiosqlite
from pathlib import Path

BLIND_INDEX_INFO = b"blind-index-v1"
KEY_ID_INFO = b"memcore-key-id-v1"

# Every connection waits this long for a lock instead of sqlite3's 5 s default:
# the host store is opened by the MCP server, the CLI and the sync script at once.
BUSY_TIMEOUT_S = 10.0

# aad_v marks rows whose ciphertexts are bound to their row id and column. NULL
# means the row was written before that existed and decrypts without AAD; it is
# upgraded the next time the row is written, or by reencrypt_legacy_rows().
AAD_VERSION = 1

# How many existing rows init() tries before deciding the loaded key is foreign.
_KEY_PROBE_ROWS = 32


class CorruptRow(Exception):
    """One stored row cannot be turned back into a MemoryItem.

    Carries the id and the column, never plaintext or ciphertext bytes, so the
    message is safe to log and to return over an API.
    """

    def __init__(self, memory_id: str, column: str, cause: BaseException):
        self.memory_id = memory_id
        self.column = column
        self.cause = cause
        detail = str(cause)[:160]
        reason = type(cause).__name__ + (f": {detail}" if detail else "")
        if isinstance(cause, InvalidTag):
            reason = ("failed GCM authentication (tampered, truncated, moved from another "
                      "row, or written under a different key)")
        super().__init__(f"row {memory_id}: {column} unreadable ({reason})")


def _log(msg: str):
    # stderr only: stdout is the JSON-RPC stream under the stdio MCP server.
    print(f"[memcore] {msg}", file=sys.stderr)


class EncryptedStore:
    def __init__(self, db_path: Path, cipher: AES256GCM, blind_index: bool = None):
        self.db_path = db_path
        self.cipher = cipher
        if blind_index is None:
            from ..config import settings
            blind_index = getattr(settings, "blind_index_enabled", True)
        # Keyed separately from the content cipher - see AES256GCM.derive_subkey.
        self.blind = BlindIndex(cipher.derive_subkey(BLIND_INDEX_INFO)) if blind_index else None

    def _connect(self):
        return aiosqlite.connect(self.db_path, timeout=BUSY_TIMEOUT_S)

    @property
    def key_id(self) -> str:
        """Fingerprint of the content key, safe to store: an HMAC under it, not the key."""
        return self.cipher.derive_subkey(KEY_ID_INFO)[:16].hex()

    def _create_private(self):
        # Created here, 0600, rather than by sqlite3 with the umask default (0644
        # on most hosts): the plaintext columns - tier, timestamps, embeddings, the
        # blind index - were readable by every local account. SQLite gives -wal and
        # -shm the mode of the database file, so they follow. An existing file keeps
        # its mode; tightening it is left to the operator.
        if str(self.db_path) == ":memory:":
            return
        try:
            fd = os.open(self.db_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            return
        os.close(fd)

    async def init(self):
        self._create_private()
        async with self._connect() as db:
            # WAL lets readers and the writer proceed together; the rollback journal
            # made every list block writers from the other processes on this host.
            # The setting persists in the file. With a connection per call, the last
            # one to close checkpoints and removes -wal, so memory.db alone stays a
            # complete copy whenever the store is idle.
            await db.execute('PRAGMA journal_mode=WAL')
            await db.execute('''CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tier TEXT,
                timestamp REAL,
                content_enc BLOB,
                nonce BLOB,
                metadata_enc BLOB,
                meta_nonce BLOB,
                forgetting_json TEXT,
                entities_json TEXT,
                embedding BLOB,
                blind_index_json TEXT,
                aad_v INTEGER,
                entities_enc BLOB,
                entities_nonce BLOB
            )''')
            # Stores created before these columns existed. Their rows stay NULL: a
            # NULL blind index simply doesn't match until the row is rewritten or
            # rebuild_blind_index() runs, and a NULL aad_v decrypts without AAD.
            async with db.execute("PRAGMA table_info(memories)") as cur:
                columns = {row[1] async for row in cur}
            for name, decl in (("blind_index_json", "TEXT"), ("aad_v", "INTEGER"),
                               ("entities_enc", "BLOB"), ("entities_nonce", "BLOB")):
                if name not in columns:
                    await self._add_column(db, name, decl)
            await db.execute('CREATE TABLE IF NOT EXISTS store_meta (k TEXT PRIMARY KEY, v TEXT)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
            await db.commit()
            await self._check_key(db)
        # Only after _check_key: a migration run under the wrong key would write
        # rows that the right key can never read.
        await self._migrate_legacy_rows()

    @staticmethod
    async def _add_column(db, name: str, decl: str):
        try:
            await db.execute(f"ALTER TABLE memories ADD COLUMN {name} {decl}")
        except sqlite3.OperationalError as e:
            # Another process opening the same old store added it first.
            if "duplicate column" not in str(e):
                raise

    async def _meta(self, db, k: str) -> Optional[str]:
        async with db.execute('SELECT v FROM store_meta WHERE k=?', (k,)) as cur:
            row = await cur.fetchone()
        return row[0] if row else None

    async def _check_key(self, db):
        """Refuse to open a store under a key it was not written with.

        Nothing else would notice: opening reads no encrypted row, so a wrong key
        (a restored memory.db without its master.key, a typo in MNEM_KEY_PATH, 32
        random bytes) used to open cleanly and accept writes - rows that became
        unrecoverable the moment the right key was put back.
        """
        ours = self.key_id
        stored = await self._meta(db, 'key_id')
        async with db.execute('SELECT COUNT(*) FROM memories') as cur:
            count = (await cur.fetchone())[0]
        if stored is not None and stored != ours and count:
            raise MasterKeyMismatch(
                f"the loaded master key (id {ours}) is not the key {self.db_path} was written "
                f"under (id {stored}); refusing to open it. Restore the matching master.key."
            )
        if stored is None and count:
            # A store from before the fingerprint existed: stamp it only once the
            # key has proven itself on real rows, never on faith.
            if not await self._key_decrypts_any_row(db):
                raise MasterKeyMismatch(
                    f"the loaded master key (id {ours}) decrypts none of the first "
                    f"{_KEY_PROBE_ROWS} rows of {self.db_path}; refusing to open it. "
                    f"Restore the matching master.key."
                )
        if stored != ours:
            await db.execute('INSERT OR REPLACE INTO store_meta (k, v) VALUES (?, ?)', ('key_id', ours))
        if count == 0 and await self._meta(db, 'blind_index_v') is None:
            await db.execute('INSERT OR REPLACE INTO store_meta (k, v) VALUES (?, ?)',
                             ('blind_index_v', str(TOKENIZER_VERSION)))
        await db.commit()

    async def _key_decrypts_any_row(self, db) -> bool:
        db.row_factory = aiosqlite.Row
        async with db.execute('SELECT id, content_enc, nonce, aad_v FROM memories ORDER BY rowid '
                              'LIMIT ?', (_KEY_PROBE_ROWS,)) as cur:
            rows = await cur.fetchall()
        db.row_factory = None
        for row in rows:
            try:
                self._dec(row['nonce'], row['content_enc'], self._row_aad(row, 'content'))
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _aad(memory_id: str, column: str) -> bytes:
        # Binds a ciphertext to its row and column, so one row's content can't be
        # moved into another row (or into the metadata slot) and still decrypt.
        return f"memcore:v{AAD_VERSION}:{memory_id}:{column}".encode()

    def _row_aad(self, row, column: str) -> bytes:
        aad_v = row['aad_v'] if 'aad_v' in row.keys() else None
        return self._aad(row['id'], column) if aad_v else b""

    def _enc(self, plaintext: str, aad: bytes = b""):
        nonce, ct = self.cipher.encrypt(plaintext.encode(), aad)
        return nonce, ct

    def _dec(self, nonce: bytes, ct: bytes, aad: bytes = b"") -> str:
        return self.cipher.decrypt(nonce, ct, aad).decode()

    def _blind_index_json(self, item: MemoryItem) -> Optional[str]:
        if not self.blind:
            return None
        return json.dumps(self.blind.compute_index(item.content, item.metadata))

    def _encrypted_columns(self, item: MemoryItem):
        c_nonce, c_ct = self._enc(item.content, self._aad(item.id, 'content'))
        m_nonce, m_ct = self._enc(json.dumps(item.metadata), self._aad(item.id, 'metadata'))
        return c_nonce, c_ct, m_nonce, m_ct

    def _enc_entities(self, memory_id: str, entities_json: str):
        # entities_json used to be a plaintext column: every person, host and
        # project name was readable from memory.db without the key. Always bound
        # to the row (no pre-AAD variant exists for this column).
        return self._enc(entities_json, self._aad(memory_id, 'entities'))

    async def put(self, item: MemoryItem):
        """Insert or fully replace a row. For creation and import - not for updates.

        A full-row write of a snapshot read earlier undoes whatever happened in
        between (a delete, a promotion), so updates go through update_forgetting,
        update_content and update_tier, which touch only their own columns and
        cannot bring back a deleted row.
        """
        c_nonce, c_ct, m_nonce, m_ct = self._encrypted_columns(item)
        forgetting_json = json.dumps(item.forgetting.to_dict())
        e_nonce, e_ct = self._enc_entities(item.id, json.dumps(item.entities))
        emb_blob = json.dumps(item.embedding).encode() if item.embedding else None
        blind_json = self._blind_index_json(item)
        async with self._connect() as db:
            await db.execute('''INSERT OR REPLACE INTO memories
                (id, tier, timestamp, content_enc, nonce, metadata_enc, meta_nonce, forgetting_json, entities_json, embedding, blind_index_json, aad_v, entities_enc, entities_nonce)
                VALUES (?,?,?,?,?,?,?,?,NULL,?,?,?,?,?)''',
                (item.id, item.tier.value, item.timestamp, c_ct, c_nonce, m_ct, m_nonce, forgetting_json, emb_blob, blind_json, AAD_VERSION, e_ct, e_nonce))
            await db.commit()

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """One memory, or None. A row that exists but can't be read raises CorruptRow."""
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT * FROM memories WHERE id=?', (memory_id,)) as cur:
                row = await cur.fetchone()
        if not row:
            return None
        return self._row_to_item(row)

    async def get_many(self, ids: List[str]) -> Dict[str, MemoryItem]:
        """Readable memories among `ids`, on one connection. Missing or unreadable ids are absent."""
        wanted = list(dict.fromkeys(ids))
        rows = []
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            # Well under SQLite's bound-parameter limit on every build.
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                marks = ",".join("?" * len(chunk))
                async with db.execute(f'SELECT * FROM memories WHERE id IN ({marks})', chunk) as cur:
                    rows.extend(await cur.fetchall())
        out = {}
        for row in rows:
            try:
                out[row['id']] = self._row_to_item(row)
            except CorruptRow as e:
                _log(f"skipping {e}")
        return out

    def _row_to_item(self, row) -> MemoryItem:
        """Decode one row, raising CorruptRow that names the column that failed.

        Types are checked here rather than left to fail later: a forgetting_json
        that parses but holds a string where a number belongs used to pass this
        point and then break every retention() call, outside any guard.
        """
        mid = row['id']

        def fail(column, e):
            return CorruptRow(mid, column, e)

        try:
            content = self._dec(row['nonce'], row['content_enc'], self._row_aad(row, 'content'))
        except Exception as e:
            raise fail('content_enc', e) from None
        try:
            metadata = json.loads(self._dec(row['meta_nonce'], row['metadata_enc'],
                                            self._row_aad(row, 'metadata')))
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata is {type(metadata).__name__}, not an object")
        except Exception as e:
            raise fail('metadata_enc', e) from None
        try:
            raw = json.loads(row['forgetting_json']) if row['forgetting_json'] else {}
            if not isinstance(raw, dict):
                raise TypeError(f"forgetting is {type(raw).__name__}, not an object")
            forgetting = ForgettingCurve.from_dict(raw, default_last_access=row['timestamp'])
            forgetting.retention()
        except Exception as e:
            raise fail('forgetting_json', e) from None
        # A non-NULL entities_json can only come from code that predates
        # entities_enc (the current code always writes it NULL), so it is the newer
        # of the two whenever both are set.
        keys = row.keys()
        plain = row['entities_json']
        if plain is None and 'entities_enc' in keys and row['entities_enc'] is not None:
            column = 'entities_enc'
        else:
            column = 'entities_json'
        try:
            if column == 'entities_enc':
                raw_entities = self._dec(row['entities_nonce'], row['entities_enc'],
                                         self._aad(mid, 'entities'))
            else:
                raw_entities = plain
            entities = json.loads(raw_entities) if raw_entities else []
            if not isinstance(entities, list):
                raise TypeError(f"entities is {type(entities).__name__}, not a list")
        except Exception as e:
            raise fail(column, e) from None
        try:
            emb = json.loads(row['embedding'].decode()) if row['embedding'] else None
            if emb is not None and not (isinstance(emb, list) and all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in emb)):
                raise TypeError("embedding is not a list of numbers")
        except Exception as e:
            raise fail('embedding', e) from None
        try:
            tier = Tier(row['tier'])
        except Exception as e:
            raise fail('tier', e) from None
        try:
            return MemoryItem(
                id=mid,
                content=content,
                tier=tier,
                timestamp=row['timestamp'],
                embedding=emb,
                metadata=metadata,
                forgetting=forgetting,
                entities=entities
            )
        except Exception as e:
            raise fail('row', e) from None

    async def _scan(self, tier: Tier = None) -> Tuple[List[MemoryItem], Dict[str, CorruptRow]]:
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            if tier is None:
                sql, params = 'SELECT * FROM memories ORDER BY timestamp DESC', ()
            else:
                # Same order as the untiered list: callers slice [:limit], and in
                # rowid order that was the oldest rows, reshuffled by every rewrite.
                sql, params = 'SELECT * FROM memories WHERE tier=? ORDER BY timestamp DESC', (tier.value,)
            # Fetched first and decrypted after, so the read isn't held open for the
            # whole decrypt loop.
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
        items, unreadable = [], {}
        for row in rows:
            try:
                items.append(self._row_to_item(row))
            except CorruptRow as e:
                unreadable[e.memory_id] = e
        return items, unreadable

    async def scan(self, tier: Tier = None) -> Tuple[List[MemoryItem], Dict[str, str]]:
        """Every readable memory (optionally one tier), plus {id: reason} for the rest.

        The race-free way to learn what was skipped: the result belongs to this call,
        not to shared instance state that concurrent requests would overwrite.
        """
        items, unreadable = await self._scan(tier)
        return items, {mid: str(e) for mid, e in unreadable.items()}

    async def _list(self, tier: Optional[Tier], strict: bool) -> List[MemoryItem]:
        items, unreadable = await self._scan(tier)
        if unreadable:
            if strict:
                raise next(iter(unreadable.values()))
            # One bad row used to take down every list, health check, export and
            # both repair commands. Skip it, but loudly, so it can be found and fixed.
            for e in unreadable.values():
                _log(f"skipping {e}")
        return items

    async def list_all(self, strict: bool = False) -> List[MemoryItem]:
        """Readable memories, newest first. Unreadable rows are skipped and logged to
        stderr; use scan() to get them back, or strict=True to fail instead."""
        return await self._list(None, strict)

    async def list_by_tier(self, tier: Tier, strict: bool = False) -> List[MemoryItem]:
        return await self._list(tier, strict)

    async def verify(self) -> Dict:
        """Read-only integrity summary: how many rows decode, and why the others don't."""
        items, unreadable = await self.scan()
        return {"total": len(items) + len(unreadable), "readable": len(items),
                "unreadable": unreadable}

    async def delete(self, memory_id: str) -> bool:
        """Delete one row. True iff a row was actually removed."""
        async with self._connect() as db:
            cur = await db.execute('DELETE FROM memories WHERE id=?', (memory_id,))
            await db.commit()
            return cur.rowcount > 0

    async def update_forgetting(self, memory_id: str, forgetting: ForgettingCurve) -> bool:
        """Persist a rehearsal. Touches forgetting_json only and never inserts.

        Rehearsal used to write the whole row back with INSERT OR REPLACE, so a
        delete or promotion landing between the read and the write was undone: the
        deleted memory reappeared and the new tier reverted. False when the row is gone.
        """
        async with self._connect() as db:
            cur = await db.execute('UPDATE memories SET forgetting_json=? WHERE id=?',
                                   (json.dumps(forgetting.to_dict()), memory_id))
            await db.commit()
            return cur.rowcount > 0

    async def update_content(self, item: MemoryItem) -> bool:
        """Rewrite content, metadata, entities, embedding and blind index of an existing row.

        Leaves tier and forgetting alone and never inserts, so an edit racing a
        promotion or a delete can neither revert the one nor resurrect the other.
        """
        c_nonce, c_ct, m_nonce, m_ct = self._encrypted_columns(item)
        e_nonce, e_ct = self._enc_entities(item.id, json.dumps(item.entities))
        emb_blob = json.dumps(item.embedding).encode() if item.embedding else None
        async with self._connect() as db:
            cur = await db.execute(
                'UPDATE memories SET content_enc=?, nonce=?, metadata_enc=?, meta_nonce=?, '
                'entities_json=NULL, entities_enc=?, entities_nonce=?, embedding=?, '
                'blind_index_json=?, aad_v=? WHERE id=?',
                (c_ct, c_nonce, m_ct, m_nonce, e_ct, e_nonce, emb_blob,
                 self._blind_index_json(item), AAD_VERSION, item.id))
            await db.commit()
            return cur.rowcount > 0

    async def update_tier(self, memory_id: str, tier: Tier) -> bool:
        """Move a memory between tiers, raising its decay strength to the new floor.

        Strength is only ever raised, never lowered: a memory that earned a long
        retention through rehearsal keeps it even if it is demoted. False when the
        row is gone.
        """
        from ..core.tiers import TIER_BASE_STRENGTH
        async with self._connect() as db:
            # The read and the write must be one transaction, or a rehearsal from
            # another process between them is lost.
            await db.execute('BEGIN IMMEDIATE')
            try:
                async with db.execute('SELECT forgetting_json, timestamp FROM memories WHERE id=?',
                                      (memory_id,)) as cur:
                    row = await cur.fetchone()
                if row is None:
                    await db.rollback()
                    return False
                # The same parse as every read, so a curve that reads fine can also be
                # promoted: this used to work on the raw dict, where a NULL column
                # crashed json.loads and a string strength crashed max().
                try:
                    raw = json.loads(row[0]) if row[0] else {}
                    if not isinstance(raw, dict):
                        raise TypeError(f"forgetting is {type(raw).__name__}, not an object")
                    curve = ForgettingCurve.from_dict(raw, default_last_access=row[1] or 0.0)
                except Exception as e:
                    raise CorruptRow(memory_id, 'forgetting_json', e) from None
                curve.strength = max(curve.strength, TIER_BASE_STRENGTH.get(tier, 7.0))
                await db.execute('UPDATE memories SET tier=?, forgetting_json=? WHERE id=?',
                                 (tier.value, json.dumps(curve.to_dict()), memory_id))
                await db.commit()
                return True
            except BaseException:
                await db.rollback()
                raise

    async def search_by_content(self, query: str, limit: int = 50) -> List[MemoryItem]:
        all_items = await self.list_all()
        q = query.lower()
        return [i for i in all_items if q in i.content.lower()][:limit]

    async def search_by_blind_index(self, query: str, limit: int = 10) -> List[str]:
        """Find memory ids whose keyword HMACs overlap the query's, without decrypting.

        This is the point of the blind index: the server can answer a keyword search
        over ciphertext it cannot read. The tradeoff is that the HMACs sit in a
        plaintext column, so anyone with the database file learns which rows share
        keywords - but not what those keywords are, since they are HMACed under a
        key derived separately from the content key.

        Returns ids ranked by how many query terms matched, best first.
        """
        if not self.blind:
            return []
        matches = []
        async with self._connect() as db:
            version = await self._meta(db, 'blind_index_v')
            # Until rebuild_blind_index() has run, some rows still carry tokens from
            # the ASCII-only tokenizer; query with those too so they keep matching.
            legacy = version is None or int(version) < TOKENIZER_VERSION
            wanted = set(self.blind.search_query_hmacs(query, include_legacy=legacy))
            if not wanted:
                return []
            async with db.execute(
                'SELECT id, blind_index_json FROM memories WHERE blind_index_json IS NOT NULL'
            ) as cur:
                async for row in cur:
                    try:
                        overlap = len(wanted & set(json.loads(row[1])))
                    except (ValueError, TypeError):
                        continue
                    if overlap:
                        matches.append((overlap, row[0]))
        matches.sort(key=lambda m: m[0], reverse=True)
        return [memory_id for _, memory_id in matches[:limit]]

    async def rebuild_blind_index(self) -> int:
        """Recompute the blind index for every row, returning how many were written.

        Needed for rows stored before the index worked at all, and after any change
        to the tokenizer - a stale index doesn't error, it just quietly stops
        matching. Requires the key, so it runs from a process that has the store open.
        """
        if not self.blind:
            return 0
        items, unreadable = await self.scan()
        for reason in unreadable.values():
            _log(f"not reindexed: {reason}")
        async with self._connect() as db:
            for item in items:
                await db.execute(
                    'UPDATE memories SET blind_index_json=? WHERE id=?',
                    (self._blind_index_json(item), item.id),
                )
            if not unreadable:
                await db.execute('INSERT OR REPLACE INTO store_meta (k, v) VALUES (?, ?)',
                                 ('blind_index_v', str(TOKENIZER_VERSION)))
            await db.commit()
        return len(items)

    async def _migrate_legacy_rows(self) -> int:
        """Encrypt entity lists that earlier code wrote in clear; returns rows migrated.

        Guarded per row, so processes opening the store together cannot undo each
        other, and a row whose list doesn't parse is logged and left as it was.
        This makes starting the previous release on the same store lossy: it
        reads a migrated row's entities as [], and since it writes the whole row
        back on every get or recall, it then stores that [] for good. Roll back by
        restoring the backup taken before the upgrade (deploy-skynas.sh takes one
        on every run), never by starting the old image on a migrated store.
        Rebinding pre-AAD ciphertexts (reencrypt_legacy_rows) goes further - the
        old release cannot decrypt those rows at all - so it stays an explicit step.
        """
        async with self._connect() as db:
            async with db.execute('SELECT COUNT(*) FROM memories WHERE entities_json IS NOT NULL') as cur:
                pending = (await cur.fetchone())[0]
        if not pending:
            return 0
        done = await self._encrypt_plain_entities()
        if done:
            await self._scrub()
        return done

    async def _encrypt_plain_entities(self) -> int:
        async with self._connect() as db:
            async with db.execute('SELECT id, entities_json FROM memories '
                                  'WHERE entities_json IS NOT NULL') as cur:
                rows = await cur.fetchall()
        done = 0
        async with self._connect() as db:
            # Zero the old cells as they are rewritten, rather than leaving the
            # names on free space inside the page.
            await db.execute('PRAGMA secure_delete=ON')
            for mid, plain in rows:
                try:
                    if not isinstance(json.loads(plain), list):
                        raise TypeError
                except (ValueError, TypeError):
                    _log(f"row {mid}: entities_json is not a JSON list; left in place, unencrypted")
                    continue
                nonce, ct = self._enc_entities(mid, plain)
                # Guarded on the value read, so an update that landed since wins.
                cur = await db.execute(
                    'UPDATE memories SET entities_enc=?, entities_nonce=?, entities_json=NULL '
                    'WHERE id=? AND entities_json=?', (ct, nonce, mid, plain))
                done += cur.rowcount
            await db.commit()
        return done

    async def _scrub(self):
        """Drop plaintext left behind by a migration: freed pages and old -wal frames.

        Best effort - the data is already migrated, and a busy store only means the
        remnants survive until the next VACUUM.
        """
        try:
            async with self._connect() as db:
                await db.execute('VACUUM')
                await db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except Exception as e:
            _log(f"migration done, but VACUUM of {self.db_path} failed ({type(e).__name__}: {e}); "
                 f"old plaintext entity names may remain in free pages until the next VACUUM")

    async def reencrypt_legacy_rows(self) -> int:
        """Bind every pre-AAD row's ciphertexts to its id, returning how many were upgraded.

        Legacy rows are otherwise upgraded only when next written; until then their
        ciphertexts can still be swapped between rows undetected.

        One-way: the release before AAD fails GCM on every upgraded row, and its
        list_all has no per-row guard, so after this (or after any write by this
        code) roll back by restoring a backup of the store, never by starting the
        old image on it.
        """
        async with self._connect() as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT * FROM memories WHERE aad_v IS NULL') as cur:
                rows = await cur.fetchall()
        done = 0
        for row in rows:
            try:
                item = self._row_to_item(row)
            except CorruptRow as e:
                _log(f"not re-encrypted: {e}")
                continue
            c_nonce, c_ct, m_nonce, m_ct = self._encrypted_columns(item)
            async with self._connect() as db:
                # Guarded on aad_v IS NULL so a concurrent write that already
                # upgraded the row is not overwritten with this older snapshot.
                cur = await db.execute(
                    'UPDATE memories SET content_enc=?, nonce=?, metadata_enc=?, meta_nonce=?, '
                    'aad_v=? WHERE id=? AND aad_v IS NULL AND content_enc=? AND metadata_enc=?',
                    (c_ct, c_nonce, m_ct, m_nonce, AAD_VERSION, item.id,
                     row['content_enc'], row['metadata_enc']))
                await db.commit()
                done += cur.rowcount
        return done
