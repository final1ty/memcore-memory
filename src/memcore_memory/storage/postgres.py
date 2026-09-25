"""
Postgres + pgvector backend - production alternative to SQLite + HNSW

Encryption matches the SQLite store: content, metadata and entities are
AES-GCM encrypted and bound to their row and column (associated data
memcore:v1:{id}:{column}), so a ciphertext moved to another row fails
authentication. Rows written before that (aad_v NULL) still decrypt without it
and are rebound when rewritten, or by reencrypt_legacy_rows(). The blind index
gives keyword search over ciphertext. Plaintext columns, as in SQLite: id, tier,
timestamp, forgetting_json (which holds importance, so the importance column
adds nothing to it), the embedding and the blind-index HMACs.

Only memories and their vectors live in Postgres. The knowledge graph stays in a
local SQLite file under data_dir, so several instances sharing one database each
keep their own graph.

Not exercised against a real server in this repository's test suite (no
Postgres, no pgvector installed): the ORM paths are tested on SQLite with a
stand-in vector type, the Postgres-only DDL in init() is not.

Requires:
  CREATE EXTENSION IF NOT EXISTS vector;

Usage:
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mnemosyne
  storage = PostgresStore(dsn, cipher, embedding_dim=384)
  await storage.init()
"""

import sys
import json
from typing import Dict, List, Optional, Tuple
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, Integer, Text, LargeBinary, Index, func, select, delete, update, text
from pgvector.sqlalchemy import Vector

from ..core.tiers import MemoryItem, Tier, TIER_BASE_STRENGTH
from ..core.ebbinghaus import ForgettingCurve
from ..crypto.aes_gcm import AES256GCM
from ..crypto.blind_index import BlindIndex
from .encrypted_sqlite import AAD_VERSION, BLIND_INDEX_INFO, CorruptRow


class EmbeddingDimMismatch(RuntimeError):
    """The memories.embedding column holds vectors of another width."""


def _log(msg: str):
    # stderr only: stdout is the JSON-RPC stream under the stdio MCP server.
    print(f"[postgres] {msg}", file=sys.stderr)


def _build_models(dim: int):
    """A model bound to one embedding width.

    The column used to be a module-level Vector(768) that nothing ever altered,
    while the default embedder produces 384: every put failed. Per instance, on
    its own metadata, so two stores of different widths can't redefine each other.
    """
    Base = declarative_base()

    class MemoryRow(Base):
        __tablename__ = "memories"
        id = Column(String, primary_key=True)
        tier = Column(String, index=True)
        timestamp = Column(Float, index=True)
        content_enc = Column(LargeBinary, nullable=False)
        nonce = Column(LargeBinary, nullable=False)
        metadata_enc = Column(LargeBinary, nullable=False)
        meta_nonce = Column(LargeBinary, nullable=False)
        forgetting_json = Column(Text)
        # Plaintext entity list, written by versions before entities_enc. Read
        # as a fallback, emptied by init().
        entities_json = Column(Text)
        entities_enc = Column(LargeBinary)
        entities_nonce = Column(LargeBinary)
        embedding = Column(Vector(dim))
        # Intentionally plaintext, for filtering: forgetting_json carries it too.
        importance = Column(Float, default=0.5)
        blind_index_json = Column(Text)
        # 1: content and metadata are bound to the row (AAD). NULL: written
        # before that, decrypted without AAD.
        aad_v = Column(Integer)

        # No ivfflat index: built on an empty table its lists are untrained. The
        # HNSW index init() creates needs no training data.
        __table_args__ = (
            Index("idx_memories_tier_timestamp", "tier", "timestamp"),
        )

    return Base, MemoryRow


# Columns added after the first release; create_all never adds a column to an
# existing table.
_LATER_COLUMNS = {"entities_enc": LargeBinary(), "entities_nonce": LargeBinary(),
                  "blind_index_json": Text(), "aad_v": Integer()}


class PostgresStore:
    def __init__(self, dsn: str, cipher: AES256GCM, embedding_dim: int = 768, blind_index: bool = None):
        # dsn e.g. postgresql+asyncpg://user:pass@host/db
        self.dsn = dsn
        self.cipher = cipher
        self.dim = int(embedding_dim)
        self.Base, self.MemoryRow = _build_models(self.dim)
        if blind_index is None:
            from ..config import settings
            blind_index = getattr(settings, "blind_index_enabled", True)
        self.blind = BlindIndex(cipher.derive_subkey(BLIND_INDEX_INFO)) if blind_index else None
        pool = {} if dsn.startswith("sqlite") else {"pool_size": 10, "max_overflow": 20}
        self.engine = create_async_engine(dsn, echo=False, **pool)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self):
        async with self.engine.begin() as conn:
            pg = conn.dialect.name == "postgresql"
            if pg:
                try:
                    # A savepoint: a failed statement aborts a Postgres transaction,
                    # so without one everything after this failed too.
                    async with conn.begin_nested():
                        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                except Exception as e:
                    _log(f"could not create the vector extension (needs superuser): {e}")
            await conn.run_sync(self.Base.metadata.create_all)
            await self._add_later_columns(conn)
            if pg:
                await self._check_embedding_dim(conn)
        await self._encrypt_legacy_entities()
        await self._backfill_blind_index()
        if self.engine.dialect.name == "postgresql":
            await self._ensure_hnsw()

    async def _add_later_columns(self, conn):
        def existing(sync_conn):
            from sqlalchemy import inspect
            return {c["name"] for c in inspect(sync_conn).get_columns("memories")}
        have = await conn.run_sync(existing)
        for name, type_ in _LATER_COLUMNS.items():
            if name not in have:
                await conn.execute(text(
                    f"ALTER TABLE memories ADD COLUMN {name} {type_.compile(dialect=conn.dialect)}"))

    async def _check_embedding_dim(self, conn):
        # For pgvector the column's typmod is its dimension (-1 when untyped).
        typmod = (await conn.execute(text(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'memories'::regclass AND attname = 'embedding'"))).scalar()
        if typmod == self.dim:
            return
        stored = (await conn.execute(text(
            "SELECT count(*) FROM memories WHERE embedding IS NOT NULL"))).scalar()
        if stored:
            # Never re-dimension populated vectors: pgvector can't cast between
            # widths, and the old vectors would be meaningless under a new model.
            raise EmbeddingDimMismatch(
                f"memories.embedding is vector({typmod}) and holds {stored} vectors, but the "
                f"embedder produces {self.dim}; re-embed into another database or set "
                f"MNEM_EMBEDDING_DIM={typmod}")
        # Empty: a table created by the old hard-coded vector(768), which could
        # never have stored a 384-dim row.
        _log(f"memories.embedding re-typed from vector({typmod}) to vector({self.dim}) (no vectors stored)")
        await conn.execute(text(f"ALTER TABLE memories ALTER COLUMN embedding TYPE vector({self.dim})"))

    async def _ensure_hnsw(self):
        # The ivfflat index earlier versions built on an empty table has untrained
        # lists, which is worse than no index. Dropped on its own, before the HNSW
        # attempt: in one transaction, a pgvector without HNSW kept it alive.
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text("DROP INDEX IF EXISTS idx_memories_embedding"))
        except Exception as e:
            _log(f"could not drop the old ivfflat index idx_memories_embedding ({e})")
        try:
            async with self.engine.begin() as conn:
                await conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw "
                    "ON memories USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"))
        except Exception as e:
            # pgvector < 0.5 has no HNSW: exact scans still work, just slower.
            _log(f"no HNSW index on memories.embedding ({e}); vector search will scan")

    async def _backfill_blind_index(self):
        """Index rows written before blind_index_json existed.

        They held NULL, which search_by_blind_index cannot match, so they were
        invisible to encrypted search until someone ran rebuild_blind_index by
        hand. Only NULL rows: a row whose index is already there is not re-read.
        """
        if not self.blind:
            return
        M = self.MemoryRow
        async with self.Session() as sess:
            pending = (await sess.execute(
                select(func.count()).select_from(M).where(M.blind_index_json.is_(None)))).scalar()
        if not pending:
            return
        done, skipped = 0, 0
        async with self.Session() as sess:
            async with sess.begin():
                rows = (await sess.execute(select(M).where(M.blind_index_json.is_(None)))).scalars().all()
                for row in rows:
                    try:
                        item = self._row_to_item(row)
                    except CorruptRow as e:
                        skipped += 1
                        _log(f"not indexed: {e}")
                        continue
                    row.blind_index_json = self._blind_index_json(item)
                    done += 1
        _log(f"blind index written for {done} rows that had none"
             + (f"; {skipped} unreadable rows still have none" if skipped else ""))

    async def _encrypt_legacy_entities(self):
        """Move entity lists written in plaintext into entities_enc."""
        M = self.MemoryRow
        async with self.Session() as sess:
            async with sess.begin():
                rows = (await sess.execute(
                    select(M.id, M.entities_json).where(M.entities_json.is_not(None)))).all()
                for mid, raw in rows:
                    try:
                        entities = json.loads(raw)
                    except ValueError:
                        _log(f"row {mid}: entities_json is not JSON; left as it is")
                        continue
                    nonce, ct = self._enc_entities(mid, entities)
                    await sess.execute(update(M).where(M.id == mid).values(
                        entities_enc=ct, entities_nonce=nonce, entities_json=None))
        if rows:
            _log(f"encrypted the entity lists of {len(rows)} rows; VACUUM to drop the old plaintext")

    @staticmethod
    def _aad(memory_id: str, column: str) -> bytes:
        return f"memcore:v1:{memory_id}:{column}".encode()

    def _row_aad(self, row, column: str) -> bytes:
        return self._aad(row.id, column) if getattr(row, "aad_v", None) else b""

    def _enc(self, plaintext: str, aad: bytes = b""):
        nonce, ct = self.cipher.encrypt(plaintext.encode(), aad)
        return nonce, ct

    def _dec(self, nonce: bytes, ct: bytes, aad: bytes = b"") -> str:
        return self.cipher.decrypt(nonce, ct, aad).decode()

    def _enc_entities(self, memory_id: str, entities) -> Tuple[bytes, bytes]:
        return self.cipher.encrypt(json.dumps(entities or []).encode(), self._aad(memory_id, "entities"))

    def _blind_index_json(self, item: MemoryItem) -> Optional[str]:
        if not self.blind:
            return None
        return json.dumps(self.blind.compute_index(item.content, item.metadata))

    def _row_to_item(self, row) -> MemoryItem:
        """Decode one row, raising CorruptRow that names the column that failed,
        with the same checks as EncryptedStore._row_to_item."""
        mid = row.id
        try:
            content = self._dec(row.nonce, row.content_enc, self._row_aad(row, "content"))
        except Exception as e:
            raise CorruptRow(mid, "content_enc", e) from None
        try:
            metadata = json.loads(self._dec(row.meta_nonce, row.metadata_enc, self._row_aad(row, "metadata")))
            if not isinstance(metadata, dict):
                raise TypeError(f"metadata is {type(metadata).__name__}, not an object")
        except Exception as e:
            raise CorruptRow(mid, "metadata_enc", e) from None
        try:
            forgetting = self._curve(row.forgetting_json, row.timestamp)
            forgetting.retention()
        except Exception as e:
            raise CorruptRow(mid, "forgetting_json", e) from None
        column = "entities_enc" if row.entities_enc is not None else "entities_json"
        try:
            if column == "entities_enc":
                entities = json.loads(self.cipher.decrypt(
                    row.entities_nonce, row.entities_enc, self._aad(mid, "entities")).decode())
            else:
                entities = json.loads(row.entities_json) if row.entities_json else []
            if not isinstance(entities, list):
                raise TypeError(f"entities is {type(entities).__name__}, not a list")
        except Exception as e:
            raise CorruptRow(mid, column, e) from None
        try:
            emb = row.embedding.tolist() if hasattr(row.embedding, 'tolist') else row.embedding
            if emb is not None:
                emb = [float(x) for x in emb]
        except Exception as e:
            raise CorruptRow(mid, "embedding", e) from None
        try:
            tier = Tier(row.tier)
        except Exception as e:
            raise CorruptRow(mid, "tier", e) from None
        try:
            return MemoryItem(
                id=mid,
                content=content,
                tier=tier,
                timestamp=row.timestamp,
                embedding=emb,
                metadata=metadata,
                forgetting=forgetting,
                entities=entities
            )
        except Exception as e:
            raise CorruptRow(mid, "row", e) from None

    @staticmethod
    def _curve(raw_json, timestamp) -> ForgettingCurve:
        raw = json.loads(raw_json) if raw_json else {}
        if not isinstance(raw, dict):
            raise TypeError(f"forgetting is {type(raw).__name__}, not an object")
        return ForgettingCurve.from_dict(raw, default_last_access=timestamp or 0.0)

    def _sealed(self, item: MemoryItem) -> dict:
        c_nonce, c_ct = self._enc(item.content, self._aad(item.id, "content"))
        m_nonce, m_ct = self._enc(json.dumps(item.metadata), self._aad(item.id, "metadata"))
        return dict(content_enc=c_ct, nonce=c_nonce, metadata_enc=m_ct, meta_nonce=m_nonce,
                    aad_v=AAD_VERSION)

    def _content_values(self, item: MemoryItem) -> dict:
        e_nonce, e_ct = self._enc_entities(item.id, item.entities)
        return dict(self._sealed(item),
                    entities_enc=e_ct, entities_nonce=e_nonce, entities_json=None,
                    embedding=item.embedding, importance=item.metadata.get("importance", 0.5),
                    blind_index_json=self._blind_index_json(item))

    async def put(self, item: MemoryItem):
        """Insert or fully replace a row. For creation and import - not for updates."""
        values = dict(self._content_values(item), tier=item.tier.value, timestamp=item.timestamp,
                      forgetting_json=json.dumps(item.forgetting.to_dict()))
        async with self.Session() as sess:
            async with sess.begin():
                existing = await sess.get(self.MemoryRow, item.id)
                if existing:
                    for k, v in values.items():
                        setattr(existing, k, v)
                else:
                    sess.add(self.MemoryRow(id=item.id, **values))

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        """One memory, or None. A row that exists but can't be read raises CorruptRow."""
        async with self.Session() as sess:
            row = await sess.get(self.MemoryRow, memory_id)
            if not row:
                return None
            return self._row_to_item(row)

    async def get_many(self, ids: List[str]) -> Dict[str, MemoryItem]:
        """Readable memories among `ids`, in one session. Missing or unreadable ids are absent."""
        wanted = list(dict.fromkeys(ids))
        out = {}
        async with self.Session() as sess:
            for start in range(0, len(wanted), 500):
                chunk = wanted[start:start + 500]
                rows = (await sess.execute(select(self.MemoryRow).where(self.MemoryRow.id.in_(chunk)))).scalars().all()
                for row in rows:
                    try:
                        out[row.id] = self._row_to_item(row)
                    except CorruptRow as e:
                        _log(f"skipping {e}")
        return out

    async def _scan(self, tier: Tier = None) -> Tuple[List[MemoryItem], Dict[str, CorruptRow]]:
        M = self.MemoryRow
        stmt = select(M).order_by(M.timestamp.desc())
        if tier is not None:
            stmt = stmt.where(M.tier == tier.value)
        async with self.Session() as sess:
            rows = (await sess.execute(stmt)).scalars().all()
        items, unreadable = [], {}
        for r in rows:
            try:
                items.append(self._row_to_item(r))
            except CorruptRow as e:
                unreadable[r.id] = e
        return items, unreadable

    async def scan(self, tier: Tier = None) -> Tuple[List[MemoryItem], Dict[str, str]]:
        """Every readable memory (optionally one tier), plus {id: reason} for the rest."""
        items, unreadable = await self._scan(tier)
        return items, {mid: str(e) for mid, e in unreadable.items()}

    async def _list(self, tier: Optional[Tier], strict: bool) -> List[MemoryItem]:
        items, unreadable = await self._scan(tier)
        if unreadable and strict:
            raise next(iter(unreadable.values()))
        for e in unreadable.values():
            _log(f"skipping {e}")
        return items

    async def list_all(self, strict: bool = False) -> List[MemoryItem]:
        """Readable memories, newest first; the same contract as EncryptedStore.list_all."""
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
        async with self.Session() as sess:
            async with sess.begin():
                result = await sess.execute(delete(self.MemoryRow).where(self.MemoryRow.id == memory_id))
                return result.rowcount > 0

    async def update_forgetting(self, memory_id: str, forgetting: ForgettingCurve) -> bool:
        """Persist a rehearsal. Touches forgetting_json only and never inserts,
        so a delete landing between the read and this write stays a delete."""
        async with self.Session() as sess:
            async with sess.begin():
                result = await sess.execute(
                    update(self.MemoryRow).where(self.MemoryRow.id == memory_id)
                    .values(forgetting_json=json.dumps(forgetting.to_dict())))
                return result.rowcount > 0

    async def update_content(self, item: MemoryItem) -> bool:
        """Rewrite content, metadata, entities, embedding and blind index of an existing row.
        Leaves tier and forgetting alone and never inserts."""
        async with self.Session() as sess:
            async with sess.begin():
                result = await sess.execute(
                    update(self.MemoryRow).where(self.MemoryRow.id == item.id)
                    .values(**self._content_values(item)))
                return result.rowcount > 0

    async def update_tier(self, memory_id: str, tier: Tier) -> bool:
        """Move a memory between tiers, raising its decay strength to the new floor.

        Only the tier used to change here. Since loading stopped re-seeding strength
        from the tier, that left a promoted working memory decaying on its 20-minute
        schedule inside episodic. Never lowered, so a demotion keeps what rehearsal
        earned. False when the row is gone.
        """
        async with self.Session() as sess:
            async with sess.begin():
                # Locked, so a concurrent rehearsal can't land between read and write.
                row = await sess.get(self.MemoryRow, memory_id, with_for_update=True)
                if row is None:
                    return False
                # The same parse as every read: on the raw dict a NULL column or a
                # string strength crashed here.
                try:
                    curve = self._curve(row.forgetting_json, row.timestamp)
                except Exception as e:
                    raise CorruptRow(memory_id, "forgetting_json", e) from None
                curve.strength = max(curve.strength, TIER_BASE_STRENGTH.get(tier, 7.0))
                row.tier = tier.value
                row.forgetting_json = json.dumps(curve.to_dict())
                return True

    # Vector search using pgvector cosine distance
    async def vector_search(self, query_embedding: List[float], k: int = 10, tier_filter: List[str] = None):
        M = self.MemoryRow
        async with self.Session() as sess:
            # cosine distance: 1 - cosine_sim, lower is better, so order by distance
            distance = M.embedding.cosine_distance(query_embedding).label("distance")
            stmt = select(M.id, distance).where(M.embedding.is_not(None))
            if tier_filter:
                stmt = stmt.where(M.tier.in_(tier_filter))
            stmt = stmt.order_by(distance).limit(k)
            rows = (await sess.execute(stmt)).all()
            return [(mid, 1 - float(dist)) for mid, dist in rows]

    async def search_by_content(self, query: str, limit: int = 50) -> List[MemoryItem]:
        # Decrypts every row; search_by_blind_index is the one that scales.
        all_items = await self.list_all()
        q = query.lower()
        return [i for i in all_items if q in i.content.lower()][:limit]

    async def search_by_blind_index(self, query: str, limit: int = 10) -> List[str]:
        """Memory ids whose keyword HMACs overlap the query's, best first, without
        decrypting. Same index and tradeoff as EncryptedStore.search_by_blind_index."""
        if not self.blind:
            return []
        # Every Postgres index was written by the current tokenizer: the column is
        # newer than the unicode change, so no legacy tokens to query with.
        wanted = set(self.blind.search_query_hmacs(query))
        if not wanted:
            return []
        M = self.MemoryRow
        matches = []
        async with self.Session() as sess:
            rows = (await sess.execute(
                select(M.id, M.blind_index_json).where(M.blind_index_json.is_not(None)))).all()
        for mid, raw in rows:
            try:
                overlap = len(wanted & set(json.loads(raw)))
            except (ValueError, TypeError):
                continue
            if overlap:
                matches.append((overlap, mid))
        matches.sort(key=lambda m: m[0], reverse=True)
        return [mid for _, mid in matches[:limit]]

    async def rebuild_blind_index(self) -> int:
        """Recompute the blind index for every readable row; returns how many were written."""
        if not self.blind:
            return 0
        items, unreadable = await self.scan()
        for reason in unreadable.values():
            _log(f"not reindexed: {reason}")
        async with self.Session() as sess:
            async with sess.begin():
                for item in items:
                    await sess.execute(update(self.MemoryRow).where(self.MemoryRow.id == item.id)
                                       .values(blind_index_json=self._blind_index_json(item)))
        return len(items)

    async def reencrypt_legacy_rows(self) -> int:
        """Bind every pre-AAD row's content and metadata to its id; returns how many.

        Same one-way step as EncryptedStore.reencrypt_legacy_rows: the release
        before AAD cannot read an upgraded row, so roll back by restoring a backup.
        """
        M = self.MemoryRow
        async with self.Session() as sess:
            rows = (await sess.execute(select(M).where(M.aad_v.is_(None)))).scalars().all()
        done = 0
        for row in rows:
            try:
                item = self._row_to_item(row)
            except CorruptRow as e:
                _log(f"not re-encrypted: {e}")
                continue
            async with self.Session() as sess:
                async with sess.begin():
                    # Guarded on the ciphertext read, so a write since then wins.
                    result = await sess.execute(
                        update(M).where(M.id == item.id, M.aad_v.is_(None),
                                        M.content_enc == row.content_enc, M.metadata_enc == row.metadata_enc)
                        .values(**self._sealed(item)))
                    done += result.rowcount
        return done
