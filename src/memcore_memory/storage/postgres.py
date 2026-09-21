
"""
Postgres + pgvector backend - production alternative to SQLite + HNSW

Keeps same encryption: content_enc + nonce, metadata_enc + meta_nonce
Embedding stored as pgvector vector type for ANN search (IVFFlat / HNSW)

Requires:
  CREATE EXTENSION IF NOT EXISTS vector;
  CREATE EXTENSION IF NOT EXISTS pgcrypto;

Usage:
  DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/mnemosyne
  storage = PostgresStore(dsn, cipher)
  await storage.init()
"""

import sys
import json, time, uuid
from typing import List, Optional
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, String, Float, Text, LargeBinary, Index, select, delete
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

from ..core.tiers import MemoryItem, Tier
from ..core.ebbinghaus import ForgettingCurve
from ..crypto.aes_gcm import AES256GCM

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
    entities_json = Column(Text)
    embedding = Column(Vector(768))  # will be altered if dim !=768
    # optional: for faster filtering
    importance = Column(Float, default=0.5)

    __table_args__ = (
        Index("idx_memories_tier_timestamp", "tier", "timestamp"),
        Index("idx_memories_embedding", "embedding", postgresql_using="ivfflat", postgresql_with={"lists": 100}, postgresql_ops={"embedding": "vector_cosine_ops"}),
    )

class KGNodeRow(Base):
    __tablename__ = "kg_nodes"
    id = Column(String, primary_key=True)
    type = Column(String)
    label_enc = Column(LargeBinary)
    nonce = Column(LargeBinary)
    props_enc = Column(LargeBinary)
    props_nonce = Column(LargeBinary)

class KGEdgeRow(Base):
    __tablename__ = "kg_edges"
    id = Column(String, primary_key=True)
    src = Column(String, index=True)
    dst = Column(String, index=True)
    relation = Column(String)
    weight = Column(Float)
    timestamp = Column(Float)
    props_enc = Column(LargeBinary)
    props_nonce = Column(LargeBinary)

class PostgresStore:
    def __init__(self, dsn: str, cipher: AES256GCM, embedding_dim: int = 768):
        # dsn e.g. postgresql+asyncpg://user:pass@host/db
        self.dsn = dsn
        self.cipher = cipher
        self.dim = embedding_dim
        self.engine = create_async_engine(dsn, echo=False, pool_size=10, max_overflow=20)
        self.Session = sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self):
        # For pgvector, we need to adjust vector dim if not 768
        # Create extensions
        async with self.engine.begin() as conn:
            await conn.execute(select(1))  # test
            # Extensions need to be created via sync - use raw
            try:
                from sqlalchemy import text
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
            except Exception as e:
                print(f"[postgres] Could not create extensions (need superuser): {e}", file=sys.stderr)
            await conn.run_sync(Base.metadata.create_all)

        # Try create HNSW index if pgvector >=0.5
        try:
            async with self.engine.begin() as conn:
                from sqlalchemy import text
                await conn.execute(text(f"""
                CREATE INDEX IF NOT EXISTS idx_memories_embedding_hnsw 
                ON memories USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)
                """))
        except Exception:
            pass  # fallback to ivfflat

    def _enc(self, plaintext: str):
        nonce, ct = self.cipher.encrypt(plaintext.encode())
        return nonce, ct

    def _dec(self, nonce: bytes, ct: bytes) -> str:
        return self.cipher.decrypt(nonce, ct).decode()

    def _row_to_item(self, row: MemoryRow) -> MemoryItem:
        content = self._dec(row.nonce, row.content_enc)
        metadata = json.loads(self._dec(row.meta_nonce, row.metadata_enc))
        f = json.loads(row.forgetting_json) if row.forgetting_json else {}
        forgetting = ForgettingCurve(
            strength=f.get("strength", 1.0),
            last_access=f.get("last_access", row.timestamp),
            rehearsals=f.get("rehearsals", 0),
            importance=f.get("importance", 0.5)
        )
        entities = json.loads(row.entities_json) if row.entities_json else []
        # embedding is already list
        emb = row.embedding.tolist() if hasattr(row.embedding, 'tolist') else row.embedding
        return MemoryItem(
            id=row.id,
            content=content,
            tier=Tier(row.tier),
            timestamp=row.timestamp,
            embedding=emb,
            metadata=metadata,
            forgetting=forgetting,
            entities=entities
        )

    async def put(self, item: MemoryItem):
        async with self.Session() as sess:
            async with sess.begin():
                # upsert
                existing = await sess.get(MemoryRow, item.id)
                c_nonce, c_ct = self._enc(item.content)
                m_nonce, m_ct = self._enc(json.dumps(item.metadata))
                forgetting_json = json.dumps({
                    "strength": item.forgetting.strength,
                    "last_access": item.forgetting.last_access,
                    "rehearsals": item.forgetting.rehearsals,
                    "importance": item.forgetting.importance
                })
                entities_json = json.dumps(item.entities)

                if existing:
                    existing.tier = item.tier.value
                    existing.timestamp = item.timestamp
                    existing.content_enc = c_ct
                    existing.nonce = c_nonce
                    existing.metadata_enc = m_ct
                    existing.meta_nonce = m_nonce
                    existing.forgetting_json = forgetting_json
                    existing.entities_json = entities_json
                    existing.embedding = item.embedding
                    existing.importance = item.metadata.get("importance", 0.5)
                else:
                    row = MemoryRow(
                        id=item.id,
                        tier=item.tier.value,
                        timestamp=item.timestamp,
                        content_enc=c_ct,
                        nonce=c_nonce,
                        metadata_enc=m_ct,
                        meta_nonce=m_nonce,
                        forgetting_json=forgetting_json,
                        entities_json=entities_json,
                        embedding=item.embedding,
                        importance=item.metadata.get("importance", 0.5)
                    )
                    sess.add(row)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        async with self.Session() as sess:
            row = await sess.get(MemoryRow, memory_id)
            if not row:
                return None
            return self._row_to_item(row)

    async def list_all(self) -> List[MemoryItem]:
        async with self.Session() as sess:
            result = await sess.execute(select(MemoryRow).order_by(MemoryRow.timestamp.desc()))
            rows = result.scalars().all()
            return [self._row_to_item(r) for r in rows]

    async def list_by_tier(self, tier: Tier) -> List[MemoryItem]:
        async with self.Session() as sess:
            result = await sess.execute(select(MemoryRow).where(MemoryRow.tier == tier.value))
            rows = result.scalars().all()
            return [self._row_to_item(r) for r in rows]

    async def delete(self, memory_id: str):
        async with self.Session() as sess:
            async with sess.begin():
                await sess.execute(delete(MemoryRow).where(MemoryRow.id == memory_id))

    async def update_tier(self, memory_id: str, tier: Tier):
        async with self.Session() as sess:
            async with sess.begin():
                row = await sess.get(MemoryRow, memory_id)
                if row:
                    row.tier = tier.value

    # Vector search using pgvector cosine distance
    async def vector_search(self, query_embedding: List[float], k: int = 10, tier_filter: List[str] = None):
        async with self.Session() as sess:
            # cosine distance: 1 - cosine_sim, lower is better, so order by distance
            stmt = select(MemoryRow, MemoryRow.embedding.cosine_distance(query_embedding).label("distance"))
            if tier_filter:
                stmt = stmt.where(MemoryRow.tier.in_(tier_filter))
            stmt = stmt.order_by("distance").limit(k)
            result = await sess.execute(stmt)
            rows = result.all()
            out = []
            for row, dist in rows:
                sim = 1 - float(dist)
                out.append((row.id, sim))
            return out

    async def search_by_content(self, query: str, limit: int = 50) -> List[MemoryItem]:
        # Since content encrypted, brute-force decrypt scan - same limitation as SQLite version
        # In prod with Postgres, you would add a blind index column (HMAC) for encrypted search
        all_items = await self.list_all()
        q = query.lower()
        return [i for i in all_items if q in i.content.lower()][:limit]
