
import json, time
from typing import List, Optional
from ..core.tiers import MemoryItem, Tier
from ..core.ebbinghaus import ForgettingCurve
from ..crypto.aes_gcm import AES256GCM
import aiosqlite
from pathlib import Path

class EncryptedStore:
    def __init__(self, db_path: Path, cipher: AES256GCM):
        self.db_path = db_path
        self.cipher = cipher

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
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
                embedding BLOB
            )''')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
            await db.commit()

    def _enc(self, plaintext: str):
        nonce, ct = self.cipher.encrypt(plaintext.encode())
        return nonce, ct

    def _dec(self, nonce: bytes, ct: bytes) -> str:
        return self.cipher.decrypt(nonce, ct).decode()

    async def put(self, item: MemoryItem):
        c_nonce, c_ct = self._enc(item.content)
        m_nonce, m_ct = self._enc(json.dumps(item.metadata))
        forgetting_json = json.dumps({
            'strength': item.forgetting.strength,
            'last_access': item.forgetting.last_access,
            'rehearsals': item.forgetting.rehearsals,
            'importance': item.forgetting.importance
        })
        entities_json = json.dumps(item.entities)
        emb_blob = json.dumps(item.embedding).encode() if item.embedding else None
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''INSERT OR REPLACE INTO memories
                (id, tier, timestamp, content_enc, nonce, metadata_enc, meta_nonce, forgetting_json, entities_json, embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?)''',
                (item.id, item.tier.value, item.timestamp, c_ct, c_nonce, m_ct, m_nonce, forgetting_json, entities_json, emb_blob))
            await db.commit()

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute('SELECT * FROM memories WHERE id=?', (memory_id,)) as cur:
                row = await cur.fetchone()
                if not row:
                    return None
                return self._row_to_item(row)

    def _row_to_item(self, row) -> MemoryItem:
        content = self._dec(row['nonce'], row['content_enc'])
        metadata = json.loads(self._dec(row['meta_nonce'], row['metadata_enc']))
        f = json.loads(row['forgetting_json'])
        forgetting = ForgettingCurve(**f)
        entities = json.loads(row['entities_json']) if row['entities_json'] else []
        emb = json.loads(row['embedding'].decode()) if row['embedding'] else None
        return MemoryItem(
            id=row['id'],
            content=content,
            tier=Tier(row['tier']),
            timestamp=row['timestamp'],
            embedding=emb,
            metadata=metadata,
            forgetting=forgetting,
            entities=entities
        )

    async def list_all(self) -> List[MemoryItem]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            items = []
            async with db.execute('SELECT * FROM memories ORDER BY timestamp DESC') as cur:
                async for row in cur:
                    items.append(self._row_to_item(row))
            return items

    async def list_by_tier(self, tier: Tier) -> List[MemoryItem]:
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            items = []
            async with db.execute('SELECT * FROM memories WHERE tier=?', (tier.value,)) as cur:
                async for row in cur:
                    items.append(self._row_to_item(row))
            return items

    async def delete(self, memory_id: str):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('DELETE FROM memories WHERE id=?', (memory_id,))
            await db.commit()

    async def update_tier(self, memory_id: str, tier: Tier):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('UPDATE memories SET tier=? WHERE id=?', (tier.value, memory_id))
            await db.commit()

    async def search_by_content(self, query: str, limit: int = 50) -> List[MemoryItem]:
        all_items = await self.list_all()
        q = query.lower()
        return [i for i in all_items if q in i.content.lower()][:limit]
