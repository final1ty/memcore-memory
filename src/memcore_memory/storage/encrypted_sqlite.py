
import json, time
from typing import List, Optional
from ..core.tiers import MemoryItem, Tier
from ..core.ebbinghaus import ForgettingCurve
from ..crypto.aes_gcm import AES256GCM
from ..crypto.blind_index import BlindIndex
import aiosqlite
from pathlib import Path

BLIND_INDEX_INFO = b"blind-index-v1"


class EncryptedStore:
    def __init__(self, db_path: Path, cipher: AES256GCM, blind_index: bool = None):
        self.db_path = db_path
        self.cipher = cipher
        if blind_index is None:
            from ..config import settings
            blind_index = getattr(settings, "blind_index_enabled", True)
        # Keyed separately from the content cipher - see AES256GCM.derive_subkey.
        self.blind = BlindIndex(cipher.derive_subkey(BLIND_INDEX_INFO)) if blind_index else None

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
                embedding BLOB,
                blind_index_json TEXT
            )''')
            # Stores created before the blind index existed are missing the column.
            # Their rows stay NULL and simply don't match blind searches until they
            # are written again or rebuild_blind_index() is run.
            async with db.execute("PRAGMA table_info(memories)") as cur:
                columns = {row[1] async for row in cur}
            if "blind_index_json" not in columns:
                await db.execute("ALTER TABLE memories ADD COLUMN blind_index_json TEXT")
            await db.execute('CREATE INDEX IF NOT EXISTS idx_tier ON memories(tier)')
            await db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)')
            await db.commit()

    def _enc(self, plaintext: str):
        nonce, ct = self.cipher.encrypt(plaintext.encode())
        return nonce, ct

    def _dec(self, nonce: bytes, ct: bytes) -> str:
        return self.cipher.decrypt(nonce, ct).decode()

    def _blind_index_json(self, item: MemoryItem) -> Optional[str]:
        if not self.blind:
            return None
        return json.dumps(self.blind.compute_index(item.content, item.metadata))

    async def put(self, item: MemoryItem):
        c_nonce, c_ct = self._enc(item.content)
        m_nonce, m_ct = self._enc(json.dumps(item.metadata))
        forgetting_json = json.dumps(item.forgetting.to_dict())
        entities_json = json.dumps(item.entities)
        emb_blob = json.dumps(item.embedding).encode() if item.embedding else None
        blind_json = self._blind_index_json(item)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''INSERT OR REPLACE INTO memories
                (id, tier, timestamp, content_enc, nonce, metadata_enc, meta_nonce, forgetting_json, entities_json, embedding, blind_index_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (item.id, item.tier.value, item.timestamp, c_ct, c_nonce, m_ct, m_nonce, forgetting_json, entities_json, emb_blob, blind_json))
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
        forgetting = ForgettingCurve.from_dict(
            json.loads(row['forgetting_json']) if row['forgetting_json'] else {},
            default_last_access=row['timestamp'],
        )
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
        """Move a memory between tiers, raising its decay strength to the new floor.

        Strength is only ever raised, never lowered: a memory that earned a long
        retention through rehearsal keeps it even if it is demoted.
        """
        from ..core.tiers import TIER_BASE_STRENGTH
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT forgetting_json FROM memories WHERE id=?', (memory_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return
            forgetting = json.loads(row[0])
            forgetting['strength'] = max(forgetting.get('strength', 1.0),
                                         TIER_BASE_STRENGTH.get(tier, 7.0))
            await db.execute('UPDATE memories SET tier=?, forgetting_json=? WHERE id=?',
                             (tier.value, json.dumps(forgetting), memory_id))
            await db.commit()

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
        wanted = set(self.blind.search_query_hmacs(query))
        if not wanted:
            return []
        matches = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                'SELECT id, blind_index_json FROM memories WHERE blind_index_json IS NOT NULL'
            ) as cur:
                async for row in cur:
                    overlap = len(wanted & set(json.loads(row[1])))
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
        items = await self.list_all()
        async with aiosqlite.connect(self.db_path) as db:
            for item in items:
                await db.execute(
                    'UPDATE memories SET blind_index_json=? WHERE id=?',
                    (self._blind_index_json(item), item.id),
                )
            await db.commit()
        return len(items)
