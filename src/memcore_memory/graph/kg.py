
import json, time, uuid
from typing import List, Dict
import networkx as nx
from pathlib import Path
import aiosqlite
from ..crypto.aes_gcm import AES256GCM

class KnowledgeGraph:
    def __init__(self, db_path: Path, cipher: AES256GCM):
        self.db_path = db_path
        self.cipher = cipher
        self.graph = nx.MultiDiGraph()

    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute('''CREATE TABLE IF NOT EXISTS kg_nodes (
                id TEXT PRIMARY KEY,
                type TEXT,
                label_enc BLOB,
                nonce BLOB,
                props_enc BLOB,
                props_nonce BLOB
            )''')
            await db.execute('''CREATE TABLE IF NOT EXISTS kg_edges (
                id TEXT PRIMARY KEY,
                src TEXT,
                dst TEXT,
                relation TEXT,
                weight REAL,
                timestamp REAL,
                props_enc BLOB,
                props_nonce BLOB
            )''')
            await db.commit()

    async def add_memory_entities(self, mem_item):
        async with aiosqlite.connect(self.db_path) as db:
            for ent in mem_item.entities:
                node_id = ent.lower()
                n, ct = self.cipher.encrypt(ent.encode())
                pn, pct = self.cipher.encrypt(b'{}')
                await db.execute('INSERT OR IGNORE INTO kg_nodes (id, type, label_enc, nonce, props_enc, props_nonce) VALUES (?,?,?,?,?,?)',
                                 (node_id, 'entity', ct, n, pct, pn))
                self.graph.add_node(node_id, label=ent)
            for i in range(len(mem_item.entities)):
                for j in range(i+1, len(mem_item.entities)):
                    eid = str(uuid.uuid4())
                    src = mem_item.entities[i].lower()
                    dst = mem_item.entities[j].lower()
                    n, ct = self.cipher.encrypt(json.dumps({'memory_id': mem_item.id}).encode())
                    await db.execute('INSERT INTO kg_edges (id, src, dst, relation, weight, timestamp, props_enc, props_nonce) VALUES (?,?,?,?,?,?,?,?)',
                                     (eid, src, dst, 'co_occurs', 1.0, time.time(), ct, n))
                    self.graph.add_edge(src, dst, relation='co_occurs')
            await db.commit()

    async def add_entity(self, entity: str, type: str = 'entity', props: Dict = None) -> str:
        node_id = entity.lower()
        async with aiosqlite.connect(self.db_path) as db:
            n, ct = self.cipher.encrypt(entity.encode())
            pn, pct = self.cipher.encrypt(json.dumps(props or {}).encode())
            await db.execute('INSERT OR REPLACE INTO kg_nodes (id, type, label_enc, nonce, props_enc, props_nonce) VALUES (?,?,?,?,?,?)',
                             (node_id, type, ct, n, pct, pn))
            await db.commit()
        self.graph.add_node(node_id, label=entity)
        return node_id

    async def add_relation(self, src: str, dst: str, relation: str = 'related_to', weight: float = 1.0) -> str:
        src, dst = src.lower(), dst.lower()
        eid = str(uuid.uuid4())
        async with aiosqlite.connect(self.db_path) as db:
            for node in (src, dst):
                n, ct = self.cipher.encrypt(node.encode())
                pn, pct = self.cipher.encrypt(b'{}')
                await db.execute('INSERT OR IGNORE INTO kg_nodes (id, type, label_enc, nonce, props_enc, props_nonce) VALUES (?,?,?,?,?,?)',
                                 (node, 'entity', ct, n, pct, pn))
            n, ct = self.cipher.encrypt(b'{}')
            await db.execute('INSERT INTO kg_edges (id, src, dst, relation, weight, timestamp, props_enc, props_nonce) VALUES (?,?,?,?,?,?,?,?)',
                             (eid, src, dst, relation, weight, time.time(), ct, n))
            await db.commit()
        self.graph.add_edge(src, dst, relation=relation)
        return eid

    async def list_entities(self, limit: int = 100) -> List[Dict]:
        out = []
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute('SELECT id, type, label_enc, nonce FROM kg_nodes LIMIT ?', (limit,)) as cur:
                async for row in cur:
                    out.append({'id': row[0], 'type': row[1],
                                'label': self.cipher.decrypt(row[3], row[2]).decode()})
        return out

    async def delete_entity(self, entity: str) -> Dict:
        node_id = entity.lower()
        async with aiosqlite.connect(self.db_path) as db:
            cur = await db.execute('DELETE FROM kg_nodes WHERE id=?', (node_id,))
            nodes = cur.rowcount
            cur = await db.execute('DELETE FROM kg_edges WHERE src=? OR dst=?', (node_id, node_id))
            edges = cur.rowcount
            await db.commit()
        if self.graph.has_node(node_id):
            self.graph.remove_node(node_id)
        return {'entity': node_id, 'nodes_deleted': nodes, 'edges_deleted': edges}

    async def traverse(self, entity: str, depth: int = 2, limit: int = 20) -> List[Dict]:
        entity = entity.lower()
        visited = set()
        queue = [(entity, 0)]
        results = []
        async with aiosqlite.connect(self.db_path) as db:
            while queue and len(results) < limit:
                curr, d = queue.pop(0)
                if curr in visited or d > depth:
                    continue
                visited.add(curr)
                async with db.execute('SELECT * FROM kg_edges WHERE src=? OR dst=?', (curr, curr)) as cur:
                    async for row in cur:
                        other = row[2] if row[1] == curr else row[1]
                        results.append({'src': row[1], 'dst': row[2], 'relation': row[3], 'weight': row[4]})
                        if other not in visited:
                            queue.append((other, d+1))
        return results
