
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

class VectorStore:
    def __init__(self, path: Path, dim: int = 768):
        self.path = path
        self.dim = dim
        self.id_to_meta: Dict[str, dict] = {}
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        if HAS_HNSW:
            self.index = hnswlib.Index(space='cosine', dim=dim)
            self.index.init_index(max_elements=100000, ef_construction=200, M=16)
            if path.exists():
                try:
                    self.index.load_index(str(path))
                    meta_path = path.with_suffix('.meta.json')
                    if meta_path.exists():
                        data = json.loads(meta_path.read_text())
                        self.id_to_meta = data.get('meta', {})
                        self.ids = data.get('ids', [])
                except Exception:
                    pass
        else:
            self.index = None

    async def add(self, mem_id: str, embedding: List[float], meta: dict):
        self.id_to_meta[mem_id] = meta
        if mem_id not in self.ids:
            self.ids.append(mem_id)
            self.vectors.append(embedding)
            if self.index:
                try:
                    self.index.add_items(np.array([embedding], dtype=np.float32), np.array([len(self.ids)-1]))
                except Exception:
                    pass
        else:
            idx = self.ids.index(mem_id)
            self.vectors[idx] = embedding
        self._persist()

    async def delete(self, mem_id: str):
        if mem_id in self.id_to_meta:
            del self.id_to_meta[mem_id]
        if mem_id in self.ids:
            idx = self.ids.index(mem_id)
            del self.ids[idx]
            del self.vectors[idx]
        self._persist()

    def _persist(self):
        if self.index and len(self.ids) > 0:
            try:
                self.index.save_index(str(self.path))
                self.path.with_suffix('.meta.json').write_text(json.dumps({'meta': self.id_to_meta, 'ids': self.ids}))
            except Exception:
                pass

    async def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        if self.index and len(self.ids) >= k:
            try:
                labels, distances = self.index.knn_query(q, k=min(k, len(self.ids)))
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    mem_id = self.ids[label]
                    sim = 1 - dist
                    results.append((mem_id, float(sim)))
                return results
            except Exception:
                pass
        sims = []
        for mem_id, vec in zip(self.ids, self.vectors):
            v = np.array(vec, dtype=np.float32)
            denom = (np.linalg.norm(q) * np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q, v) / denom)
            sims.append((mem_id, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]
