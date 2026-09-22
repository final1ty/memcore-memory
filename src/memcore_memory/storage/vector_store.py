
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False


class VectorStore:
    """Vectors plus an optional HNSW index over them.

    The lists are the source of truth and are always written to a JSON sidecar.
    Persistence used to run only `if self.index` - so on any machine without
    hnswlib (which is every machine here: it is not installed in the container or
    in .venv) nothing was ever saved and nothing was ever loaded. The vector arm
    of the hybrid retriever silently started empty after every restart and only
    knew about memories added since boot.

    The index is rebuilt from the lists rather than mutated in place. Labels used
    to be `len(self.ids)-1`, i.e. positions, while `delete()` removed from the
    middle of the list and shifted every later position - so after a single
    delete the index mapped labels to the wrong memories, silently.
    """

    SIDECAR_SUFFIX = ".vectors.json"

    def __init__(self, path: Path, dim: int = 768):
        self.path = Path(path)
        self.dim = dim
        self.id_to_meta: Dict[str, dict] = {}
        self.ids: List[str] = []
        self.vectors: List[List[float]] = []
        self.index = None
        self._dirty = True
        self._load()

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(self.SIDECAR_SUFFIX)

    def _load(self):
        if not self.sidecar_path.exists():
            return
        try:
            data = json.loads(self.sidecar_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"[vectors] cannot read {self.sidecar_path}: {e}", file=sys.stderr)
            return
        stored_dim = data.get("dim")
        if stored_dim and stored_dim != self.dim:
            # Vectors from a different embedder are not comparable to new ones;
            # mixing them produces plausible-looking nonsense similarities.
            print(f"[vectors] {self.sidecar_path} holds {stored_dim}-dim vectors but this "
                  f"store is {self.dim}-dim - ignoring them. Re-embed to use them again.",
                  file=sys.stderr)
            return
        self.ids = data.get("ids", [])
        self.vectors = data.get("vectors", [])
        self.id_to_meta = data.get("meta", {})
        if len(self.ids) != len(self.vectors):
            print(f"[vectors] {self.sidecar_path} is inconsistent "
                  f"({len(self.ids)} ids, {len(self.vectors)} vectors) - starting empty",
                  file=sys.stderr)
            self.ids, self.vectors = [], []

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sidecar_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "dim": self.dim,
            "ids": self.ids,
            "vectors": self.vectors,
            "meta": self.id_to_meta,
        }))
        tmp.replace(self.sidecar_path)
        if self.index is not None and self.ids:
            try:
                self.index.save_index(str(self.path))
            except Exception as e:  # noqa: BLE001 - the sidecar already holds the data
                print(f"[vectors] could not save the HNSW index: {e}", file=sys.stderr)

    def _rebuild_index(self):
        """Rebuild from the lists, so labels and positions can never drift apart."""
        self._dirty = False
        if not HAS_HNSW or not self.ids:
            self.index = None if not HAS_HNSW else self.index
            return
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=max(1024, len(self.ids) * 2), ef_construction=200, M=16)
        index.add_items(np.array(self.vectors, dtype=np.float32), np.arange(len(self.ids)))
        index.set_ef(max(32, min(200, len(self.ids))))
        self.index = index

    async def add(self, mem_id: str, embedding: List[float], meta: dict):
        if len(embedding) != self.dim:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, store expects {self.dim}. "
                f"Storing it would make every similarity against it meaningless."
            )
        self.id_to_meta[mem_id] = meta
        if mem_id in self.ids:
            self.vectors[self.ids.index(mem_id)] = embedding
        else:
            self.ids.append(mem_id)
            self.vectors.append(embedding)
        self._dirty = True
        self._persist()

    async def delete(self, mem_id: str):
        self.id_to_meta.pop(mem_id, None)
        if mem_id in self.ids:
            idx = self.ids.index(mem_id)
            del self.ids[idx]
            del self.vectors[idx]
            self._dirty = True
        self._persist()

    async def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        if not self.vectors:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        if q.shape[0] != self.dim:
            print(f"[vectors] query is {q.shape[0]}-dim, store is {self.dim}-dim - no results",
                  file=sys.stderr)
            return []

        if HAS_HNSW:
            if self._dirty:
                self._rebuild_index()
            if self.index is not None:
                try:
                    labels, distances = self.index.knn_query(q, k=min(k, len(self.ids)))
                    return [(self.ids[label], float(1 - dist))
                            for label, dist in zip(labels[0], distances[0])]
                except Exception as e:  # noqa: BLE001 - fall back to exact search
                    print(f"[vectors] HNSW query failed ({e}), using exact search",
                          file=sys.stderr)

        matrix = np.asarray(self.vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9
        sims = (matrix @ q) / norms
        order = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i])) for i in order]
