
import hashlib, numpy as np
from typing import List
from .base import BaseEmbedder

class LocalHashEmbedder(BaseEmbedder):
    """Deterministic fallback - for tests / offline"""
    def __init__(self, dim: int = 768):
        self.dim = dim
        self.model_name = f"local-hash-{dim}"

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            seed = int.from_bytes(h[:4], 'big')
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.dim).astype(float).tolist()
            # L2 normalize
            import math
            norm = math.sqrt(sum(x*x for x in vec)) or 1.0
            vec = [x / norm for x in vec]
            out.append(vec)
        return out
