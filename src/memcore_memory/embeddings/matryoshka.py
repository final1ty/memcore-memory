
"""
EXPERIMENTAL, NOT WIRED IN. get_embedder() never returns this class, and
``settings.matryoshka_enabled`` / ``settings.binary_quantization`` are read by
nothing: every deployment embeds with BGEEmbedder or its hash fallback.

Matryoshka (MRL) truncation of a nomic-embed model plus optional binary
quantization. Kept as a starting point only; switching to it would re-dimension
or re-embed everything already stored.
"""
import sys
from typing import List
from .base import BaseEmbedder
import numpy as np

class MatryoshkaEmbedder(BaseEmbedder):
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", dim: int = 768, binary: bool = False):
        self.model_name = model_name
        self.dim = dim
        self.binary = binary
        self.full_dim = 768  # nomic 768
        self._model = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            print(f"[matryoshka] Loaded {self.model_name} full_dim={self.full_dim} target_dim={self.dim}", file=sys.stderr)
        except Exception as e:
            print(f"[matryoshka] Failed: {e}, fallback to hash", file=sys.stderr)
            from .local import LocalHashEmbedder
            self._fallback = LocalHashEmbedder(dim=self.dim)
            self._model = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._model:
            return self._fallback.embed(texts)
        
        embs = self._model.encode(texts, normalize_embeddings=True)
        # MRL truncation: take first dim dimensions
        truncated = embs[:, :self.dim]
        # Re-normalize after truncation
        norms = np.linalg.norm(truncated, axis=1, keepdims=True)
        truncated = truncated / np.maximum(norms, 1e-12)
        
        if self.binary:
            # Binary quantization: 1 bit per dimension
            binary = (truncated > 0).astype(np.float32)
            # For pgvector binary, would store as bitstring, here return float for compat
            return binary.tolist()
        
        return truncated.tolist()

    def get_binary_embedding(self, text: str) -> bytes:
        # For pgvector binary quantization storage
        vec = self.embed([text])[0]
        # Pack bits
        import struct
        # Simple: convert to bytes of 0/1
        bits = ''.join('1' if x > 0.5 else '0' for x in vec)
        # Pack into bytes
        byte_array = bytearray()
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            byte_array.append(int(byte.ljust(8, '0'), 2))
        return bytes(byte_array)
