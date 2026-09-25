
import sys
from typing import List
from .base import BaseEmbedder

class BGEEmbedder(BaseEmbedder):
    """
    Real BGE / E5 embeddings via sentence-transformers.
    Supports:
    - BAAI/bge-small-en-v1.5 (384 dim, fast)
    - BAAI/bge-base-en-v1.5 (768 dim)
    - BAAI/bge-large-en-v1.5 (1024 dim)
    - intfloat/e5-small-v2 (384)
    - intfloat/e5-base-v2 (768)
    - intfloat/e5-large-v2 (1024)
    """
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dim: int = None, normalize: bool = True, device: str = None):
        self.model_name = model_name
        self.normalize = normalize
        self._model = None
        self.dim = dim
        self.device = device
        self._load()

    def _load(self):
        try:
            from sentence_transformers import SentenceTransformer
            import torch
        except ImportError:
            print(f"[embeddings] sentence-transformers not installed, fallback to hash. pip install sentence-transformers torch", file=sys.stderr)
            self._use_hash_fallback()
            return
        try:
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = SentenceTransformer(self.model_name, device=device)
            native = model.get_sentence_embedding_dimension()
            if not native:
                native = len(model.encode(["dimension probe"], show_progress_bar=False)[0])
        except Exception as e:
            print(f"[embeddings] Failed to load {self.model_name}: {e}, using hash fallback", file=sys.stderr)
            self._use_hash_fallback()
            return
        # Outside the fallback's except on purpose. A requested dim used to win
        # over the model's own, so bge-base kept dim=384 while emitting 768-wide
        # vectors, and every add() failed only after its row was written. A
        # mismatch is a configuration error; degrading it to the hash embedder
        # would quietly swap a working model for noise.
        if self.dim and self.dim != native:
            raise ValueError(
                f"{self.model_name} produces {native}-dim embeddings but {self.dim} were "
                f"requested (MNEM_EMBEDDING_DIM={self.dim}); set MNEM_EMBEDDING_DIM={native}. "
                f"Vectors already stored at {self.dim} dims must be re-embedded.")
        self._model = model
        self.dim = native
        print(f"[embeddings] Loaded {self.model_name} dim={self.dim} on {device}", file=sys.stderr)

    def _use_hash_fallback(self):
        # Keeps the requested dim: the live store's 384-dim hash vectors depend on it.
        from .local import LocalHashEmbedder
        fallback = LocalHashEmbedder(dim=self.dim or 768)
        self.embed = fallback.embed
        self.dim = fallback.dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not self._model:
            from .local import LocalHashEmbedder
            return LocalHashEmbedder(dim=self.dim).embed(texts)

        # BGE uses specific instruction for query vs doc for best MRR
        # For BGE: query should be with no prefix, or "Represent this sentence for searching relevant passages: "
        # E5: query prefix "query: ", doc prefix "passage: "
        processed = []
        for t in texts:
            if "e5" in self.model_name.lower():
                # caller should have added prefix, but we add if not
                if not t.startswith("query:") and not t.startswith("passage:"):
                    processed.append(f"passage: {t}")
                else:
                    processed.append(t)
            else:
                processed.append(t)

        embs = self._model.encode(processed, normalize_embeddings=self.normalize, show_progress_bar=False)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        if "e5" in self.model_name.lower() and not text.startswith("query:"):
            text = f"query: {text}"
        elif "bge" in self.model_name.lower():
            # BGE best practice for query
            # text = f"Represent this sentence for searching relevant passages: {text}"
            pass
        return super().embed_query(text)
