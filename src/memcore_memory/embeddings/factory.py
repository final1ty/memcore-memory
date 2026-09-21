
from typing import Literal
from .base import BaseEmbedder

EmbeddingProvider = Literal["bge-small", "bge-base", "bge-large", "e5-small", "e5-base", "e5-large", "local"]

MODEL_MAP = {
    "bge-small": "BAAI/bge-small-en-v1.5",
    "bge-base": "BAAI/bge-base-en-v1.5",
    "bge-large": "BAAI/bge-large-en-v1.5",
    "e5-small": "intfloat/e5-small-v2",
    "e5-base": "intfloat/e5-base-v2",
    "e5-large": "intfloat/e5-large-v2",
    "local": "local-hash",
}

def get_embedder(provider: EmbeddingProvider = "bge-small", dim: int = None, device: str = None) -> BaseEmbedder:
    if provider == "local":
        from .local import LocalHashEmbedder
        return LocalHashEmbedder(dim=dim or 768)

    model_name = MODEL_MAP.get(provider, provider)  # allow direct HF id
    from .bge import BGEEmbedder
    return BGEEmbedder(model_name=model_name, dim=dim, device=device)
