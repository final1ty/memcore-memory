
from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):
    dim: int
    model_name: str

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts)
