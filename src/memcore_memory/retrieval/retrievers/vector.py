
from typing import List, Dict
from ...embeddings.factory import get_embedder

class VectorRetriever:
    def __init__(self, vector_store, embedder=None):
        self.vs = vector_store
        # Without an embedder HybridRetriever treats this arm as non-semantic and gives
        # it weight 0, so loading a real model here would only cost time and memory.
        self.embedder = embedder or get_embedder("local", dim=getattr(vector_store, 'dim', 768))

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        # Real BGE/E5 query embedding with prefix handling
        q_emb = self.embedder.embed_query(query) if hasattr(self.embedder, 'embed_query') else self.embedder.embed([query])[0]
        results = await self.vs.search(q_emb, k=k)
        return [{'id': mid, 'score': score, 'source': 'vector'} for mid, score in results]
