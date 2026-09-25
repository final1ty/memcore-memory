
"""
EXPERIMENTAL, NOT WIRED IN. Nothing in the retrieval path imports this module.

HybridRetriever.search never reranks, whatever ``settings.reranker_enabled`` says:
no cross-encoder stage exists in any recall, and no MRR figure for one has ever
been measured. SPLADERetriever and GraphRAG are placeholders that raise rather
than return empty or canned results, so nothing can mistake them for working.
Kept as a starting point; wiring the reranker needs sentence-transformers, which
no deployment here has installed, so it could not be tested.
"""
import sys
from typing import List, Dict
import asyncio

class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large", device: str = None):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._load()

    def _load(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name, device=self.device)
            print(f"[reranker] Loaded {self.model_name}", file=sys.stderr)
        except Exception as e:
            print(f"[reranker] Failed to load {self.model_name}: {e}, using fallback scoring", file=sys.stderr)
            self._model = None

    def rerank(self, query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
        if not docs:
            return []
        if not self._model:
            # Fallback: sort by existing score
            return sorted(docs, key=lambda x: x.get('score', 0), reverse=True)[:top_k]
        
        pairs = [[query, d.get('content', '')[:512]] for d in docs]
        scores = self._model.predict(pairs)
        # Copies: the caller's result dicts are not ours to rewrite.
        out = [dict(d, reranker_score=float(s)) for d, s in zip(docs, scores)]
        return sorted(out, key=lambda x: x['reranker_score'], reverse=True)[:top_k]

    async def arerank(self, query: str, docs: List[Dict], top_k: int = 10) -> List[Dict]:
        # Async wrapper
        return await asyncio.to_thread(self.rerank, query, docs, top_k)

class SPLADERetriever:
    """Placeholder for SPLADE learned sparse retrieval. Not implemented."""
    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self._model = None

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        # An empty list would read as "searched, found nothing".
        raise NotImplementedError("SPLADE retrieval is not implemented")

class GraphRAG:
    """Placeholder for GraphRAG (Edge et al. 2024). No community detection exists;
    retrieve() is plain GraphRetriever."""
    def __init__(self, kg):
        self.kg = kg

    async def retrieve(self, query: str, k: int = 10) -> List[Dict]:
        # 1. Find relevant entities in query
        # 2. Expand via KG community detection
        # 3. Summarize community
        # Stub: use existing graph retriever
        from .retrievers.graph import GraphRetriever
        retriever = GraphRetriever(self.kg)
        return await retriever.retrieve(query, k=k)

    async def community_summarize(self, entity_ids: List[str]) -> str:
        # This returned "Community summary for N entities" - a canned string
        # dressed up as a result.
        raise NotImplementedError("GraphRAG community summarisation is not implemented")
