
from typing import List, Dict
from .retrievers.vector import VectorRetriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.graph import GraphRetriever
from .retrievers.temporal import TemporalRetriever
from .retrievers.importance import ImportanceRetriever
from .retrievers.metadata import MetadataRetriever

class HybridRetriever:
    def __init__(self, store, vector_store, kg, embedder=None):
        self.store = store
        self.vector = VectorRetriever(vector_store, embedder=embedder)
        self.bm25 = BM25Retriever(store)
        self.graph = GraphRetriever(kg)
        self.temporal = TemporalRetriever(store)
        self.importance = ImportanceRetriever(store)
        self.metadata = MetadataRetriever(store)
        self.weights = {
            'vector': 0.35,
            'bm25': 0.25,
            'graph': 0.15,
            'temporal': 0.10,
            'importance': 0.10,
            'metadata': 0.05
        }
        self.rrf_k = 60

    def _rrf(self, ranked_lists: List[List[Dict]]) -> Dict[str, float]:
        scores = {}
        for lst in ranked_lists:
            for rank, item in enumerate(lst, start=1):
                mid = item['id']
                rrf_score = 1.0 / (self.rrf_k + rank)
                w = self.weights.get(item['source'], 0.1)
                combined = rrf_score * 0.6 + item['score'] * 0.4 * w
                scores[mid] = scores.get(mid, 0) + combined
        return scores

    async def search(self, query: str, k: int = 10, tier_filter: List[str] = None, metadata_filter: dict = None) -> List[Dict]:
        import asyncio
        tasks = [
            self.vector.retrieve(query, k*2),
            self.bm25.retrieve(query, k*2),
            self.graph.retrieve(query, k*2),
            self.temporal.retrieve(query, k*2),
            self.importance.retrieve(query, k*2),
            self.metadata.retrieve(query, k*2, filters=metadata_filter)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        lists = [r for r in results if isinstance(r, list)]
        fused = self._rrf(lists)
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k*2]
        final = []
        for mem_id, fused_score in sorted_ids:
            item = await self.store.get(mem_id)
            if not item:
                continue
            if tier_filter and item.tier.value not in tier_filter:
                continue
            retention = item.forgetting.retention()
            final_score = fused_score * (0.7 + 0.3 * retention)
            final.append({
                'id': item.id,
                'content': item.content,
                'tier': item.tier.value,
                'score': final_score,
                'retention': retention,
                'timestamp': item.timestamp,
                'metadata': item.metadata,
                'entities': item.entities
            })
        final.sort(key=lambda x: x['score'], reverse=True)
        return final[:k]
