
from typing import List, Dict
from rank_bm25 import BM25Okapi

class BM25Retriever:
    def __init__(self, store):
        self.store = store

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        all_items = await self.store.list_all()
        if not all_items:
            return []
        corpus = [m.content for m in all_items]
        tokenized_corpus = [c.lower().split() for c in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        scored = list(zip(all_items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': float(s), 'source': 'bm25'} for m, s in scored[:k] if s > 0]
