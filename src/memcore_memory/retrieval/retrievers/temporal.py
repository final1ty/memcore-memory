
from typing import List, Dict
import time, math

class TemporalRetriever:
    def __init__(self, store):
        self.store = store

    async def retrieve(self, query: str = "", k: int = 20, items=None) -> List[Dict]:
        """Rank by Ebbinghaus retention and recency. ``query`` is ignored.

        A query-independent prior: every query gets the same order. HybridRetriever
        uses it only to reorder what the query-dependent arms found; the signature
        takes ``query`` so all arms can be called alike.
        """
        if items is None:
            items = await self.store.list_all()
        now = time.time()
        scored = []
        for item in items:
            # recency + Ebbinghaus retention
            retention = item.forgetting.retention(now)
            time_decay = math.exp(-(now - item.timestamp) / 86400.0 / 7.0)  # week decay
            score = 0.6 * retention + 0.4 * time_decay
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': s, 'source': 'temporal'} for m, s in scored[:k]]
