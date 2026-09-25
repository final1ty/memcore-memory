
from typing import List, Dict

class ImportanceRetriever:
    def __init__(self, store):
        self.store = store

    async def retrieve(self, query: str = "", k: int = 20, items=None) -> List[Dict]:
        """Rank by importance, tier and rehearsals. ``query`` is ignored.

        A query-independent prior: HybridRetriever uses it only to reorder what
        the query-dependent arms found.
        """
        if items is None:
            items = await self.store.list_all()
        scored = []
        for item in items:
            imp = item.metadata.get('importance', 0.5)
            # boost by rehearsals and tier
            tier_boost = {'sensory':0.1, 'working':0.3, 'episodic':0.6, 'semantic':1.0}.get(item.tier.value, 0.5)
            score = imp * 0.6 + tier_boost * 0.2 + (item.forgetting.rehearsals / 10.0) * 0.2
            scored.append((item, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': s, 'source': 'importance'} for m, s in scored[:k]]
