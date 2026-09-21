
from typing import List, Dict

class MetadataRetriever:
    def __init__(self, store):
        self.store = store

    async def retrieve(self, query: str, k: int = 20, filters: dict = None) -> List[Dict]:
        items = await self.store.list_all()
        filters = filters or {}
        scored = []
        for item in items:
            match = 0
            for fk, fv in filters.items():
                if item.metadata.get(fk) == fv:
                    match += 1
            # also tag matching
            q_lower = query.lower()
            meta_str = ' '.join(str(v).lower() for v in item.metadata.values())
            if q_lower in meta_str:
                match += 0.5
            if match > 0:
                scored.append((item, match))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': s, 'source': 'metadata'} for m, s in scored[:k]]
