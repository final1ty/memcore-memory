
from typing import List, Dict

class GraphRetriever:
    def __init__(self, kg):
        self.kg = kg

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        # naive: extract entity-like tokens
        tokens = [t for t in query.split() if len(t) > 3]
        all_hits = []
        for tok in tokens[:3]:
            hits = await self.kg.traverse(tok, depth=1, limit=k)
            for h in hits:
                # need to find memory ids linked to this edge
                all_hits.append(h)
        # Convert graph hits to memory ids via related memories
        mem_ids = []
        for tok in tokens[:3]:
            mids = await self.kg.get_related_memories(tok) if hasattr(self.kg, 'get_related_memories') else []
            for mid in mids:
                mem_ids.append({'id': mid, 'score': 0.7, 'source': 'graph'})
        return mem_ids[:k]
