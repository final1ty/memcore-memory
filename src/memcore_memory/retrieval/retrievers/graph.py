
from typing import List, Dict


class GraphRetriever:
    """Memories whose linked knowledge-graph entities are named in the query.

    This used to call ``kg.get_related_memories`` behind a ``hasattr`` guard on a
    method that did not exist, so the arm returned [] for every query while
    carrying the second-largest fusion weight. No guard now: a graph without the
    method is a bug that should surface (HybridRetriever logs a failing arm).
    """

    def __init__(self, kg):
        self.kg = kg

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        # Entity matching belongs to the graph, which knows its entity names -
        # including multi-word ones that splitting the query into tokens would miss.
        mem_ids = await self.kg.get_related_memories(query, limit=k)
        n = len(mem_ids)
        # Only the order matters to RRF; the score just keeps it visible to callers.
        return [{'id': mid, 'score': (n - i) / n, 'source': 'graph'}
                for i, mid in enumerate(mem_ids[:k])]
