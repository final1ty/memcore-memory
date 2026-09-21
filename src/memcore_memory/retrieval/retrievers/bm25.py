
from typing import List, Dict
from rank_bm25 import BM25L

class BM25Retriever:
    """Lexical retrieval over the decrypted corpus.

    Uses BM25L rather than BM25Okapi. Okapi's IDF is log((N-df+0.5)/(df+0.5)), which
    hits zero once a term appears in half the corpus and goes negative beyond that -
    so on a small personal store the *most characteristic* terms ("SkyNAS", "Docker")
    silently return nothing. BM25L keeps non-matching documents at zero while giving
    every genuine match a positive score, which is what the RRF fusion needs.
    """

    def __init__(self, store):
        self.store = store
        self._cache_key = None
        self._bm25 = None
        self._items = None

    def _index(self, items):
        # Rebuilding on every query is wasteful; re-index only when the corpus changes.
        key = (len(items), tuple(m.id for m in items))
        if key != self._cache_key:
            self._bm25 = BM25L([m.content.lower().split() for m in items])
            self._items = items
            self._cache_key = key
        return self._bm25

    async def retrieve(self, query: str, k: int = 20) -> List[Dict]:
        all_items = await self.store.list_all()
        if not all_items:
            return []
        bm25 = self._index(all_items)
        scores = bm25.get_scores(query.lower().split())
        scored = sorted(zip(all_items, scores), key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': float(s), 'source': 'bm25'} for m, s in scored[:k] if s > 0]
