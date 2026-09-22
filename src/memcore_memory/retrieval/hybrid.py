
from typing import List, Dict
from .retrievers.vector import VectorRetriever
from .retrievers.bm25 import BM25Retriever
from .retrievers.graph import GraphRetriever
from .retrievers.temporal import TemporalRetriever
from .retrievers.importance import ImportanceRetriever
from .retrievers.metadata import MetadataRetriever

import sys

# Which retrievers actually look at the query. The other two rank the whole store
# by recency and importance and return the same order for every search.
QUERY_DEPENDENT = {'vector', 'bm25', 'graph', 'metadata'}

# Weights from the project's grid search on LoCoMo + LongMemEval, which was run
# with real BGE embeddings.
DEFAULT_WEIGHTS = {
    'vector': 0.35,
    'bm25': 0.25,
    'graph': 0.15,
    'temporal': 0.10,
    'importance': 0.10,
    'metadata': 0.05,
}


def is_semantic(embedder) -> bool:
    """True when the embedder produces meaning-bearing vectors.

    LocalHashEmbedder hashes text to a deterministic random vector: identical
    strings match and everything else is noise, so similarity carries no meaning.
    BGEEmbedder degrades to it when sentence-transformers is missing while keeping
    its model_name, so the loaded model is what to check, not the label.
    """
    if embedder is None:
        return False
    if type(embedder).__name__ == "LocalHashEmbedder":
        return False
    if hasattr(embedder, "_model"):
        return getattr(embedder, "_model") is not None
    return True


class HybridRetriever:
    def __init__(self, store, vector_store, kg, embedder=None):
        self.store = store
        self.vector = VectorRetriever(vector_store, embedder=embedder)
        self.bm25 = BM25Retriever(store)
        self.graph = GraphRetriever(kg)
        self.temporal = TemporalRetriever(store)
        self.importance = ImportanceRetriever(store)
        self.metadata = MetadataRetriever(store)
        self.weights = dict(DEFAULT_WEIGHTS)
        # Those weights assume the vector arm means something. Under the hash
        # fallback it is noise, and it carries the largest weight of the six - so
        # letting it vote actively buries the lexical matches BM25 found. Give its
        # share to the retrievers that still work.
        if not is_semantic(embedder):
            share = self.weights.pop('vector')
            total = sum(self.weights.values())
            for name in self.weights:
                self.weights[name] += share * self.weights[name] / total
            self.weights['vector'] = 0.0
            print("[retrieval] embedder is not semantic (hash fallback): vector weight "
                  "set to 0 and redistributed. Install sentence-transformers to use it.",
                  file=sys.stderr)
        self.rrf_k = 60

    def _rrf(self, ranked_lists: List[List[Dict]]) -> Dict[str, float]:
        """Weighted Reciprocal Rank Fusion: sum of w / (k + rank) per retriever.

        This used to be `rrf_score * 0.6 + item['score'] * 0.4 * w`, which had two
        problems. The rank term carried no weight at all, so the documented weights
        (vector 0.35, bm25 0.25, ...) could only ever nudge the score term - they
        were very nearly inert. And the score term mixes raw numbers from six
        retrievers on different scales: cosine similarity in [0,1], unbounded BM25,
        a recency weight, an importance value. Comparing those directly is exactly
        what RRF exists to avoid; rank is the only comparable quantity here.
        """
        scores: Dict[str, float] = {}
        for lst in ranked_lists:
            for rank, item in enumerate(lst, start=1):
                w = self.weights.get(item['source'], 0.1)
                scores[item['id']] = scores.get(item['id'], 0.0) + w / (self.rrf_k + rank)
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

        # TemporalRetriever and ImportanceRetriever never look at the query: they
        # rank the whole store by recency and by importance, identically for every
        # search. Fused as equals they contributed ~31% of the weight in favour of
        # the same few documents no matter what was asked, which is why the newest
        # high-importance memory came back top of every query. They are priors, so
        # let them reorder what the query-dependent retrievers actually found
        # rather than nominate candidates of their own.
        candidates = {item['id'] for lst in lists for item in lst
                      if item['source'] in QUERY_DEPENDENT}
        if candidates:
            lists = [[item for item in lst
                      if item['source'] in QUERY_DEPENDENT or item['id'] in candidates]
                     for lst in lists]

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
