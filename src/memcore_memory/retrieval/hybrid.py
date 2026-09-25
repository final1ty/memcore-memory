
import asyncio
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

    async def _run_arms(self, names, calls):
        results = await asyncio.gather(*(calls[name]() for name in names),
                                       return_exceptions=True)
        lists, failed = [], []
        for name, r in zip(names, results):
            if isinstance(r, asyncio.CancelledError):
                raise r
            if isinstance(r, BaseException):
                # Dropping it silently turned a broken arm (InvalidTag, a dimension
                # mismatch) into quietly worse results with no trace anywhere.
                failed.append(name)
                print(f"[retrieval] {name} arm failed: {r!r}", file=sys.stderr)
            elif isinstance(r, list):
                lists.append(r)
            else:
                print(f"[retrieval] {name} arm returned {type(r).__name__}, expected a list; ignored",
                      file=sys.stderr)
        return lists, failed

    async def search(self, query: str, k: int = 10, tier_filter: List[str] = None, metadata_filter: dict = None) -> List[Dict]:
        # One decrypted snapshot per search. Four retrievers used to call list_all()
        # each and then store.get() every fused id: ~4N+2k AES-GCM decrypts and ~30
        # SQLite connections for a single recall.
        items = await self.store.list_all()
        if not items:
            return []

        depth = k * 2
        allowed = None
        if tier_filter:
            # Filter before any arm truncates. Filtering the fused top-2k afterwards
            # returned nothing whenever the requested tier ranked below 2k, which on
            # a store of 7 working vs 100 episodic memories is the common case. Tier
            # comes from SQLite: the vector sidecar's copy goes stale on update_tier.
            allowed = {m.id for m in items if m.tier.value in tier_filter}
            if not allowed:
                return []
            depth = max(depth, len(items))

        calls = {
            'vector': lambda: self.vector.retrieve(query, depth),
            'bm25': lambda: self.bm25.retrieve(query, depth, items=items),
            'graph': lambda: self.graph.retrieve(query, depth),
            'metadata': lambda: self.metadata.retrieve(query, depth, filters=metadata_filter, items=items),
        }
        # A zero-weight arm contributes nothing to the fused score, but it would
        # still nominate candidates for the priors - under the hash fallback that
        # is 2k random memories per query. Don't run it at all.
        active = [name for name in calls if self.weights.get(name, 0.0) > 0.0]
        lists, failed = await self._run_arms(active, calls)
        if active and len(failed) == len(active):
            # Returning the priors alone would look like a successful search.
            raise RuntimeError(f"every query-dependent retriever failed: {failed}")

        if allowed is not None:
            lists = [[item for item in lst if item['id'] in allowed] for lst in lists]

        # TemporalRetriever and ImportanceRetriever never look at the query: they
        # rank the whole store by recency and by importance, identically for every
        # search. Fused as equals they contributed ~31% of the weight in favour of
        # the same few documents no matter what was asked, which is why the newest
        # high-importance memory came back top of every query. They are priors, so
        # let them reorder what the query-dependent retrievers actually found
        # rather than nominate candidates of their own - and when nothing matched,
        # the answer is nothing, not k rows of recent-and-important padding.
        candidates = {item['id'] for lst in lists for item in lst}
        if not candidates:
            return []
        # Ranked among the candidates, not cut to the store-wide top 2k first. The
        # cut gave a match just outside that top 2k no prior at all while every
        # match inside it got up to a third of the weight, so a graph hit lost to
        # stopword matches that happened to be important; and because a tier filter
        # changes depth, the same memories ranked differently with and without one.
        pool = [m for m in items if m.id in candidates]
        prior_calls = {
            'temporal': lambda: self.temporal.retrieve(query, len(pool), items=pool),
            'importance': lambda: self.importance.retrieve(query, len(pool), items=pool),
        }
        prior_lists, _ = await self._run_arms(
            [name for name in prior_calls if self.weights.get(name, 0.0) > 0.0], prior_calls)
        lists.extend(prior_lists)

        fused = self._rrf(lists)
        sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k*2]
        # A fresh read rather than the snapshot, in one connection: tier or
        # rehearsals may have moved since the snapshot, and ids from the vector
        # store or the graph may no longer exist at all.
        fetched = await self.store.get_many([mem_id for mem_id, _ in sorted_ids])
        final = []
        for mem_id, fused_score in sorted_ids:
            item = fetched.get(mem_id)
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
                'entities': item.entities,
                # Always true today, since only query-dependent arms nominate.
                # MnemosyneMemory.recall rehearses only matched results, so this
                # keeps that bound if a prior-only row ever gets through again.
                'matched': mem_id in candidates,
            })
        final.sort(key=lambda x: x['score'], reverse=True)
        return final[:k]
