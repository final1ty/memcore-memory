
from typing import List, Dict
from .bm25 import tokenize

# Written by MnemosyneMemory.add on every memory, so matching against it matched
# the whole store: the query "5" hit every memory stored with importance 0.5.
# The rest is provenance, not content. scripts/sync-rest-to-local.py stamps
# imported_from="docker-rest-store" on every record it copies, so any query
# containing "docker", "rest" or "store" nominated the whole imported store and
# the priors ranked it - padding through the back door.
_RESERVED = {'importance', 'imported_from', 'origin_id', 'id', 'uuid', 'hash', 'checksum'}

# A token carried by more than this share of the snapshot distinguishes nothing:
# on the live store every memory has source="claude-code-session", so "claude"
# or "session" would nominate all of them. Only applied from this many memories
# up, so a handful of notes sharing one tag stay findable by it.
_COMMON_SHARE = 0.5
_COMMON_MIN_ITEMS = 10


def _skipped(key) -> bool:
    return key in _RESERVED or (isinstance(key, str) and key.endswith('_id'))


def _value_tokens(metadata: dict) -> set:
    tokens = set()
    for key, value in metadata.items():
        if _skipped(key):
            continue
        # Text only. Numbers and booleans stringified into the haystack are what
        # made numeric queries match everything.
        if isinstance(value, str):
            tokens.update(tokenize(value))
        elif isinstance(value, (list, tuple)):
            for v in value:
                if isinstance(v, str):
                    tokens.update(tokenize(v))
    return tokens


class MetadataRetriever:
    def __init__(self, store):
        self.store = store

    async def retrieve(self, query: str, k: int = 20, filters: dict = None, items=None) -> List[Dict]:
        """Match query terms against string metadata values.

        Tokenised exactly like BM25, so "claude-export hiking" finds memories whose
        source is "claude-export" - the old whole-query substring test matched no
        multi-word query at all. Provenance keys, and tokens shared by most of a
        store of 10 or more, are ignored: they would nominate everything. ``filters``
        (exact key/value equality) is reachable only through
        HybridRetriever.search(metadata_filter=...), i.e. the Python API; no REST,
        MCP or SDK surface passes it.
        """
        if items is None:
            items = await self.store.list_all()
        filters = filters or {}
        q_tokens = set(tokenize(query))
        per_item = [(item, _value_tokens(item.metadata or {}) & q_tokens) for item in items]
        if q_tokens and len(items) >= _COMMON_MIN_ITEMS:
            df = {}
            for _, toks in per_item:
                for t in toks:
                    df[t] = df.get(t, 0) + 1
            common = {t for t, n in df.items() if n > _COMMON_SHARE * len(items)}
            per_item = [(item, toks - common) for item, toks in per_item]
        scored = []
        for item, overlap in per_item:
            match = 0.0
            for fk, fv in filters.items():
                if item.metadata.get(fk) == fv:
                    match += 1
            if q_tokens:
                match += 0.5 * len(overlap) / len(q_tokens)
            if match > 0:
                scored.append((item, match))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [{'id': m.id, 'score': s, 'source': 'metadata'} for m, s in scored[:k]]
