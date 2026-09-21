
import time
from typing import List, Optional, Dict, Any
from .tiers import MemoryItem, Tier, TierManager
from ..storage.encrypted_sqlite import EncryptedStore
from ..storage.vector_store import VectorStore
from ..graph.kg import KnowledgeGraph
from ..retrieval.hybrid import HybridRetriever
from ..config import settings
from ..embeddings.factory import get_embedder

class MnemosyneMemory:
    def __init__(self, store, vector_store, kg, embedder=None):
        self.store = store
        self.vectors = vector_store
        self.kg = kg
        self.tiers = TierManager(settings)
        # Embedder: BGE/E5 real model or local hash fallback
        self.embedder = embedder or get_embedder(
            provider=getattr(settings, 'embedding_provider', 'bge-small'),
            dim=getattr(settings, 'embedding_dim', 768)
        )
        # Update vector store dim to match embedder if needed
        if hasattr(vector_store, 'dim'):
            vector_store.dim = self.embedder.dim
        self.retriever = HybridRetriever(store, vector_store, kg, embedder=self.embedder)

    async def add(self, content: str, metadata: Dict[str, Any] = None, tier: str = None, importance: float = 0.5, entities: List[str] = None) -> MemoryItem:
        metadata = metadata or {}
        metadata['importance'] = importance
        item = MemoryItem(content=content, metadata=metadata, tier=Tier(tier) if tier else Tier.EPISODIC, entities=entities or [])
        # Real embedding via BGE/E5
        if not item.embedding:
            # Use passage prefix for E5, handled inside embedder
            item.embedding = self.embedder.embed_query(content) if hasattr(self.embedder, 'embed_query') else self.embedder.embed([content])[0]

        if not tier:
            item.tier = self.tiers.assign_tier(item, {'tier': tier})
        await self.store.put(item)
        await self.vectors.add(item.id, item.embedding, {'tier': item.tier.value, 'content': content[:500]})
        if item.entities:
            await self.kg.add_memory_entities(item)
        if item.tier == Tier.WORKING:
            self.tiers.working_buffer.append(item)
            if len(self.tiers.working_buffer) > settings.working_capacity:
                evicted = self.tiers.working_buffer.pop(0)
                evicted.tier = Tier.EPISODIC
                await self.store.update_tier(evicted.id, Tier.EPISODIC)
        return item

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        item = await self.store.get(memory_id)
        if item:
            item.touch()
            await self.store.put(item)
        return item

    async def recall(self, query: str, k: int = 10, tier_filter: List[str] = None) -> List[Dict]:
        results = await self.retriever.search(query, k=k, tier_filter=tier_filter)
        for r in results[:3]:
            mem = await self.store.get(r['id'])
            if mem:
                mem.touch()
                await self.store.put(mem)
        return results

    async def forget_expired(self) -> int:
        items = await self.store.list_all()
        count = 0
        for item in items:
            action = self.tiers.should_demote_or_forget(item)
            promo = self.tiers.should_promote(item)
            if promo:
                await self.store.update_tier(item.id, promo)
            if action == "forget":
                await self.store.delete(item.id)
                await self.vectors.delete(item.id)
                count += 1
        return count

    async def consolidate(self):
        episodic = await self.store.list_by_tier(Tier.EPISODIC)
        for item in episodic:
            if item.forgetting.rehearsals >= settings.semantic_consolidation_threshold and item.forgetting.retention() > 0.6:
                await self.store.update_tier(item.id, Tier.SEMANTIC)
