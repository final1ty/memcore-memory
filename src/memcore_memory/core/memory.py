
import sys
from typing import List, Optional, Dict, Any
from .tiers import MemoryItem, Tier, TierManager, TIER_BASE_STRENGTH, validate_importance
from ..storage.encrypted_sqlite import EncryptedStore
from ..storage.vector_store import VectorStore, embedder_identity
from ..graph.kg import KnowledgeGraph
from ..retrieval.hybrid import HybridRetriever, is_semantic
from ..config import settings
from ..embeddings.factory import get_embedder

# How many of a recall's matched results count as rehearsed.
RECALL_REHEARSE_TOP = 3


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
        # This used to overwrite vector_store.dim after the store had loaded (and
        # dimension-checked) its vectors, so an embedder of another width silently
        # mixed two vector spaces in one index.
        store_dim = getattr(vector_store, 'dim', None)
        if store_dim is not None and store_dim != self.embedder.dim:
            raise ValueError(
                f"embedder produces {self.embedder.dim}-dim vectors but the vector store "
                f"is {store_dim}-dim; set MNEM_EMBEDDING_DIM to match the model and re-embed")
        # Callers should pass embedder_id to VectorStore so the sidecar is checked
        # as it loads (create_memory_system does); this covers the ones that don't.
        if getattr(vector_store, 'embedder_id', None) is None:
            vector_store.embedder_id = embedder_identity(self.embedder)
        # Set by create_memory_system when memory.kg.db could not be opened: the
        # memories are served without the graph rather than not at all.
        self.kg_error = getattr(kg, 'unavailable_reason', None)
        self.retriever = HybridRetriever(store, vector_store, kg, embedder=self.embedder)

    async def add(self, content: str, metadata: Dict[str, Any] = None, tier: str = None, importance: float = 0.5, entities: List[str] = None) -> MemoryItem:
        importance = validate_importance(importance)
        if not entities and settings.auto_extract_entities:
            # Opt-in: on by default it would change graph ranking in existing stores.
            from ..graph.extractor import extract_entities
            entities = extract_entities(content)
        # A copy: the caller's dict used to be mutated and shared between items.
        metadata = {**(metadata or {}), 'importance': importance}
        item = MemoryItem(content=content, metadata=metadata, tier=Tier(tier) if tier else Tier.EPISODIC, entities=entities or [])
        if not tier:
            item.tier = self.tiers.assign_tier(item, {'tier': tier})
            # The curve was seeded for the placeholder tier; a new item has nothing
            # earned to lose, so give it the baseline of the tier it actually got.
            item.forgetting.strength = TIER_BASE_STRENGTH.get(item.tier, 7.0)
        if not item.embedding:
            # A stored memory is a document, not a query: E5 needs "passage:" here.
            item.embedding = (self.embedder.embed_documents([content])[0]
                              if hasattr(self.embedder, 'embed_documents')
                              else self.embedder.embed([content])[0])

        expected = getattr(self.vectors, 'dim', None)
        if expected is not None and len(item.embedding) != expected:
            raise ValueError(f"embedding has {len(item.embedding)} dimensions, vector store expects {expected}")
        await self.store.put(item)
        try:
            await self.vectors.add(item.id, item.embedding, {'tier': item.tier.value})
            if item.entities:
                await self.kg.add_memory_entities(item)
        except Exception:
            # A row without its vector or graph links used to stay behind, and a
            # caller retrying the failed add duplicated it.
            await self._purge(item.id)
            raise
        # Nothing schedules the lifecycle pass, so expired working memories would
        # otherwise sit there reporting a near-zero retention (their curve is the
        # 20-minute working one) until someone calls it. The memory is stored by
        # now; failing the add here would invite a retry that duplicates it.
        try:
            await self._maintain_working(keep=item.id)
        except Exception as e:  # noqa: BLE001
            print(f"[memory] working tier not maintained: {type(e).__name__}: {e}", file=sys.stderr)
        return item

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory with its vector and graph links. True iff it existed."""
        return await self._purge(memory_id)

    async def _purge(self, memory_id: str) -> bool:
        # The row goes last: if removing the vector or the links fails, the memory
        # is still there and a retry finishes the job, rather than reporting it
        # gone while pieces of it remain searchable.
        await self.vectors.delete(memory_id)
        await self.kg.remove_memory(memory_id)
        return bool(await self.store.delete(memory_id))

    async def _maintain_working(self, keep: str = None) -> int:
        """Demote expired working memories, then enforce the cap. Returns how many moved."""
        demoted = 0
        for w in await self.store.list_by_tier(Tier.WORKING):
            if w.id != keep and self.tiers.should_demote_or_forget(w) == "demote":
                if await self.store.update_tier(w.id, Tier.EPISODIC):
                    demoted += 1
        return demoted + await self._enforce_working_capacity(keep=keep)

    async def _enforce_working_capacity(self, keep: str = None) -> int:
        """Demote the oldest working memories beyond working_capacity to episodic.

        The cap used to be an in-process list: it started empty in every process,
        so each CLI call and each restart admitted seven more, and it demoted
        whatever it had buffered even after that memory had been promoted or
        deleted. The store is the only thing every process agrees on.
        """
        working = await self.store.list_by_tier(Tier.WORKING)
        overflow = len(working) - settings.working_capacity
        if overflow <= 0:
            return 0
        victims = sorted((w for w in working if w.id != keep),
                         key=lambda w: (w.timestamp, w.id))[:overflow]
        for v in victims:
            await self.store.update_tier(v.id, Tier.EPISODIC)  # raises strength to the floor
        return len(victims)

    async def get(self, memory_id: str) -> Optional[MemoryItem]:
        item = await self.store.get(memory_id)
        if item:
            item.touch()
            # Only the curve: writing the whole row back would resurrect a memory
            # deleted in the meantime and overwrite a concurrent edit.
            if not await self.store.update_forgetting(item.id, item.forgetting):
                return None  # deleted between the read and the write
        return item

    async def recall(self, query: str, k: int = 10, tier_filter: List[str] = None, rehearse: bool = True) -> List[Dict]:
        """Search, and count the top results as rehearsed.

        Rehearsal is what makes the curve strengthen with use, so it stays on by
        default. It is bounded to the top RECALL_REHEARSE_TOP results, each
        strengthened less the lower it ranked, and only the best hit adds to the
        rehearsal count that tier promotion looks at. The retention returned is
        the value after the rehearsal, not before it.

        What "a hit" means is up to the retriever: HybridRetriever only returns
        memories a query-dependent arm nominated, but under a semantic embedder
        the vector arm nominates the nearest neighbours of any query, relevant or
        not. A result marked ``matched: False`` is never rehearsed; the retriever
        does not set that yet, so the rank rules above are what bound it today.
        """
        results = await self.retriever.search(query, k=k, tier_filter=tier_filter)
        if not rehearse:
            return results
        top = [r for r in results if r.get('matched', True)][:RECALL_REHEARSE_TOP]
        if not top:
            return results
        items = await self.store.get_many([r['id'] for r in top])
        for rank, r in enumerate(top):
            mem = items.get(r['id'])
            if mem is None:
                continue
            mem.forgetting.rehearse(feedback=1.0 / (rank + 1), count=rank == 0)
            if await self.store.update_forgetting(mem.id, mem.forgetting):
                r['retention'] = mem.forgetting.retention()
        return results

    async def lifecycle_pass(self) -> Dict[str, Any]:
        """Apply the tier lifecycle (see TierManager) to every memory.

        Each memory is handled on its own: a bad row is reported and skipped
        instead of aborting the pass after earlier deletions had already been
        committed, which left the caller with a 500 and no count.
        """
        report = {"forgotten": 0, "demoted": 0, "promoted": 0, "errors": []}
        scan = getattr(self.store, 'scan', None)
        if scan is not None:
            # list_all() drops rows it cannot decode with only a stderr line, so a
            # caller of the pass never learned that a memory had been skipped.
            items, unreadable = await scan()
            for mid, reason in unreadable.items():
                print(f"[lifecycle] skipped {mid}: {reason}", file=sys.stderr)
                report["errors"].append({"id": mid, "error": f"unreadable: {reason}"})
        else:
            items = await self.store.list_all()
        for item in items:
            try:
                # Promotion first, and it wins: judging deletion on the state before
                # promotion deleted the very memories that had just earned a move up.
                promo = self.tiers.should_promote(item)
                if promo is not None:
                    await self.store.update_tier(item.id, promo)
                    report["promoted"] += 1
                    continue
                action = self.tiers.should_demote_or_forget(item)
                if action == "demote":
                    await self.store.update_tier(item.id, Tier.EPISODIC)
                    report["demoted"] += 1
                elif action == "forget":
                    if await self._purge(item.id):
                        report["forgotten"] += 1
            except Exception as e:  # noqa: BLE001 - reported, and the pass goes on
                print(f"[lifecycle] skipped {item.id}: {type(e).__name__}: {e}", file=sys.stderr)
                report["errors"].append({"id": item.id, "error": f"{type(e).__name__}: {e}"})
        # Sensory promotions can push working over its cap, and so can a store
        # written before the cap was enforced against the store.
        try:
            report["demoted"] += await self._enforce_working_capacity()
        except Exception as e:  # noqa: BLE001
            print(f"[lifecycle] working capacity not enforced: {type(e).__name__}: {e}", file=sys.stderr)
            report["errors"].append({"id": None, "error": f"{type(e).__name__}: {e}"})
        return report

    async def forget_expired(self) -> int:
        """Run the lifecycle pass and return how many memories were deleted.

        Kept as an int for existing callers; lifecycle_pass() has the full report,
        including promotions, demotions and any rows that could not be processed.
        """
        return (await self.lifecycle_pass())["forgotten"]

    async def consolidate(self) -> int:
        """Promote episodic memories that meet the semantic rule. Returns how many.

        Uses TierManager.should_promote, so there is one promotion rule rather than
        the three this used to have between here, forget_expired and assign_tier.
        """
        promoted = 0
        for item in await self.store.list_by_tier(Tier.EPISODIC):
            try:
                if self.tiers.should_promote(item) == Tier.SEMANTIC:
                    await self.store.update_tier(item.id, Tier.SEMANTIC)
                    promoted += 1
            except Exception as e:  # noqa: BLE001 - one bad row must not stop the rest
                print(f"[consolidate] skipped {item.id}: {type(e).__name__}: {e}", file=sys.stderr)
        return promoted
