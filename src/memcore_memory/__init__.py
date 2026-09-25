
__version__ = "1.0.0"
import sys
from .config import settings, normalise_backend
from .core.memory import MnemosyneMemory
from .storage.encrypted_sqlite import EncryptedStore
from .storage.vector_store import VectorStore, embedder_identity
from .graph.kg import KnowledgeGraph, KnowledgeGraphUnreadable
from .crypto.key_manager import KeyManager
from .crypto.aes_gcm import AES256GCM
from .embeddings.factory import get_embedder

class _UnavailableGraph:
    """Stands in for a knowledge graph that could not be opened.

    One damaged or unmigratable memory.kg.db used to stop REST, MCP and the CLI
    from starting at all, though memory.db was fine. The memories are served
    without it now: writes skip the graph (the links are rebuilt from the store
    once a fresh graph file is started), and every graph read raises the reason,
    so the graph arm of retrieval reports its failure instead of finding nothing.
    """

    created_fresh = False
    needs_link_backfill = False

    def __init__(self, db_path, error: Exception):
        self.db_path = db_path
        self.unavailable_reason = str(error)

    def _fail(self, *a, **k):
        raise KnowledgeGraphUnreadable(self.unavailable_reason)

    async def add_memory_entities(self, item):
        return None

    async def remove_memory(self, memory_id: str) -> int:
        return 0

    async def get_related_memories(self, query: str, limit: int = 20):
        self._fail()

    async def traverse(self, *a, **k):
        self._fail()

    async def add_entity(self, *a, **k):
        self._fail()

    async def add_relation(self, *a, **k):
        self._fail()

    async def list_entities(self, *a, **k):
        self._fail()

    async def delete_entity(self, *a, **k):
        self._fail()


async def _open_graph(path, cipher, store):
    kg = KnowledgeGraph(path, cipher)
    try:
        await kg.init()
    except KnowledgeGraphUnreadable as e:
        print(f"[kg] {e}", file=sys.stderr)
        print("[kg] continuing WITHOUT the knowledge graph: memories are unaffected, but graph "
              "search and the entity tools fail until the file is repaired or moved aside "
              "(a new one is then rebuilt from the memories at the next start)", file=sys.stderr)
        return _UnavailableGraph(path, e)
    if kg.needs_link_backfill:
        # A graph migrated from v1 has no links for memories tagged with a single
        # entity, and a new file has none at all; without this they stayed
        # invisible to graph search until someone ran rebuild_links by hand.
        # rebuild_links is idempotent and marks the graph complete, so this runs
        # once per file. A failure is retried at the next start.
        try:
            items = await store.list_all()
            if kg.created_fresh and items:
                print(f"[kg] WARNING: {path} did not exist and has been created empty for a store "
                      f"holding {len(items)} memories; rebuilding memory-entity links from them. "
                      f"Entities and relations added by hand, not through a memory, are gone.",
                      file=sys.stderr)
            await kg.rebuild_links(items)
        except Exception as e:  # noqa: BLE001 - the memories must still open
            print(f"[kg] memory links not rebuilt ({type(e).__name__}: {e}); retrying at next start",
                  file=sys.stderr)
    return kg


async def create_memory_system(password: str = None, backend: str = None, embedding_provider: str = None):
    # Same closed set Settings enforces for MNEM_BACKEND: an unknown value used to
    # fall through to the SQLite branch without a word.
    backend = normalise_backend(backend or settings.backend)
    if backend not in ("sqlite", "postgres"):
        raise ValueError(f"unknown backend {backend!r}; expected 'sqlite' or 'postgres'")
    provider = embedding_provider or settings.embedding_provider
    from .embeddings.factory import MODEL_MAP
    if provider not in MODEL_MAP and "/" not in provider:
        # Direct HuggingFace ids ("org/model") are supported; a bare unknown name
        # is almost always a typo, and it ends in the hash fallback.
        print(f"[config] warning: embedding provider {provider!r} is not one of "
              f"{sorted(MODEL_MAP)} and not a HuggingFace id", file=sys.stderr)
    embedder = get_embedder(
        provider=provider,
        dim=settings.embedding_dim,
        device=getattr(settings, 'embedding_device', None)
    )

    settings.ensure_data_dir()

    km = KeyManager(settings.key_path)
    key = km.load_or_create(password=password)
    cipher = AES256GCM(key)

    if backend == "postgres":
        from .storage.postgres import PostgresStore
        store = PostgresStore(settings.database_url, cipher, embedding_dim=embedder.dim)
        await store.init()
        # For postgres, vector search is inside store.vector_search, but we keep VectorStore wrapper for compatibility
        # Use PostgresStore as both store and vector store adapter
        class PgVectorAdapter:
            def __init__(self, pg_store):
                self.pg_store = pg_store
                self.dim = pg_store.dim
            async def add(self, mem_id, embedding, meta):
                pass  # already in pg_store.put
            async def delete(self, mem_id):
                pass
            async def search(self, q_emb, k=10):
                return await self.pg_store.vector_search(q_emb, k=k)
        vs = PgVectorAdapter(store)
        kg_store_path = settings.db_path.with_suffix('.kg.db') if hasattr(settings.db_path, 'with_suffix') else settings.data_dir / "kg.db"
        # The graph stays in a local SQLite file even on Postgres: each instance keeps its own.
        kg = await _open_graph(kg_store_path, cipher, store)
    else:
        store = EncryptedStore(settings.db_path, cipher)
        await store.init()
        vs = VectorStore(settings.vector_path, dim=embedder.dim, embedder_id=embedder_identity(embedder))
        kg = await _open_graph(settings.db_path.with_suffix('.kg.db'), cipher, store)

    return MnemosyneMemory(store, vs, kg, embedder=embedder)
