
__version__ = "1.0.0"
from .config import settings
from .core.memory import MnemosyneMemory
from .storage.encrypted_sqlite import EncryptedStore
from .storage.vector_store import VectorStore
from .graph.kg import KnowledgeGraph
from .crypto.key_manager import KeyManager
from .crypto.aes_gcm import AES256GCM
from .embeddings.factory import get_embedder

async def create_memory_system(password: str = None, backend: str = None, embedding_provider: str = None):
    backend = backend or settings.backend
    embedder = get_embedder(
        provider=embedding_provider or settings.embedding_provider,
        dim=settings.embedding_dim,
        device=getattr(settings, 'embedding_device', None)
    )

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
        # KG can also use postgres - reuse same DB for simplicity
        kg = KnowledgeGraph(kg_store_path, cipher)
        await kg.init()
    else:
        store = EncryptedStore(settings.db_path, cipher)
        await store.init()
        vs = VectorStore(settings.vector_path, dim=embedder.dim)
        kg = KnowledgeGraph(settings.db_path.with_suffix('.kg.db'), cipher)
        await kg.init()

    return MnemosyneMemory(store, vs, kg, embedder=embedder)
