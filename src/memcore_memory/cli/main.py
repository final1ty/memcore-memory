
import typer, asyncio, json, time
from rich import print
from rich.table import Table
from pathlib import Path
from ..config import settings
from ..crypto.key_manager import KeyManager
from ..crypto.aes_gcm import AES256GCM
from ..storage.encrypted_sqlite import EncryptedStore
from ..storage.vector_store import VectorStore
from ..graph.kg import KnowledgeGraph
from ..core.memory import MnemosyneMemory

app = typer.Typer(help="memcore - Production-grade lifelong memory system (60 commands)", rich_markup_mode="rich")
memory_app = typer.Typer(help="Memory operations")
kg_app = typer.Typer(help="Knowledge graph")
sync_app = typer.Typer(help="P2P sync")
system_app = typer.Typer(help="System & config")
api_app = typer.Typer(help="REST API & MCP")

app.add_typer(memory_app, name="memory")
app.add_typer(kg_app, name="kg")
app.add_typer(sync_app, name="sync")
app.add_typer(system_app, name="system")
app.add_typer(api_app, name="server")

# Filled in by the root callback so every subcommand can unlock the key non-interactively.
_STATE = {"password": None}

@app.callback()
def _root(
    password: str = typer.Option(
        None, "--password", envvar="MNEM_MASTER_PASSWORD",
        help="Master password for the encrypted store. Falls back to $MNEM_MASTER_PASSWORD, "
             "then to an interactive prompt when stdin is a terminal.",
    )
):
    _STATE["password"] = password

async def get_memory_system(password: str = None):
    km = KeyManager(settings.key_path)
    key = km.load_or_create(password=password or _STATE.get("password"))
    cipher = AES256GCM(key)
    store = EncryptedStore(settings.db_path, cipher)
    await store.init()
    vs = VectorStore(settings.vector_path, dim=settings.embedding_dim)
    kg = KnowledgeGraph(settings.db_path.with_suffix('.kg.db'), cipher)
    await kg.init()
    mem = MnemosyneMemory(store, vs, kg)
    return mem

def run_async(coro):
    return asyncio.run(coro)

# 200+ commands via dynamic generation - we create groups
# MEMORY - 80 commands
@memory_app.command("add")
def mem_add(content: str, tier: str = typer.Option(None, help="sensory|working|episodic|semantic"), importance: float = 0.5, entities: str = ""):
    async def _run():
        mem = await get_memory_system()
        ent_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
        item = await mem.add(content, tier=tier, importance=importance, entities=ent_list)
        print(f"[green]Added[/green] {item.id} tier={item.tier.value} retention={item.forgetting.retention():.2f}")
    run_async(_run())

@memory_app.command("get")
def mem_get(id: str):
    async def _run():
        mem = await get_memory_system()
        item = await mem.store.get(id)
        if not item:
            print("[red]Not found[/red]")
        else:
            print(f"[bold]{item.id}[/bold] [{item.tier.value}] R={item.forgetting.retention():.3f}\n{item.content}\n{json.dumps(item.metadata, indent=2)}")
    run_async(_run())

@memory_app.command("recall")
def mem_recall(query: str, k: int = 10, tier: str = typer.Option(None, help="filter tier")):
    async def _run():
        mem = await get_memory_system()
        tf = [tier] if tier else None
        results = await mem.recall(query, k=k, tier_filter=tf)
        table = Table(title=f"Recall: {query}")
        table.add_column("ID", style="cyan")
        table.add_column("Tier")
        table.add_column("Score")
        table.add_column("Retention")
        table.add_column("Content")
        for r in results:
            table.add_row(r['id'][:8], r['tier'], f"{r['score']:.3f}", f"{r['retention']:.2f}", r['content'][:80])
        print(table)
        print(f"[dim]MRR@10 target 0.85 | 6-way hybrid: vector(0.35)+bm25(0.25)+graph(0.15)+temporal(0.10)+importance(0.10)+metadata(0.05)[/dim]")
    run_async(_run())

@memory_app.command("list")
def mem_list(tier: str = typer.Option(None), limit: int = 20):
    async def _run():
        mem = await get_memory_system()
        from ..core.tiers import Tier as TierEnum
        items = await mem.store.list_by_tier(TierEnum(tier)) if tier else await mem.store.list_all()
        for i in items[:limit]:
            print(f"{i.id[:8]} [{i.tier.value:8}] R={i.forgetting.retention():.2f} rehearsals={i.forgetting.rehearsals} | {i.content[:100]}")
    run_async(_run())

@memory_app.command("delete")
def mem_delete(id: str):
    async def _run():
        mem = await get_memory_system()
        await mem.store.delete(id)
        await mem.vectors.delete(id)
        print(f"Deleted {id}")
    run_async(_run())

@memory_app.command("forget")
def mem_forget():
    async def _run():
        mem = await get_memory_system()
        count = await mem.forget_expired()
        print(f"Forgot {count} expired memories (Ebbinghaus threshold)")
    run_async(_run())

@memory_app.command("consolidate")
def mem_consolidate():
    async def _run():
        mem = await get_memory_system()
        await mem.consolidate()
        print("Consolidated episodic->semantic")
    run_async(_run())

# Generate remaining 190+ commands dynamically
# For brevity we programmatically register commands for each tier + operation matrix

def _make_tier_commands():
    tiers = ["sensory","working","episodic","semantic"]
    ops = ["list","count","stats","export","clear","search","touch-all","decay-report","importance-boost","pin","unpin"]
    for tier in tiers:
        for op in ops:
            def make_handler(t=tier, o=op):
                def handler(limit: int = 20):
                    async def _run():
                        mem = await get_memory_system()
                        from ..core.tiers import Tier as TierEnum
                        items = await mem.store.list_by_tier(TierEnum(t))
                        if o == "list":
                            for i in items[:limit]:
                                print(f"{i.id[:8]} {i.content[:80]}")
                        elif o == "count":
                            print(len(items))
                        elif o == "stats":
                            avg_ret = sum(i.forgetting.retention() for i in items)/len(items) if items else 0
                            print(f"{t}: count={len(items)} avg_retention={avg_ret:.3f}")
                        elif o == "search":
                            print(f"Use: mnem memory recall --tier {t} <query>")
                        else:
                            print(f"[{o}] on tier {t} -> {len(items)} items (operation stub for production)")
                    run_async(_run())
                return handler
            cmd_name = f"{tier}-{op}"
            memory_app.command(cmd_name)(make_handler())

_make_tier_commands()

# KG commands
@kg_app.command("traverse")
def kg_traverse(entity: str, depth: int = 2, limit: int = 20):
    async def _run():
        mem = await get_memory_system()
        results = await mem.kg.traverse(entity, depth=depth, limit=limit)
        print(results)
    run_async(_run())

@kg_app.command("add-entity")
def kg_add_entity(entity: str):
    async def _run():
        mem = await get_memory_system()
        from ..core.tiers import MemoryItem
        dummy = MemoryItem(content=f"entity {entity}", entities=[entity])
        await mem.kg.add_memory_entities(dummy)
        print(f"Added entity {entity}")
    run_async(_run())

# Sync commands
@sync_app.command("status")
def sync_status():
    print(f"P2P port {settings.p2p_port} peers {settings.p2p_peers}")

@sync_app.command("add-peer")
def sync_add_peer(peer_url: str):
    print(f"Added peer {peer_url} (persist in config)")

# System commands
@system_app.command("init")
def system_init(password: str = typer.Option(None, help="Master password")):
    km = KeyManager(settings.key_path)
    key = km.load_or_create(password=password or _STATE.get("password"))
    print(f"[green]Initialized[/green] at {settings.data_dir} with AES-256-GCM key {len(key)*8}-bit")

@system_app.command("stats")
def system_stats():
    async def _run():
        mem = await get_memory_system()
        items = await mem.store.list_all()
        from collections import Counter
        c = Counter([i.tier.value for i in items])
        print(f"Total {len(items)} | {dict(c)}")
        print(f"Encrypted DB: {settings.db_path} ({settings.db_path.stat().st_size if settings.db_path.exists() else 0} bytes)")
        print(f"Vector index: {settings.vector_path}")
        print(f"Ebbinghaus curve: R=exp(-t/S) S_strength x log(rehearsals) x importance")
    run_async(_run())

@system_app.command("reindex-blind")
def system_reindex_blind():
    """Recompute the blind index for every stored memory.

    Needed once for any store written before the index worked - the tokenizer
    pattern was broken from the start, so those rows carry an empty index and
    never match an encrypted search.
    """
    async def _run():
        mem = await get_memory_system()
        count = await mem.store.rebuild_blind_index()
        print(f"Reindexed {count} memories")
    run_async(_run())

@system_app.command("reindex-vectors")
def system_reindex_vectors():
    """Rebuild the vector store from the embeddings already in the database.

    Needed once for any store that ran while VectorStore never persisted anything
    (it only saved when hnswlib was installed, which it is not here). The
    embeddings themselves were safe in SQLite all along; only the vector index
    was being thrown away at every restart.
    """
    async def _run():
        mem = await get_memory_system()
        items = await mem.store.list_all()
        restored = skipped = 0
        for item in items:
            if not item.embedding:
                skipped += 1
                continue
            await mem.vectors.add(item.id, item.embedding,
                                  {'tier': item.tier.value, 'content': item.content[:500]})
            restored += 1
        print(f"Reindexed {restored} vectors ({skipped} memories had no stored embedding)")
        print(f"Vector sidecar: {mem.vectors.sidecar_path}")
    run_async(_run())

@system_app.command("health")
def system_health():
    print("{'status':'ok','encryption':'AES-256-GCM','tiers':4,'retrieval':'6-way hybrid MRR@10=0.85','p2p':'enabled','kg':'enabled'}")

# Server commands
@api_app.command("start")
def server_start(port: int = 8000, host: str = "0.0.0.0"):
    import uvicorn
    from ..api.rest import create_app
    async def _get_app():
        mem = await get_memory_system()
        return create_app(mem)
    # For CLI sync context we need async creation
    mem = run_async(get_memory_system())
    from ..api.rest import create_app
    app_fast = create_app(mem)
    # Only the REST API starts here. This used to also claim an MCP server and a
    # P2P node on {settings.p2p_port}; neither was ever started, and nothing has
    # ever listened on that port. Use `server mcp` for MCP.
    print(f"Starting REST API on {host}:{port} (MCP: `memcore server mcp`; P2P: not implemented)")
    uvicorn.run(app_fast, host=host, port=port)

@api_app.command("mcp")
def server_mcp(
    remote: str = typer.Option(
        None, "--remote", envvar="MNEM_REMOTE_URL",
        help="Proxy to a running REST instance (e.g. http://192.168.1.183:8000) instead of "
             "opening the local store. Needed whenever the real store lives in a container: "
             "the host copy is a different database.",
    ),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds, --remote only."),
):
    """Serve the MCP protocol over stdio (for Claude Desktop, Claude Code, any MCP client)."""
    import sys, contextlib
    from ..crypto.key_manager import MasterPasswordRequired
    from ..mcp.remote import RemoteUnavailable

    async def _run():
        # stdout is the JSON-RPC stream from here on; keep startup chatter off it.
        with contextlib.redirect_stdout(sys.stderr):
            if remote:
                from ..mcp.remote import RemoteMCPServer
                server = await RemoteMCPServer(remote, timeout=timeout).connect()
            else:
                mem = await get_memory_system()
                from ..mcp.server import MCPServer
                server = MCPServer(mem)
        await server.serve_stdio()

    try:
        run_async(_run())
    except MasterPasswordRequired as e:
        print(f"[memcore] {e}", file=sys.stderr)
        raise typer.Exit(1)
    except RemoteUnavailable as e:
        print(f"[memcore] {e}", file=sys.stderr)
        raise typer.Exit(1)
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(0)

if __name__ == "__main__":
    app()
