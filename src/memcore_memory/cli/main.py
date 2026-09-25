
import typer, asyncio, json, os, sys
from typing import Optional
from rich import print
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from pathlib import Path
from ..config import settings
from ..core.tiers import Tier
from ..crypto.key_manager import KeyManager, MasterKeyError
from ..graph.kg import KnowledgeGraphUnreadable

# The help text used to carry a hand-written command count that went stale the
# moment a command was added; it is filled in from the command tree at the bottom.
app = typer.Typer(help="memcore - Production-grade lifelong memory system", rich_markup_mode="rich")
memory_app = typer.Typer(help="Memory operations")
kg_app = typer.Typer(help="Knowledge graph")
sync_app = typer.Typer(help="P2P sync (not implemented)")
system_app = typer.Typer(help="System & config")
api_app = typer.Typer(help="REST API & MCP")

app.add_typer(memory_app, name="memory")
app.add_typer(kg_app, name="kg")
app.add_typer(sync_app, name="sync")
app.add_typer(system_app, name="system")
app.add_typer(api_app, name="server")

# Stored memories go through this, never through rich's print: rich parses
# `[word]` as markup, so bracketed text vanished and a stray `[/x]` in one memory
# raised MarkupError for every list and recall that returned it. soft_wrap keeps
# long lines whole instead of breaking them at the terminal width.
_out = Console(markup=False, highlight=False, soft_wrap=True)

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
    # The library factory, not a copy of it: the copy never read MNEM_BACKEND, so
    # with Postgres configured the CLI, `server start` and local `server mcp` all
    # quietly served a SQLite file instead.
    from .. import create_memory_system
    return await create_memory_system(password=password or _STATE.get("password"))

def _fail(message: str):
    # Plain write: through rich, "[memcore]" and any bracket in the message were
    # parsed as markup and dropped.
    sys.stderr.write(f"[memcore] {message}\n")
    raise typer.Exit(1)

def run_async(coro):
    # Everything a user can cause - a missing password, an unreadable key, a bad
    # value, a backend whose driver isn't installed - used to end in a hundred-line
    # traceback with the one useful line at the bottom.
    try:
        return asyncio.run(coro)
    except (MasterKeyError, KnowledgeGraphUnreadable) as e:
        _fail(str(e))
    except (ValueError, ImportError, OSError, asyncio.TimeoutError) as e:
        # OSError covers an unreachable Postgres (refused, unknown host).
        _fail(f"{type(e).__name__}: {e}")

def _newest_first(items):
    # The tier listing comes back in row order, not by time, so `--limit N` used
    # to show the N rows written longest ago.
    return sorted(items, key=lambda i: i.timestamp, reverse=True)

@memory_app.command("add")
def mem_add(content: str,
            tier: Optional[Tier] = typer.Option(None, help="Tier; assigned automatically when omitted"),
            importance: float = typer.Option(0.5, min=0.0, max=1.0),
            entities: str = ""):
    async def _run():
        mem = await get_memory_system()
        ent_list = [e.strip() for e in entities.split(",") if e.strip()] if entities else []
        item = await mem.add(content, tier=tier.value if tier else None,
                             importance=importance, entities=ent_list)
        print(f"[green]Added[/green] {item.id} tier={item.tier.value} retention={item.forgetting.retention():.2f}")
    run_async(_run())

@memory_app.command("get")
def mem_get(id: str):
    async def _run():
        mem = await get_memory_system()
        # store.get, not mem.get: reading a memory here is not a rehearsal.
        return await mem.store.get(id)
    item = run_async(_run())
    if not item:
        _fail(f"Not found: {id}")
    print(f"[bold]{escape(item.id)}[/bold] {escape(f'[{item.tier.value}]')} R={item.forgetting.retention():.3f}")
    _out.print(item.content)
    _out.print(json.dumps(item.metadata, indent=2, ensure_ascii=False))

@memory_app.command("recall")
def mem_recall(query: str, k: int = 10, tier: Optional[Tier] = typer.Option(None, help="filter tier")):
    async def _run():
        mem = await get_memory_system()
        tf = [tier.value] if tier else None
        results = await mem.recall(query, k=k, tier_filter=tf)
        table = Table(title=Text(f"Recall: {query}"))
        table.add_column("ID", style="cyan")
        table.add_column("Tier")
        table.add_column("Score")
        table.add_column("Retention")
        table.add_column("Content")
        for r in results:
            table.add_row(r['id'][:8], r['tier'], f"{r['score']:.3f}", f"{r['retention']:.2f}",
                          Text(r['content'][:80]))
        print(table)
        # The weights in effect, not the defaults: under the hash embedder the
        # vector share is redistributed, and the old constant footer said otherwise.
        w = mem.retriever.weights
        parts = " + ".join(f"{name}({v:.2f})" for name, v in sorted(w.items(), key=lambda kv: -kv[1]) if v > 0)
        note = "" if w.get('vector', 0) > 0 else " | vector off: embedder is not semantic"
        _out.print(f"hybrid RRF weights: {parts}{note}", style="dim")
    run_async(_run())

@memory_app.command("search-blind")
def mem_search_blind(query: str, k: int = typer.Option(10, min=1, help="Maximum results")):
    """Keyword search over the encrypted rows' blind index, ranked by matching terms.

    Exact keywords only: no stemming, no similarity, and nothing is rehearsed.
    """
    async def _run():
        mem = await get_memory_system()
        search = getattr(mem.store, 'search_by_blind_index', None)
        if search is None:
            raise ValueError(f"the {settings.backend} backend has no blind index")
        if getattr(mem.store, 'blind', True) is None:
            raise ValueError("the blind index is disabled (MNEM_BLIND_INDEX_ENABLED=false)")
        ids = await search(query, limit=k)
        return ids, await mem.store.get_many(ids) if ids else {}
    ids, found = run_async(_run())
    for mid in ids:
        item = found.get(mid)
        if item is not None:   # deleted between the search and the read
            _out.print(f"{mid[:8]} [{item.tier.value:8}] | {item.content[:100]}")
    if not ids:
        sys.stderr.write("[memcore] no match; rows written before the index worked need "
                         "`memcore system reindex-blind`\n")

@memory_app.command("list")
def mem_list(tier: Optional[Tier] = typer.Option(None), limit: int = 20):
    async def _run():
        mem = await get_memory_system()
        items = await mem.store.list_by_tier(tier) if tier else await mem.store.list_all()
        for i in _newest_first(items)[:limit]:
            _out.print(f"{i.id[:8]} [{i.tier.value:8}] R={i.forgetting.retention():.2f} "
                       f"rehearsals={i.forgetting.rehearsals} | {i.content[:100]}")
    run_async(_run())

@memory_app.command("delete")
def mem_delete(id: str):
    async def _run():
        mem = await get_memory_system()
        # The row, its vector and its graph links together; this used to remove
        # the row and the vector, and report success for ids that never existed.
        return await mem.delete(id)
    if not run_async(_run()):
        _fail(f"Not found: {id}")
    _out.print(f"Deleted {id}")

@memory_app.command("forget")
def mem_forget():
    async def _run():
        mem = await get_memory_system()
        return await mem.lifecycle_pass()
    r = run_async(_run())
    _out.print(f"Forgot {r['forgotten']} expired memories (Ebbinghaus threshold); "
               f"demoted {r['demoted']}, promoted {r['promoted']}")
    # Rows the pass could not process used to vanish into a stderr log line
    # while the command reported success.
    for err in r['errors']:
        sys.stderr.write(f"[memcore] skipped {err['id']}: {err['error']}\n")
    if r['errors']:
        raise typer.Exit(1)

@memory_app.command("consolidate")
def mem_consolidate():
    async def _run():
        mem = await get_memory_system()
        return await mem.consolidate()
    n = run_async(_run())
    _out.print(f"Promoted {n} episodic->semantic")

# A command per tier and operation. Only list/count/stats/search do anything;
# the rest are stubs that say so.

def _make_tier_commands():
    tiers = ["sensory","working","episodic","semantic"]
    ops = ["list","count","stats","export","clear","search","touch-all","decay-report","importance-boost","pin","unpin"]
    for tier in tiers:
        for op in ops:
            def make_handler(t=tier, o=op):
                def handler(limit: int = 20):
                    async def _run():
                        mem = await get_memory_system()
                        items = await mem.store.list_by_tier(Tier(t))
                        if o == "list":
                            for i in _newest_first(items)[:limit]:
                                _out.print(f"{i.id[:8]} {i.content[:80]}")
                        elif o == "count":
                            print(len(items))
                        elif o == "stats":
                            avg_ret = sum(i.forgetting.retention() for i in items)/len(items) if items else 0
                            print(f"{t}: count={len(items)} avg_retention={avg_ret:.3f}")
                        elif o == "search":
                            print(f"Use: mnem memory recall --tier {t} <query>")
                        else:
                            _out.print(f"[{o}] on tier {t} -> {len(items)} items (operation stub for production)")
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
        _out.print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    run_async(_run())

@kg_app.command("add-entity")
def kg_add_entity(entity: str):
    async def _run():
        mem = await get_memory_system()
        # add_entity replaces the node, so re-adding an entity extracted with a
        # type and properties would reset both to the bare defaults.
        if await _entity_exists(mem.kg, entity):
            return False
        # add_entity, not add_memory_entities with a throwaway MemoryItem: the
        # graph links entities to memory ids, and that id belongs to no memory.
        await mem.kg.add_entity(entity)
        return True
    if run_async(_run()):
        _out.print(f"Added entity {entity}")
    else:
        _out.print(f"Entity {entity} already exists; left unchanged")

async def _entity_exists(kg, entity: str) -> bool:
    has = getattr(kg, 'has_entity', None)
    if has is not None:
        return await has(entity)
    reason = getattr(kg, 'unavailable_reason', None)
    if reason is not None:
        raise KnowledgeGraphUnreadable(reason)
    async with kg._connect() as db:
        async with db.execute('SELECT 1 FROM kg_nodes WHERE id=?', (kg._nid(entity),)) as cur:
            return await cur.fetchone() is not None

# Sync commands. P2P sync is not implemented: the CRDT primitives exist, but
# nothing transports them and nothing listens on the port.
@sync_app.command("status")
def sync_status():
    """Show the P2P configuration. There is no running node to report on."""
    _out.print(f"P2P sync: not running (not implemented). Configured port {settings.p2p_port}, "
               f"configured peers {list(settings.p2p_peers)} (from MNEM_P2P_PEERS; "
               f"nothing listens or connects).")

@sync_app.command("add-peer")
def sync_add_peer(peer_url: str):
    """Not implemented. This used to answer 'Added peer' and store nothing."""
    _fail("P2P sync is not implemented; nothing was stored. Peers can only be configured "
          "via MNEM_P2P_PEERS, and nothing uses them yet.")

# System commands
@system_app.command("init")
def system_init(password: str = typer.Option(None, help="Master password")):
    """Create the key and the store tables, or load them if they already exist.

    Harmless on an existing store: the key is loaded, never replaced. It used to
    say 'Initialized' either way and create no tables, so a run against the wrong
    data dir looked exactly like a healthy existing store.
    """
    pw = password or _STATE.get("password")
    existed = settings.key_path.exists()
    run_async(get_memory_system(pw))
    protected = KeyManager(settings.key_path).is_protected()
    state = "password-protected" if protected else "UNPROTECTED (`memcore system protect` wraps it)"
    _out.print(f"Data dir: {settings.data_dir.absolute()}")
    if existed:
        _out.print(f"Loaded existing key {settings.key_path} - {state}; nothing was replaced")
        # A password given for an existing raw key protects nothing; KeyManager
        # says so on stderr, and the line above no longer reads 'Initialized'.
    else:
        _out.print(f"Created new key {settings.key_path} - {state}")
    target = "MNEM_DATABASE_URL" if settings.backend == "postgres" else settings.db_path
    _out.print(f"Store ready ({settings.backend}): {target}")

@system_app.command("protect")
def system_protect(password: str = typer.Option(None, help="New master password")):
    """Wrap an unprotected master key with a password. The key itself does not change."""
    if not settings.key_path.exists():
        _fail(f"{settings.key_path} does not exist; nothing to protect")
    pw = password or _STATE.get("password")
    if not pw:
        if not sys.stdin.isatty():
            _fail("No password given. Pass --password or set MNEM_MASTER_PASSWORD.")
        pw = typer.prompt("New master password", hide_input=True, confirmation_prompt=True)
    try:
        KeyManager(settings.key_path).protect(pw)
    except (MasterKeyError, ValueError) as e:
        _fail(str(e))
    _out.print(f"Protected {settings.key_path}; the key itself is unchanged.")
    # The processes that matter most are the ones that do not restart often: a
    # server keeps its in-memory key until the next start, and then cannot open.
    sys.stderr.write("[memcore] Running processes (REST server, MCP server, container) keep working "
                     "with the key they hold, but every next start needs --password or "
                     "MNEM_MASTER_PASSWORD. Set it wherever they are launched before restarting them.\n")

@system_app.command("stats")
def system_stats():
    async def _run():
        mem = await get_memory_system()
        items = await mem.store.list_all()
        from collections import Counter
        c = Counter([i.tier.value for i in items])
        print(f"Total {len(items)} | {dict(c)}")
        _out.print(f"Encrypted DB: {settings.db_path} ({settings.db_path.stat().st_size if settings.db_path.exists() else 0} bytes)")
        _out.print(f"Vector index: {settings.vector_path}")
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

@system_app.command("reindex-kg")
def system_reindex_kg():
    """Rebuild the knowledge graph's memory-entity links from the stored memories.

    For a graph that lost its links, or a memory.kg.db started afresh. Entities
    and relations added by hand, without a memory, are not recreated.
    """
    async def _run():
        mem = await get_memory_system()
        reason = getattr(mem.kg, 'unavailable_reason', None)
        if reason is not None:
            raise KnowledgeGraphUnreadable(f"{reason} - move the file aside; a new one is rebuilt "
                                           f"from the memories at the next start")
        if hasattr(mem.store, 'scan'):
            items, unreadable = await mem.store.scan()
        else:
            items, unreadable = await mem.store.list_all(), {}
        return await mem.kg.rebuild_links(items), unreadable
    n, unreadable = run_async(_run())
    _out.print(f"Relinked {n} memories with entities")
    for mid, reason in unreadable.items():
        sys.stderr.write(f"[memcore] not relinked: {reason}\n")
    if unreadable:
        raise typer.Exit(1)

@system_app.command("verify")
def system_verify():
    """Decrypt every row and list the ones that fail. Read-only; exits 1 if any do.

    list and recall skip an unreadable row with a stderr line, so a damaged row
    can otherwise go unnoticed until the one memory it held is needed.
    """
    async def _run():
        mem = await get_memory_system()
        items, unreadable = await mem.store.scan()
        return len(items), unreadable
    readable, unreadable = run_async(_run())
    _out.print(f"{readable} of {readable + len(unreadable)} memories readable")
    for mid, reason in unreadable.items():
        _out.print(f"UNREADABLE {mid}: {reason}")
    if unreadable:
        raise typer.Exit(1)

@system_app.command("migrate-aad")
def system_migrate_aad(yes: bool = typer.Option(False, "--yes", help="Skip the confirmation")):
    """Bind every pre-AAD row's ciphertexts to its id. One-way; back up the store first.

    Until then an old row's encrypted columns can be swapped with another row's
    without detection. The release before AAD cannot read an upgraded row, so
    after this a rollback means restoring the backup, not starting the old image.
    """
    if not yes:
        if not sys.stdin.isatty():
            _fail("migrate-aad is one-way (older releases cannot read upgraded rows); "
                  "back up the store, then rerun with --yes")
        typer.confirm("Older releases cannot read upgraded rows. Backed up and continue?", abort=True)
    async def _run():
        mem = await get_memory_system()
        upgrade = getattr(mem.store, 'reencrypt_legacy_rows', None)
        if upgrade is None:
            raise ValueError(f"the {settings.backend} backend has no pre-AAD rows to upgrade")
        return await upgrade()
    _out.print(f"Upgraded {run_async(_run())} rows to id-bound encryption")

@system_app.command("reindex-vectors")
def system_reindex_vectors(
    reembed: bool = typer.Option(False, "--reembed", help="Recompute every embedding with the current embedder "
                                 "(needed after switching embedders, e.g. hash fallback -> bge-small)."),
    force: bool = typer.Option(False, "--force", help="Allow --reembed with the non-semantic hash embedder."),
):
    """Rebuild the vector index from the database: exactly one vector per stored memory.

    It used to only add, so vectors of deleted memories survived it, and it copied
    the stored embeddings as they were - a store switched from the hash fallback
    to a real model kept its hash vectors, now weighted as if they meant something.
    Rows whose embedding is missing or of the wrong width are always re-embedded;
    --reembed does it for every row.
    """
    from ..retrieval.hybrid import is_semantic

    async def _run():
        mem = await get_memory_system()
        if reembed and not is_semantic(mem.embedder) and not force:
            raise ValueError("the current embedder is the hash fallback, so re-embedding gains nothing "
                             "and rewrites every row; install sentence-transformers, or pass --force")
        embed = getattr(mem.embedder, 'embed_documents', None) or mem.embedder.embed
        # update_content never inserts, so a memory deleted meanwhile stays deleted.
        write = getattr(mem.store, 'update_content', None) or mem.store.put
        items = await mem.store.list_all()
        dim = mem.embedder.dim
        stale = [i for i in items if reembed or not i.embedding or len(i.embedding) != dim]
        # Counted before re-embedding changes them: one row of another width used
        # to abort the whole command with a ValueError from the vector store.
        other_width = sum(1 for i in items if i.embedding and len(i.embedding) != dim)
        for start in range(0, len(stale), 64):
            batch = stale[start:start + 64]
            for item, emb in zip(batch, embed([i.content for i in batch])):
                item.embedding = [float(x) for x in emb]
                await write(item)
        rows = [(i.id, i.embedding, {'tier': i.tier.value}) for i in items]
        if hasattr(mem.vectors, 'add_many'):
            await mem.vectors.add_many(rows)
        else:
            for row in rows:
                await mem.vectors.add(*row)
        # Vector ids first, rows second: a memory's row is written before its
        # vector, so a vector whose row is absent now belongs to a deleted memory,
        # not to one another process is adding right now. Unreadable rows count as
        # present - a key problem is no reason to drop their vectors.
        vector_ids = list(getattr(mem.vectors, 'ids', []))
        if hasattr(mem.store, 'scan'):
            readable, unreadable = await mem.store.scan()
            present = {i.id for i in readable} | set(unreadable)
        else:
            present = {i.id for i in await mem.store.list_all()}
        ghosts = [vid for vid in vector_ids if vid not in present]
        if ghosts and hasattr(mem.vectors, 'delete_many'):
            await mem.vectors.delete_many(ghosts)
        else:
            for vid in ghosts:
                await mem.vectors.delete(vid)
        _out.print(f"Rebuilt {len(rows)} vectors: re-embedded {len(stale)} "
                   f"({other_width} of another width than {dim}), "
                   f"dropped {len(ghosts)} of deleted memories")
        if hasattr(mem.vectors, 'sidecar_path'):
            _out.print(f"Vector sidecar: {mem.vectors.sidecar_path}")
    run_async(_run())

def _key_state(path: Path) -> str:
    """The key file's format, judged without unlocking it."""
    if not path.exists():
        return "missing"
    try:
        if KeyManager(path).is_protected():
            return "wrapped"
        return "raw" if path.stat().st_size == 32 else "unreadable"
    except OSError as e:
        return f"unreadable ({e.strerror})"

def _db_state(path: Path):
    if not path.exists():
        return "missing"
    import sqlite3
    from urllib.parse import quote
    uri = f"file:{quote(str(path.absolute()))}"
    wal, shm = (path.with_name(path.name + s) for s in ("-wal", "-shm"))
    # Only the plaintext row count: no key needed. A mode=ro open of a WAL
    # database creates -wal and -shm when they are missing and cannot remove
    # them, so a health check run as another user (sudo) left files the store's
    # own writer could not open, and on a read-only copy it failed outright.
    # immutable=1 creates nothing; it misses rows still in an unmerged -wal,
    # which is why the WAL is read whenever a writer has left it there.
    modes = (["mode=ro"] if wal.exists() and shm.exists() else []) + ["immutable=1"]
    err = None
    for mode in modes:
        try:
            con = sqlite3.connect(f"{uri}?{mode}", uri=True)
            try:
                return {"rows": con.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                        "bytes": path.stat().st_size}
            finally:
                con.close()
        except sqlite3.Error as e:
            err = e
    return f"unreadable ({err})"

def _sidecar_state(path: Path):
    if not path.exists():
        return "missing"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {"dim": data.get("dim"), "vectors": len(data.get("ids", [])),
                "embedder": data.get("embedder")}
    except (OSError, ValueError, AttributeError) as e:
        return f"unreadable ({type(e).__name__})"

@system_app.command("health")
def system_health():
    """Passive check of the store's files, as JSON. Needs no password and writes nothing.

    Exits 1 when the key or the database is missing or unreadable. It used to print
    a fixed string claiming 'ok' and 'p2p: enabled' whatever the state of the store.
    """
    import importlib.util
    from ..storage.vector_store import VectorStore
    checks = {"key": _key_state(settings.key_path)}
    if settings.backend == "postgres":
        checks["db"] = "postgres (not checked)"
        db_ok = True
    else:
        checks["db"] = _db_state(settings.db_path)
        db_ok = isinstance(checks["db"], dict)
    checks["vectors"] = _sidecar_state(Path(settings.vector_path).with_suffix(VectorStore.SIDECAR_SUFFIX))
    kg_path = settings.db_path.with_suffix('.kg.db')
    checks["kg"] = {"bytes": kg_path.stat().st_size} if kg_path.exists() else "missing"
    ok = checks["key"] in ("raw", "wrapped") and db_ok
    out = {
        "status": "ok" if ok else "degraded",
        "data_dir": str(settings.data_dir),
        "backend": settings.backend,
        "encryption": "AES-256-GCM",
        "embedding_provider": settings.embedding_provider,
        # Configured is not the same as available: without it the hash fallback runs.
        "sentence_transformers": importlib.util.find_spec("sentence_transformers") is not None,
        "p2p": "not_implemented",
        "checks": checks,
    }
    sys.stdout.write(json.dumps(out) + "\n")
    raise typer.Exit(0 if ok else 1)

# Server commands
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}

@api_app.command("start")
def server_start(port: int = 8000, host: str = "0.0.0.0"):
    import uvicorn
    from ..api.rest import create_app

    async def _serve():
        # Built in the loop that serves it. Building it under a separate
        # asyncio.run() and handing it to uvicorn.run() left anything bound to the
        # first loop - an asyncpg pool with MNEM_BACKEND=postgres - on a closed loop.
        mem = await get_memory_system()
        # Only the REST API starts here. This used to also claim an MCP server and a
        # P2P node on {settings.p2p_port}; neither was ever started, and nothing has
        # ever listened on that port. Use `server mcp` for MCP.
        print(f"Starting REST API on {host}:{port} (MCP: `memcore server mcp`; P2P: not implemented)")
        if not settings.api_key and host not in _LOOPBACK:
            sys.stderr.write(f"[memcore] WARNING: REST API is UNAUTHENTICATED on {host}:{port}; anyone who "
                             f"can reach it can read and delete every memory. Set MNEM_API_KEY.\n")
        await uvicorn.Server(uvicorn.Config(create_app(mem), host=host, port=port)).serve()

    try:
        run_async(_serve())
    except KeyboardInterrupt:
        raise typer.Exit(0)

@api_app.command("mcp")
def server_mcp(
    remote: str = typer.Option(
        None, "--remote", envvar="MNEM_REMOTE_URL",
        help="Proxy to a running REST instance (e.g. http://192.168.1.183:8000) instead of "
             "opening the local store. Needed whenever the real store lives in a container: "
             "the host copy is a different database.",
    ),
    timeout: float = typer.Option(30.0, help="HTTP timeout in seconds, --remote only."),
    api_key: str = typer.Option(
        None, "--api-key", envvar="MNEM_API_KEY",
        help="Bearer key for a --remote instance that sets MNEM_API_KEY.",
    ),
    api_key_file: Path = typer.Option(
        None, "--api-key-file",
        help="Read the --api-key from this file (first line). For clients such as the "
             "desktop app that don't source the shell profile, so the key stays out of "
             "a committed config.",
    ),
):
    """Serve the MCP protocol over stdio (for Claude Desktop, Claude Code, any MCP client)."""
    import contextlib
    from ..mcp.remote import RemoteUnavailable

    if api_key_file and not api_key:
        try:
            api_key = api_key_file.read_text().strip() or None
        except OSError as e:
            sys.stderr.write(f"[memcore] cannot read --api-key-file {api_key_file}: {e}\n")
            raise typer.Exit(1)
        if not api_key:
            sys.stderr.write(f"[memcore] --api-key-file {api_key_file} is empty\n")
            raise typer.Exit(1)

    async def _run():
        # stdout is the JSON-RPC stream from here on; keep startup chatter off it.
        with contextlib.redirect_stdout(sys.stderr):
            # Which store answers is the one thing a client cannot see from the
            # tools: an exported MNEM_REMOTE_URL used to switch a local-mode
            # config to the bridge without a word.
            if remote:
                source = "MNEM_REMOTE_URL" if os.environ.get("MNEM_REMOTE_URL") == remote else "--remote"
                sys.stderr.write(f"[memcore] bridge mode -> {remote} (from {source})\n")
                from ..mcp.remote import RemoteMCPServer
                server = await RemoteMCPServer(remote, timeout=timeout, api_key=api_key).connect()
            else:
                sys.stderr.write(f"[memcore] local store ({settings.backend}): {settings.data_dir.absolute()}\n")
                mem = await get_memory_system()
                from ..mcp.server import MCPServer
                server = MCPServer(mem)
        await server.serve_stdio()

    try:
        run_async(_run())
    except RemoteUnavailable as e:
        _fail(str(e))
    except (KeyboardInterrupt, EOFError):
        raise typer.Exit(0)

def _command_count(t: typer.Typer) -> int:
    return len(t.registered_commands) + sum(_command_count(g.typer_instance) for g in t.registered_groups)

app.info.help = f"memcore - Production-grade lifelong memory system ({_command_count(app)} commands)"

if __name__ == "__main__":
    app()
