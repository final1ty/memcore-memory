"""MCP server over the official SDK (JSON-RPC 2.0 on stdio).

Every tool advertised in ``tools.py`` has a handler in ``HANDLERS`` below; the module
asserts the two agree at import time, so a tool can never be advertised without an
implementation behind it.

Note for anyone adding to this file: under stdio transport, stdout *is* the protocol
stream. Never print to it - use stderr.
"""

import json
import math
import os
import sys
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import mcp.types as types
from jsonschema import Draft202012Validator
from mcp.server import Server
from mcp.server.stdio import stdio_server

from ..core.ebbinghaus import ForgettingCurve
from ..core.tiers import TIER_BASE_STRENGTH, MemoryItem, Tier, validate_importance
from .tools import get_tools_schema

SERVER_NAME = "memcore-memory"
SERVER_VERSION = "1.0.0"


class ToolError(Exception):
    """Raised by a handler to return an error result to the client."""


def _item_dict(item, content_limit: int = None) -> Dict[str, Any]:
    content = item.content
    if content_limit and len(content) > content_limit:
        content = content[:content_limit] + "..."
    return {
        "id": item.id,
        "content": content,
        "tier": item.tier.value,
        "importance": item.metadata.get("importance", 0.5),
        "retention": round(item.forgetting.retention(), 4),
        "rehearsals": item.forgetting.rehearsals,
        "timestamp": item.timestamp,
        "entities": item.entities,
    }


def _tier(value: str) -> Tier:
    try:
        return Tier(value)
    except ValueError:
        raise ToolError(f"unknown tier {value!r}; valid tiers are {[t.value for t in Tier]}")


async def _require(memory, memory_id: str):
    item = await memory.store.get(memory_id)
    if not item:
        raise ToolError(f"no memory with id {memory_id!r}")
    return item


def _embed_document(memory, text: str):
    # A stored memory is a document, as in MnemosyneMemory.add. This used to call
    # embed_query, so an edited memory was indexed with E5's "query:" prefix and
    # sat in a different region of the space from every other stored memory.
    embedder = memory.embedder
    if hasattr(embedder, "embed_documents"):
        return embedder.embed_documents([text])[0]
    return embedder.embed([text])[0]


# --- argument validation --------------------------------------------------
# The schemas in tools.py were advertised but never enforced, so `importance:
# "high"` was stored and then broke every list and recall store-wide, and a wrong
# type surfaced as a KeyError or TypeError from deep inside a handler. Both
# transports dispatch through call_tool, so this is the one place to check.

_VALIDATORS = {}
for _t in get_tools_schema():
    Draft202012Validator.check_schema(_t["inputSchema"])
    _VALIDATORS[_t["name"]] = Draft202012Validator(_t["inputSchema"])


def _non_finite(value, where: str = ""):
    """Path of the first NaN or infinity inside value, else None."""
    if isinstance(value, float):
        return None if math.isfinite(value) else (where or "value")
    if isinstance(value, dict):
        items = value.items()
    elif isinstance(value, (list, tuple)):
        items = enumerate(value)
    else:
        return None
    for k, v in items:
        found = _non_finite(v, f"{where}/{k}" if where else str(k))
        if found:
            return found
    return None


def validate_args(name: str, args) -> Dict[str, Any]:
    """Check args against the tool's inputSchema. Returns them with integral floats as ints."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError(f"arguments for {name} must be an object, got {type(args).__name__}")
    # JSON has no NaN or Infinity, but Python's parser (and so Starlette's) accepts
    # them, and minimum/maximum let NaN through: importance NaN reached the core as
    # a raw ValueError, and a NaN inside metadata was stored as-is.
    bad = _non_finite(args)
    if bad:
        raise ToolError(f"invalid arguments for {name}: {bad}: NaN and Infinity are not valid JSON numbers")
    errors = sorted(_VALIDATORS[name].iter_errors(args), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        where = "/".join(str(p) for p in e.absolute_path) or "arguments"
        more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
        raise ToolError(f"invalid arguments for {name}: {where}: {e.message}{more}")
    # JSON Schema counts 5.0 as an integer, and a float k then fails as a slice index.
    props = _VALIDATORS[name].schema.get("properties", {})
    return {k: int(v) if props.get(k, {}).get("type") == "integer" and isinstance(v, float) else v
            for k, v in args.items()}


async def call_tool(memory, name: str, args) -> Dict[str, Any]:
    """Validate and dispatch one tool call. The stdio server and POST /mcp/call both use it."""
    from ..storage.encrypted_sqlite import CorruptRow
    handler = HANDLERS.get(name)
    if handler is None:
        raise ToolError(f"unknown tool {name!r}")
    try:
        return await handler(memory, validate_args(name, args))
    except CorruptRow as e:
        # The message names the row and column and carries no plaintext, so the
        # caller can be told which memory is damaged instead of a bare traceback.
        raise ToolError(str(e)) from None


async def _scan(memory):
    """(readable items, {id: reason}) - list_all() alone drops bad rows silently."""
    scan = getattr(memory.store, "scan", None)
    if scan is None:
        return await memory.store.list_all(), {}
    return await scan()


# --- handlers -------------------------------------------------------------

async def _memory_add(memory, a):
    item = await memory.add(
        a["content"],
        metadata=a.get("metadata"),
        tier=a.get("tier"),
        importance=a.get("importance", 0.5),
        entities=a.get("entities"),
    )
    return _item_dict(item)


async def _memory_get(memory, a):
    # MnemosyneMemory.get persists only the rehearsed curve; writing the whole row
    # back could resurrect a memory deleted between the read and the write.
    # touch: false reads without rehearsing, as GET /memory/{id} does by default.
    item = await (memory.get(a["id"]) if a.get("touch", True) else memory.store.get(a["id"]))
    if item is None:
        raise ToolError(f"no memory with id {a['id']!r}")
    return _item_dict(item)


async def _memory_delete(memory, a):
    # One delete path for every interface, so the vector and the graph links go
    # with the row. This used to leave the memory's co_occurs edges behind for
    # good, and kg_traverse kept answering from deleted memories.
    if not await memory.delete(a["id"]):
        raise ToolError(f"no memory with id {a['id']!r}")
    return {"deleted": a["id"]}


async def _memory_update(memory, a):
    # Everything is checked before anything is written. The sidecar used to be
    # rewritten first, so an update that then failed on bad metadata left the
    # vector describing text the row did not contain, while the caller was told
    # nothing had changed.
    if "content" not in a and "metadata" not in a:
        raise ToolError("supply content and/or metadata to update")
    if "content" in a and not isinstance(a["content"], str):
        raise ToolError("content must be a string")
    if "metadata" in a and not isinstance(a["metadata"], dict):
        raise ToolError("metadata must be an object")
    item = await _require(memory, a["id"])
    metadata = dict(a.get("metadata") or {})
    importance_changed = "importance" in metadata
    if importance_changed:
        try:
            item.set_importance(metadata.pop("importance"))
        except ValueError as e:
            raise ToolError(str(e))
    item.metadata.update(metadata)
    content_changed = "content" in a
    if content_changed:
        item.content = a["content"]
        item.embedding = _embed_document(memory, a["content"])
    # The row is the source of truth, so it goes first: a failed vector write then
    # leaves only a stale sidecar, which reindex-vectors rebuilds from SQLite.
    update_content = getattr(memory.store, "update_content", None)
    if update_content is not None:
        if not await update_content(item):
            raise ToolError(f"no memory with id {item.id!r}")
    else:
        await memory.store.put(item)
    if importance_changed:
        await memory.store.update_forgetting(item.id, item.forgetting)
    if content_changed:
        await memory.vectors.delete(item.id)
        await memory.vectors.add(item.id, item.embedding, {"tier": item.tier.value})
    return _item_dict(item)


async def _memory_recall(memory, a):
    results = await memory.recall(a["query"], k=a.get("k", 10), tier_filter=a.get("tier_filter"),
                                  rehearse=a.get("rehearse", True))
    return {"count": len(results), "results": results}


async def _memory_search_bm25(memory, a):
    results = await memory.retriever.bm25.retrieve(a["query"], a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_vector(memory, a):
    results = await memory.retriever.vector.retrieve(a["query"], a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_temporal(memory, a):
    # Query-independent: the temporal arm ranks the whole store by recency and
    # retention, so the query is optional and ignored.
    results = await memory.retriever.temporal.retrieve(a.get("query", ""), a.get("k", 10))
    return {"count": len(results), "results": results}


async def _memory_search_graph(memory, a):
    # This returned kg.traverse's entity edges and never a memory, despite the
    # name. Memories tagged with exactly the entity come first; then those whose
    # entities are named inside it ("docker" for "Docker Compose"), which is how
    # get_related_memories matches; then those tagged with entities the graph
    # reaches from it.
    from ..graph.kg import _norm
    entity, k = a["entity"], a.get("k", 10)
    start = _norm(entity)
    # Over-fetched: links can outlive a memory written by an older version, and
    # get_many drops those, so asking for exactly k came back short.
    ids = list(await memory.kg.get_related_memories(entity, limit=k * 4))
    found = await memory.store.get_many(ids)
    exact = [i for i in ids if i in found and start in {_norm(e) for e in found[i].entities}]
    ordered = exact + [i for i in ids if i in found and i not in exact]
    if len(exact) < k:
        edges = await memory.kg.traverse(entity, depth=a.get("depth", 2), limit=200)
        reached = list(dict.fromkeys(n for e in edges for n in (e["src"], e["dst"]) if n != start))
        if reached:
            more = [m for m in await memory.kg.get_related_memories(" ; ".join(reached), limit=k * 4)
                    if m not in found]
            found.update(await memory.store.get_many(more))
            ordered += [m for m in more if m in found and m not in ordered]
    results = [_item_dict(found[i], 500) for i in ordered[:k]]
    return {"entity": entity, "count": len(results), "exact_matches": min(len(exact), k), "results": results}


async def _memory_list(memory, a):
    tier = a.get("tier")
    items = await (memory.store.list_by_tier(_tier(tier)) if tier else memory.store.list_all())
    limit = a.get("limit", 20)
    return {"total": len(items), "returned": min(limit, len(items)),
            "items": [_item_dict(i, 200) for i in items[:limit]]}


async def _memory_list_all(memory, a):
    items = await memory.store.list_all()
    limit = a.get("limit", 100)
    return {"total": len(items), "returned": min(limit, len(items)),
            "items": [_item_dict(i, 200) for i in items[:limit]]}


async def _retier(memory, a, direction):
    order = [Tier.SENSORY, Tier.WORKING, Tier.EPISODIC, Tier.SEMANTIC]
    item = await _require(memory, a["id"])
    target = _tier(a["target_tier"])
    current_i, target_i = order.index(item.tier), order.index(target)
    if direction == "promote" and target_i <= current_i:
        raise ToolError(f"{target.value} is not above {item.tier.value}; use memory_demote")
    if direction == "demote" and target_i >= current_i:
        raise ToolError(f"{target.value} is not below {item.tier.value}; use memory_promote")
    if direction == "demote" and target == Tier.SENSORY:
        # Sensory expiry is judged on creation time alone, so a year-old memory
        # moved there - however well retained - was deleted by the next pass.
        raise ToolError("sensory is a 30-second ingest buffer, not a demotion target: a memory moved "
                        "there is deleted by the next memory_forget. Use working or episodic.")
    if not await memory.store.update_tier(item.id, target):
        raise ToolError(f"no memory with id {item.id!r}")
    return {"id": item.id, "from": item.tier.value, "to": target.value}


async def _memory_promote(memory, a):
    return await _retier(memory, a, "promote")


async def _memory_demote(memory, a):
    return await _retier(memory, a, "demote")


async def _memory_touch(memory, a):
    item = await _require(memory, a["id"])
    before = item.forgetting.retention()
    item.touch()
    if not await memory.store.update_forgetting(item.id, item.forgetting):
        raise ToolError(f"no memory with id {item.id!r}")
    return {"id": item.id, "rehearsals": item.forgetting.rehearsals,
            "strength": round(item.forgetting.strength, 4),
            "retention_before": round(before, 4),
            "retention_after": round(item.forgetting.retention(), 4)}


async def _memory_forget(memory, a):
    # The full report: "forgotten" alone hid demotions, promotions and every row
    # the pass could not process.
    pass_ = getattr(memory, "lifecycle_pass", None)
    if pass_ is None:
        return {"forgotten": await memory.forget_expired()}
    return await pass_()


async def _memory_consolidate(memory, a):
    before = Counter(i.tier.value for i in await memory.store.list_all())
    promoted = await memory.consolidate()
    after = Counter(i.tier.value for i in await memory.store.list_all())
    return {"promoted": promoted, "before": dict(before), "after": dict(after)}


async def _memory_stats(memory, a):
    items, unreadable = await _scan(memory)
    per_tier = {}
    for tier in Tier:
        group = [i for i in items if i.tier == tier]
        per_tier[tier.value] = {
            "count": len(group),
            "avg_retention": round(sum(i.forgetting.retention() for i in group) / len(group), 4) if group else 0.0,
        }
    return {"total": len(items), "tiers": per_tier, "unreadable_count": len(unreadable),
            "unreadable": sorted(unreadable)}


# --- export / import --------------------------------------------------------
# Both tools used to open whatever path the client sent. Over POST /mcp/call that
# let anything on the LAN overwrite master.key or memory.db (the process kept
# serving from memory, and the next start could not read the store); under stdio
# any prompt-injected client could do the same to ~/.memcore. They now touch only
# regular files directly inside <data_dir>/exports.

EXPORT_FORMAT = "memcore-export"
EXPORT_FORMAT_VERSION = 2
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def _exports_dir() -> Path:
    from ..config import settings
    root = Path(settings.data_dir) / "exports"
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    return Path(os.path.realpath(root))


def _store_files():
    from ..config import settings
    paths = [settings.key_path, settings.db_path, settings.vector_path,
             settings.audit_log_path, settings.working_buffer_path]
    if settings.db_path:
        paths.append(Path(settings.db_path).with_suffix(".kg.db"))
    if settings.vector_path:
        paths.append(Path(settings.vector_path).with_suffix(".vectors.json"))
    return {Path(os.path.realpath(p)) for p in paths if p}


def _exchange_path(raw) -> Path:
    """Resolve a client-supplied name to a file directly inside the exports dir, or refuse."""
    root = _exports_dir()
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ToolError("path must be a file name such as 'backup.json'")
    p = Path(raw)
    if not p.is_absolute():
        if len(p.parts) != 1 or p.name in (".", ".."):
            raise ToolError(f"path must be a bare file name inside {root}, got {raw!r}")
        p = root / p
    # realpath follows symlinks, so a link planted in the exports dir cannot point out of it.
    target = Path(os.path.realpath(p))
    if target.parent != root or target.name in ("", ".", ".."):
        raise ToolError(f"memory_export and memory_import only use files directly inside {root}; "
                        f"got {raw!r}. Pass a bare file name such as 'backup.json'.")
    if target in _store_files():
        raise ToolError(f"{target} is one of the store's own files")
    return target


def _write_private(path: Path, text: str, overwrite: bool):
    """Write text to a new 0600 file; replace an existing one only when told to."""
    if overwrite:
        if path.exists() and not path.is_file():
            raise ToolError(f"{path} exists and is not a regular file")
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(fd, "w") as fh:
                fh.write(text)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _O_NOFOLLOW, 0o600)
    except FileExistsError:
        raise ToolError(f"{path.name} already exists in {path.parent}; pass overwrite: true to replace it")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


async def _memory_export(memory, a):
    target = _exchange_path(a["path"])
    items, unreadable = await _scan(memory)
    # Lossless on purpose: ids, timestamps and curves are what let an import skip
    # what is already there and bring back the rehearsal history. The old format
    # had none of them, so a restore reset every memory and duplicated the store.
    payload = {
        "format": EXPORT_FORMAT, "format_version": EXPORT_FORMAT_VERSION, "exported_at": time.time(),
        "memories": [{"id": i.id, "content": i.content, "tier": i.tier.value, "timestamp": i.timestamp,
                      "metadata": i.metadata, "entities": i.entities,
                      "forgetting": i.forgetting.to_dict()} for i in items],
    }
    try:
        _write_private(target, json.dumps(payload, indent=2), bool(a.get("overwrite", False)))
    except OSError as e:
        raise ToolError(f"cannot write {target}: {e}")
    result = {"exported": len(items), "path": str(target), "encrypted": False}
    if unreadable:
        # A backup that silently left rows out was the worst kind: nobody learns
        # until the restore. Say which ones, and that the file is not complete.
        result.update(partial=True, skipped=sorted(unreadable),
                      skipped_reasons=unreadable,
                      warning=f"{len(unreadable)} unreadable row(s) are NOT in this export")
    return result


def _check_entry(i: int, entry) -> list:
    """Every reason entry i cannot be imported; empty when it can."""
    if not isinstance(entry, dict):
        return [f"[{i}] is not an object"]
    bad = _non_finite(entry)
    if bad:
        # Checked first: from_dict and float() would accept NaN, and int(inf)
        # raises OverflowError, which nothing below expects.
        return [f"[{i}] {bad} is NaN or Infinity"]
    errors = []
    if not isinstance(entry.get("content"), str) or not entry["content"]:
        errors.append(f"[{i}] needs a non-empty string 'content'")
    if entry.get("id") is not None and (not isinstance(entry["id"], str) or not entry["id"]):
        errors.append(f"[{i}] id must be a non-empty string")
    if entry.get("tier") is not None:
        try:
            _tier(entry["tier"])
        except ToolError as e:
            errors.append(f"[{i}] {e}")
    ts = entry.get("timestamp")
    if ts is not None and (isinstance(ts, bool) or not isinstance(ts, (int, float))):
        errors.append(f"[{i}] timestamp must be a number")
    md = entry.get("metadata")
    if md is not None and not isinstance(md, dict):
        errors.append(f"[{i}] metadata must be an object or null")
    elif md and "importance" in md:
        try:
            validate_importance(md["importance"])
        except ValueError as e:
            errors.append(f"[{i}] {e}")
    ents = entry.get("entities")
    if ents is not None and not (isinstance(ents, list) and all(isinstance(x, str) for x in ents)):
        errors.append(f"[{i}] entities must be a list of strings")
    curve = entry.get("forgetting")
    if curve is not None:
        if not isinstance(curve, dict):
            errors.append(f"[{i}] forgetting must be an object")
        else:
            try:
                ForgettingCurve.from_dict(curve)
            except ValueError as e:
                errors.append(f"[{i}] {e}")
    return errors


# Anything the plain memory.add path would throw away. An entry carrying one of
# these is restored as written; before, only an id sent it down that path, so a
# timestamp or curve without an id was silently replaced by "now" and a new curve.
_RESTORED_FIELDS = ("id", "tier", "timestamp", "forgetting")


def _needs_restore(entry: dict) -> bool:
    return any(entry.get(f) is not None for f in _RESTORED_FIELDS)


def _build_restored(memory, entry: dict) -> MemoryItem:
    """The MemoryItem an exported entry describes, with its own id, timestamp and curve."""
    ts = float(entry["timestamp"]) if entry.get("timestamp") is not None else time.time()
    metadata = dict(entry.get("metadata") or {})
    curve = ForgettingCurve.from_dict(entry["forgetting"], default_last_access=ts) \
        if entry.get("forgetting") else None
    metadata["importance"] = validate_importance(
        metadata.get("importance", curve.importance if curve else 0.5))
    item = MemoryItem(id=entry.get("id") or str(uuid.uuid4()), content=entry["content"],
                      tier=Tier(entry["tier"]) if entry.get("tier") else Tier.EPISODIC,
                      timestamp=ts, metadata=metadata,
                      entities=list(entry.get("entities") or []), forgetting=curve)
    if not entry.get("tier"):
        # An id without a tier used to reach Tier(entry["tier"]) and stop the import
        # halfway. Same rule as memory.add, judged on the entry's own timestamp.
        item.tier = memory.tiers.assign_tier(item, {})
        if curve is None:
            item.forgetting.strength = TIER_BASE_STRENGTH.get(item.tier, 7.0)
    # Re-embedded rather than carried over: the vectors must come from the
    # embedder this store runs now, whatever produced the export.
    item.embedding = _embed_document(memory, item.content)
    return item


async def _memory_import(memory, a):
    target = _exchange_path(a["path"])
    try:
        fd = os.open(target, os.O_RDONLY | _O_NOFOLLOW)
        with os.fdopen(fd) as fh:
            payload = json.load(fh)
    except FileNotFoundError:
        raise ToolError(f"no file {target.name!r} in {target.parent}")
    except (OSError, ValueError) as e:
        raise ToolError(f"cannot read {target.name}: {e}")
    if isinstance(payload, dict) and payload.get("format") == EXPORT_FORMAT:
        if payload.get("format_version") != EXPORT_FORMAT_VERSION or not isinstance(payload.get("memories"), list):
            raise ToolError(f"unsupported export format version {payload.get('format_version')!r}")
        entries = payload["memories"]
    elif isinstance(payload, list):
        entries = payload  # the format before ids and curves were exported
    else:
        raise ToolError("expected a memory_export file or a JSON array of memory objects")

    # The whole file is checked before the first write. A bad entry used to stop
    # the loop halfway with no count, and fixing the file and importing again
    # stored the earlier entries a second time.
    errors = [err for i, e in enumerate(entries) for err in _check_entry(i, e)]
    if errors:
        raise ToolError(f"nothing imported; {len(errors)} problem(s): " + "; ".join(errors[:20]))

    ids = [e["id"] for e in entries if e.get("id")]
    existing_ids = set(await memory.store.get_many(ids)) if ids else set()
    if ids and len(existing_ids) < len(set(ids)):
        # get_many leaves out rows it cannot decrypt, and those rows exist: taking
        # them for absent would restore over them and, on rollback, delete them.
        _, unreadable = await memory.store.scan()
        existing_ids |= set(ids) & set(unreadable)
    # Entries without an id can only be matched by content.
    existing_content = ({i.content for i in await memory.store.list_all()}
                        if any(not e.get("id") for e in entries) else set())
    todo, skipped = [], 0
    for e in entries:
        key = e.get("id")
        if (key and key in existing_ids) or (not key and e["content"] in existing_content):
            skipped += 1
            continue
        if key:
            existing_ids.add(key)
        else:
            existing_content.add(e["content"])
        todo.append(e)

    # All or nothing. Rows first, then every vector in one sidecar write (one
    # write per entry rewrote the whole sidecar each time, O(N^2) bytes for a large
    # file), then the graph links. Any failure removes what this call wrote, so a
    # retry of the corrected file starts from the same store.
    written, restored = [], []
    try:
        for e in todo:
            if _needs_restore(e):
                item = _build_restored(memory, e)
                await memory.store.put(item)
                written.append(item.id)
                restored.append(item)
            else:
                md = dict(e.get("metadata") or {})
                item = await memory.add(e["content"], metadata=md, importance=md.get("importance", 0.5),
                                        entities=e.get("entities"))
                written.append(item.id)
        if restored:
            rows = [(i.id, i.embedding, {"tier": i.tier.value}) for i in restored]
            add_many = getattr(memory.vectors, "add_many", None)
            if add_many is not None:
                await add_many(rows)
            else:
                for row in rows:
                    await memory.vectors.add(*row)
        for item in restored:
            if item.entities:
                await memory.kg.add_memory_entities(item)
    except Exception as ex:
        leftover = []
        for mid in written:
            try:
                await memory.delete(mid)
            except Exception:  # noqa: BLE001 - keep rolling back the rest
                leftover.append(mid)
        detail = (f"{len(leftover)} of the {len(written)} rows written could not be removed again: {leftover}"
                  if leftover else f"the {len(written)} rows written were removed again")
        raise ToolError(f"nothing imported ({skipped} already present); {detail}. "
                        f"Cause: {type(ex).__name__}: {ex}")
    imported = len(written)
    enforce = getattr(memory, "_enforce_working_capacity", None)
    if imported and enforce is not None:
        try:
            await enforce()
        except Exception as ex:  # noqa: BLE001 - the import itself succeeded
            print(f"[{SERVER_NAME}] working capacity not enforced after import: {ex!r}", file=sys.stderr)
    return {"imported": imported, "skipped_existing": skipped, "total": len(entries)}


async def _kg_add_entity(memory, a):
    node_id = await memory.kg.add_entity(a["entity"], type=a.get("type", "entity"), props=a.get("props"))
    return {"entity": node_id}


async def _kg_add_relation(memory, a):
    edge_id = await memory.kg.add_relation(a["src"], a["dst"],
                                           relation=a.get("relation", "related_to"),
                                           weight=a.get("weight", 1.0))
    return {"edge_id": edge_id, "src": a["src"].lower(), "dst": a["dst"].lower()}


async def _kg_traverse(memory, a):
    edges = await memory.kg.traverse(a["entity"], depth=a.get("depth", 2), limit=a.get("limit", 20))
    return {"count": len(edges), "edges": edges}


async def _kg_get_related(memory, a):
    entity = a["entity"].lower()
    edges = await memory.kg.traverse(entity, depth=1, limit=200)
    related = sorted({e["dst"] if e["src"] == entity else e["src"] for e in edges} - {entity})
    return {"entity": entity, "related": related}


async def _kg_list_entities(memory, a):
    entities = await memory.kg.list_entities(limit=a.get("limit", 100))
    return {"count": len(entities), "entities": entities}


async def _kg_delete_entity(memory, a):
    return await memory.kg.delete_entity(a["entity"])


async def _sync_status(memory, a):
    from ..config import settings
    return {"p2p_port": settings.p2p_port, "peers": list(settings.p2p_peers), "implemented": False,
            "note": "configuration only; P2P sync is not implemented and no process listens on "
                    "p2p_port (`memcore server start` runs the REST API only)."}


async def _sync_peers(memory, a):
    from ..config import settings
    return {"count": len(settings.p2p_peers), "peers": list(settings.p2p_peers)}


_CONFIG_REDACT = ("key", "password", "secret", "token", "database_url")
# Settings that exist but that no request path reads. Reported apart, because a
# bare `pii_filter_enabled: true` reads as a protection that is running.
_NOT_WIRED = ("pii_filter_enabled", "pii_filter_action", "audit_log_enabled",
              "reranker_enabled", "reranker_model")


async def _config_get(memory, a):
    from ..config import settings
    data = {}
    for field, value in settings.model_dump().items():
        if any(s in field.lower() for s in _CONFIG_REDACT):
            continue
        data[field] = str(value) if not isinstance(value, (int, float, bool, list, type(None))) else value
    not_wired = {k: data.pop(k) for k in _NOT_WIRED if k in data}
    if not_wired:
        data["not_implemented"] = not_wired
    if memory is not None:
        from ..retrieval.hybrid import is_semantic
        data["embedding_active"] = data.get("embedding_model") if is_semantic(memory.embedder) \
            else "hash fallback (not semantic)"
    key = a.get("key")
    if key:
        if key in not_wired:
            return {key: not_wired[key], "implemented": False}
        if key not in data:
            raise ToolError(f"no config key {key!r} (or it is redacted)")
        return {key: data[key]}
    return data


async def _health_check(memory, a):
    from ..config import settings
    from ..retrieval.hybrid import is_semantic
    items, unreadable = await _scan(memory)
    # The class name alone said BGEEmbedder while the hash fallback was running.
    emb = memory.embedder
    semantic = is_semantic(emb)
    name = type(emb).__name__
    if not semantic and name != "LocalHashEmbedder":
        name = f"LocalHashEmbedder (fallback; {getattr(emb, 'model_name', name)} not loaded)"
    reasons = []
    if unreadable:
        reasons.append(f"{len(unreadable)} unreadable row(s)")
    kg_error = getattr(memory, "kg_error", None)
    if kg_error:
        reasons.append(f"knowledge graph unavailable: {kg_error}")
    return {"status": "degraded" if reasons else "ok", "reasons": reasons,
            "version": SERVER_VERSION, "memories": len(items),
            "unreadable_count": len(unreadable), "unreadable": sorted(unreadable),
            "tier_counts": dict(Counter(i.tier.value for i in items)),
            "data_dir": str(settings.data_dir), "db_path": str(settings.db_path),
            "embedder": name, "semantic": semantic, "timestamp": time.time()}


HANDLERS = {
    "memory_add": _memory_add,
    "memory_get": _memory_get,
    "memory_delete": _memory_delete,
    "memory_update": _memory_update,
    "memory_recall": _memory_recall,
    "memory_search_bm25": _memory_search_bm25,
    "memory_search_vector": _memory_search_vector,
    "memory_search_temporal": _memory_search_temporal,
    "memory_search_graph": _memory_search_graph,
    "memory_list": _memory_list,
    "memory_list_all": _memory_list_all,
    "memory_promote": _memory_promote,
    "memory_demote": _memory_demote,
    "memory_touch": _memory_touch,
    "memory_forget": _memory_forget,
    "memory_consolidate": _memory_consolidate,
    "memory_stats": _memory_stats,
    "memory_export": _memory_export,
    "memory_import": _memory_import,
    "kg_add_entity": _kg_add_entity,
    "kg_add_relation": _kg_add_relation,
    "kg_traverse": _kg_traverse,
    "kg_get_related": _kg_get_related,
    "kg_list_entities": _kg_list_entities,
    "kg_delete_entity": _kg_delete_entity,
    "sync_status": _sync_status,
    "sync_peers": _sync_peers,
    "config_get": _config_get,
    "health_check": _health_check,
}

_advertised = {t["name"] for t in get_tools_schema()}
assert _advertised == set(HANDLERS), (
    f"tools.py and server.py disagree: "
    f"advertised-only={_advertised - set(HANDLERS)}, handler-only={set(HANDLERS) - _advertised}"
)


INSTRUCTIONS = (
    "Persistent 4-tier memory with Ebbinghaus forgetting. Use memory_recall to "
    "look things up and memory_add to store them. Memories decay unless rehearsed; "
    "memory_touch or a higher tier keeps something durable."
)


class StdioMCPServer:
    """Shared stdio/JSON-RPC plumbing: tool advertisement, dispatch, error mapping.

    Subclasses supply the advertised schema and implement ``handle_tool_call``.
    ``remote.py`` reuses this to proxy the same tool surface over HTTP.
    """

    def __init__(self, tools_schema, instructions: str = INSTRUCTIONS):
        self.tools = [
            types.Tool(name=t["name"], description=t["description"], inputSchema=t["inputSchema"])
            for t in tools_schema
        ]
        self.server = Server(
            SERVER_NAME,
            version=SERVER_VERSION,
            instructions=instructions,
            on_list_tools=self._on_list_tools,
            on_call_tool=self._on_call_tool,
        )

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    async def _on_list_tools(self, ctx, params) -> types.ListToolsResult:
        return types.ListToolsResult(tools=self.tools)

    async def _on_call_tool(self, ctx, params: types.CallToolRequestParams) -> types.CallToolResult:
        try:
            result = await self.handle_tool_call(params.name, params.arguments or {})
        except ToolError as e:
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(e))], isError=True)
        except Exception as e:
            print(f"[{SERVER_NAME}] {params.name} failed: {e!r}", file=sys.stderr, flush=True)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"{type(e).__name__}: {e}")], isError=True)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(result, indent=2, default=str))],
            structuredContent=result if isinstance(result, dict) else {"result": result},
        )

    async def serve_stdio(self):
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream, self.server.create_initialization_options())


class MCPServer(StdioMCPServer):
    """Serves a local MnemosyneMemory instance directly."""

    def __init__(self, memory_system):
        super().__init__(get_tools_schema())
        self.memory = memory_system

    async def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch by tool name. Raises ToolError for anything the client got wrong."""
        return await call_tool(self.memory, name, args)
