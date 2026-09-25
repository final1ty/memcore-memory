
import asyncio
import base64
import contextlib
import json
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import hnswlib
    HAS_HNSW = True
except ImportError:
    HAS_HNSW = False

try:
    import fcntl
except ImportError:  # Windows: no advisory locks, so only one writer per store is safe
    fcntl = None


# The only per-vector metadata that may reach the sidecar. Callers used to pass
# the first 500 characters of the memory as meta, and the sidecar wrote them out
# in plaintext next to a database whose content column is AES-GCM encrypted.
# Anything not listed here is dropped on add *and* on load, so an old sidecar
# loses its plaintext the next time it is written. The tier is the tier at
# the time the vector was written: retiering, consolidation and eviction don't
# update it, and nothing reads it. Filter on tiers through the store, never here.
ALLOWED_META = ("tier",)

# Format 2 stores the vectors as base64 float32. Format 1 (no "format" key)
# stored them as JSON float lists, which cost about a second of pure GIL-held
# encoding per write at 5000 memories - rewritten on every add. Both are read.
SIDECAR_FORMAT = 2

# A leftover of the pre-flock writer, which wrote every sidecar through this one
# fixed name. It survives only a write that died between write and rename, and
# then holds a full old-format copy - plaintext meta included - at 0644.
# Anything this old is not a live write: those took well under a second.
_LEGACY_TMP_SUFFIX = ".tmp"
_LEGACY_TMP_MIN_AGE = 60.0

_Op = Optional[Tuple[np.ndarray, dict]]   # None means "delete this id"
_State = Tuple[List[str], List[np.ndarray], Dict[str, dict]]


class _Unusable(Exception):
    """The sidecar exists but cannot be merged into this store."""

    def __init__(self, reason: str, foreign_dim: Optional[int] = None,
                 foreign_embedder: Optional[str] = None):
        super().__init__(reason)
        self.foreign_dim = foreign_dim
        self.foreign_embedder = foreign_embedder

    @property
    def foreign(self) -> bool:
        """Valid vectors from another embedder, as opposed to a broken file."""
        return bool(self.foreign_dim or self.foreign_embedder)


def embedder_identity(embedder) -> Optional[str]:
    """What the sidecar records as the source of its vectors.

    The dimension alone cannot tell two embedders apart: the hash fallback and
    bge-small are both 384-dim, and their vectors are unrelated. BGEEmbedder
    keeps its model_name after falling back to the hash, so the loaded model is
    what decides, the same test `retrieval.hybrid.is_semantic` makes.
    """
    if embedder is None:
        return None
    dim = getattr(embedder, "dim", None)
    if type(embedder).__name__ == "LocalHashEmbedder" or getattr(embedder, "_model", True) is None:
        return f"local-hash-{dim}"
    name = str(getattr(embedder, "model_name", type(embedder).__name__))
    return name + "-binary" if getattr(embedder, "binary", False) else name


def _legacy_embedder(dim) -> str:
    # Every sidecar written before the field existed came from the hash
    # fallback: sentence-transformers has never been installed on a deployment.
    return f"local-hash-{dim}"


def _clean_meta(meta) -> dict:
    if not isinstance(meta, dict):
        return {}
    return {k: meta[k] for k in ALLOWED_META if isinstance(meta.get(k), str)}


def _sig(st: os.stat_result) -> Tuple[int, int, int]:
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def _apply_op(ids: List[str], vectors: List[np.ndarray], meta: Dict[str, dict],
              pos: Dict[str, int], mem_id: str, op: _Op) -> bool:
    i = pos.get(mem_id)
    if op is None:
        meta.pop(mem_id, None)
        if i is None:
            return False
        del ids[i]
        del vectors[i]
        pos.clear()
        pos.update((m, j) for j, m in enumerate(ids))
        return True
    vec, m = op
    if i is None:
        pos[mem_id] = len(ids)
        ids.append(mem_id)
        vectors.append(vec)
    else:
        vectors[i] = vec
    meta[mem_id] = m
    return True


class VectorStore:
    """Vectors plus an optional HNSW index over them, cached in a JSON sidecar.

    The source of truth is the `embedding` column in SQLite; the sidecar is a
    cache of it that `memcore system reindex-vectors` can always rebuild. It
    still has to be right, because the vector arm searches nothing else.

    Several processes open the same store - the stdio MCP server for a whole
    session, every CLI command, the sync script. Each used to rewrite the whole
    file from the lists it loaded at startup, so the long-lived process wiped
    every vector the others wrote at its next add. Writes now take an flock,
    re-read the file when someone else changed it, and apply only this
    process's own adds and deletes on top. `search()` reloads when the file
    changed underneath it. Every write goes to its own temp file: a fixed
    `.tmp` name let two writers interleave bytes and rename a corrupt file
    into place.

    The index is rebuilt from the lists rather than mutated in place. Labels used
    to be `len(self.ids)-1`, i.e. positions, while `delete()` removed from the
    middle of the list and shifted every later position - so after a single
    delete the index mapped labels to the wrong memories, silently. It is never
    saved: it was only ever rebuilt from the lists, so the file was dead weight.
    """

    SIDECAR_SUFFIX = ".vectors.json"

    def __init__(self, path: Path, dim: int = 768, embedder_id: Optional[str] = None):
        self.path = Path(path)
        self.dim = dim
        # None means the caller did not say: nothing is checked and whatever the
        # file records is carried over. Given, a sidecar from any other embedder
        # is treated like one of another dimension - set aside, never merged.
        self.embedder_id = embedder_id
        # The view searches run against: the last state read from or written to
        # disk, plus this process's operations not yet written.
        self.id_to_meta: Dict[str, dict] = {}
        self.ids: List[str] = []
        self.vectors: List[np.ndarray] = []
        self._pos: Dict[str, int] = {}
        self.index = None
        self._dirty = True
        self._matrix: Optional[np.ndarray] = None
        # What is on disk as far as this process knows, and the file identity
        # it came from. Replaced as a whole, never mutated, so a writer thread
        # and the event loop can share it.
        self._base: _State = ([], [], {})
        self._base_sig: Optional[Tuple[int, int, int]] = None
        self._base_needs_rewrite = False
        self._base_embedder: Optional[str] = None
        self._legacy_tmp_checked = False
        self._ignored_sig: Optional[Tuple[int, int, int]] = None
        self._pending: Dict[str, _Op] = {}
        self._inflight: Dict[str, _Op] = {}
        self._ops_lock = threading.Lock()   # guards _pending/_inflight/_base, held briefly
        self._io_lock = threading.Lock()    # one writer per process; flock covers the rest
        self._load()

    @property
    def sidecar_path(self) -> Path:
        return self.path.with_suffix(self.SIDECAR_SUFFIX)

    @property
    def _lock_path(self) -> Path:
        return self.sidecar_path.with_name(self.sidecar_path.name + ".lock")

    # --- reading ------------------------------------------------------------

    def _stat(self) -> Optional[Tuple[int, int, int]]:
        try:
            return _sig(os.stat(self.sidecar_path))
        except FileNotFoundError:
            return None

    def _read_disk(self):
        """(sig, state, needs_rewrite, embedder), or None when there is no sidecar.

        The signature comes from the same open file the bytes are read from, so
        it can never describe a newer file than the contents.
        """
        try:
            with open(self.sidecar_path, "rb") as f:
                sig = _sig(os.fstat(f.fileno()))
                raw = f.read()
        except FileNotFoundError:
            return None
        except OSError as e:
            raise _Unusable(f"unreadable ({e})")
        try:
            data = json.loads(raw)
        except ValueError as e:   # covers JSONDecodeError and UnicodeDecodeError
            raise _Unusable(f"not valid JSON ({e})")
        return (sig, *self._parse(data))

    def _parse(self, data) -> Tuple[_State, bool, Optional[str]]:
        # Valid JSON is not enough: `[]` used to crash every entry point in
        # __init__ - reindex-vectors, the repair command, included - and one
        # short vector made every search raise inside np.asarray.
        if not isinstance(data, dict):
            raise _Unusable(f"top level is {type(data).__name__}, expected an object")
        stored_dim = data.get("dim")
        if stored_dim and stored_dim != self.dim:
            raise _Unusable(f"holds {stored_dim}-dim vectors but this store is {self.dim}-dim",
                            foreign_dim=stored_dim)
        stored_emb = data.get("embedder")
        if stored_emb is not None and not isinstance(stored_emb, str):
            raise _Unusable("embedder is not a string")
        if self.embedder_id:
            effective = stored_emb or _legacy_embedder(self.dim)
            if effective != self.embedder_id:
                raise _Unusable(f"holds vectors from embedder {effective!r} but this store embeds "
                                f"with {self.embedder_id!r}", foreign_embedder=effective)
        ids, meta = data.get("ids", []), data.get("meta", {})
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise _Unusable("ids is not a list of strings")
        if len(set(ids)) != len(ids):
            raise _Unusable("ids contains duplicates")
        if not isinstance(meta, dict):
            raise _Unusable("meta is not an object")
        try:
            if "vectors_b64" in data:
                buf = base64.b64decode(data["vectors_b64"], validate=True)
                matrix = np.frombuffer(buf, dtype="<f4")
                if matrix.size != len(ids) * self.dim:
                    raise ValueError(f"{matrix.size} floats for {len(ids)} ids")
                matrix = matrix.reshape(len(ids), self.dim)
            else:
                vectors = data.get("vectors", [])
                if not isinstance(vectors, list) or len(vectors) != len(ids):
                    raise ValueError(f"{len(ids)} ids but "
                                     f"{len(vectors) if isinstance(vectors, list) else 'no'} vectors")
                matrix = (np.asarray(vectors, dtype=np.float32) if vectors
                          else np.empty((0, self.dim), dtype=np.float32))
        except (ValueError, TypeError) as e:
            raise _Unusable(f"vectors are unusable ({e})")
        if matrix.ndim != 2 or matrix.shape != (len(ids), self.dim):
            raise _Unusable(f"vector matrix is {matrix.shape}, expected ({len(ids)}, {self.dim})")
        if not np.isfinite(matrix).all():
            raise _Unusable("vectors contain NaN or infinity")
        vectors = [np.array(row, dtype=np.float32) for row in matrix]
        known = set(ids)
        clean = {i: _clean_meta(m) for i, m in meta.items() if i in known}
        needs_rewrite = (data.get("format") != SIDECAR_FORMAT or clean != meta
                         or bool(self.embedder_id and stored_emb != self.embedder_id))
        return (ids, vectors, clean), needs_rewrite, stored_emb

    def _load(self):
        try:
            found = self._read_disk()
        except _Unusable as e:
            # Nothing is moved here: a read-only command must not disturb the
            # file. The first write sets it aside under the lock, so it is kept
            # rather than overwritten.
            if e.foreign_dim:
                advice = ("Run `memcore system reindex-vectors` to rebuild it: rows whose stored "
                          f"embedding is {e.foreign_dim}-dim are re-embedded with the current "
                          "embedder automatically.")
            elif e.foreign_embedder:
                advice = ("Run `memcore system reindex-vectors --reembed` to rebuild it: the stored "
                          "embeddings have the right width but come from the other embedder, so "
                          "a plain reindex would copy them across unchanged.")
            else:
                advice = "Run `memcore system reindex-vectors` to rebuild it from the database."
            print(f"[vectors] {self.sidecar_path} {e} - starting with an empty vector index. "
                  f"The file will be moved aside, not overwritten, on the first write. {advice}",
                  file=sys.stderr)
            self._ignored_sig = self._stat()
            return
        if found is None:
            return
        sig, state, needs_rewrite, emb = found
        self._base, self._base_sig, self._base_needs_rewrite = state, sig, needs_rewrite
        self._base_embedder = emb
        self._rebuild_view()

    def _refresh_sync(self) -> bool:
        """Adopt the on-disk file if another process replaced it. True if adopted."""
        with self._io_lock:
            cur = self._stat()
            if cur is None or cur == self._base_sig or cur == self._ignored_sig:
                return False
            try:
                found = self._read_disk()
            except _Unusable as e:
                print(f"[vectors] {self.sidecar_path} changed on disk but {e} - "
                      f"keeping the vectors already loaded", file=sys.stderr)
                self._ignored_sig = cur
                return False
            if found is None:
                return False
            sig, state, needs_rewrite, emb = found
            with self._ops_lock:
                self._base, self._base_sig, self._base_needs_rewrite = state, sig, needs_rewrite
                self._base_embedder = emb
            return True

    # --- the in-memory view ---------------------------------------------------

    def _rebuild_view(self):
        """View = base + ops being written + ops not yet written, in that order."""
        with self._ops_lock:
            base, ops = self._base, [dict(self._inflight), dict(self._pending)]
        ids, vectors, meta = list(base[0]), list(base[1]), dict(base[2])
        pos = {m: i for i, m in enumerate(ids)}
        for batch in ops:
            for mem_id, op in batch.items():
                _apply_op(ids, vectors, meta, pos, mem_id, op)
        self.ids, self.vectors, self.id_to_meta, self._pos = ids, vectors, meta, pos
        self._changed()

    def _changed(self):
        self._dirty = True
        self._matrix = None

    def _queue(self, mem_id: str, op: _Op) -> bool:
        with self._ops_lock:
            self._pending[mem_id] = op
        changed = _apply_op(self.ids, self.vectors, self.id_to_meta, self._pos, mem_id, op)
        if changed:
            self._changed()
        return changed

    def _prepare(self, mem_id: str, embedding, meta) -> Tuple[np.ndarray, dict]:
        if len(embedding) != self.dim:
            raise ValueError(
                f"embedding has {len(embedding)} dimensions, store expects {self.dim}. "
                f"Storing it would make every similarity against it meaningless."
            )
        vec = np.array(embedding, dtype=np.float32).reshape(-1)
        if not np.isfinite(vec).all():
            raise ValueError(f"embedding for {mem_id} contains NaN or infinity")
        return vec, _clean_meta(meta)

    # --- writing --------------------------------------------------------------

    @contextlib.contextmanager
    def _file_lock(self):
        if fcntl is None:
            yield
            return
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)   # releases the flock

    def _free_name(self, first: str, numbered: str) -> Path:
        # Never onto an earlier set-aside: switching A -> B -> A -> B, or two
        # corruptions within one second, used to replace the file kept first.
        # Every set-aside happens under the flock, so the check cannot race
        # another cooperating process.
        target = self.sidecar_path.with_name(first)
        n = 1
        while os.path.lexists(target):
            target = self.sidecar_path.with_name(numbered.format(n))
            n += 1
        return target

    def _set_aside(self, err: _Unusable):
        # Keep what could not be merged. Overwriting it would destroy another
        # embedder's vectors or the evidence of whatever corrupted the file.
        stem, name = self.sidecar_path.stem, self.sidecar_path.name
        if err.foreign:
            label = (f"dim{err.foreign_dim}" if err.foreign_dim else
                     "emb-" + re.sub(r"[^A-Za-z0-9._-]+", "_", err.foreign_embedder)[:80])
            target = self._free_name(f"{stem}.{label}.bak.json", f"{stem}.{label}.bak.{{}}.json")
        else:
            stamp = int(time.time())
            target = self._free_name(f"{name}.corrupt-{stamp}", f"{name}.corrupt-{stamp}-{{}}")
        try:
            if not (err.foreign and self._set_aside_cleaned(target)):
                os.replace(self.sidecar_path, target)
            with contextlib.suppress(OSError):
                os.chmod(target, 0o600)   # a corrupt file may still hold old plaintext meta
            print(f"[vectors] {self.sidecar_path} {err}; moved it to {target}", file=sys.stderr)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[vectors] {self.sidecar_path} {err} and could not be moved aside ({e}); "
                  f"it will be overwritten", file=sys.stderr)

    def _set_aside_cleaned(self, target: Path) -> bool:
        """Move another embedder's sidecar to `target` minus any plaintext meta.

        Its vectors are what is worth keeping; an old-format file's meta is the
        first 500 characters of every memory, which is not. A corrupt file is
        kept byte for byte instead, as evidence. False leaves the move to the
        caller.
        """
        try:
            with open(self.sidecar_path, "rb") as f:
                data = json.loads(f.read())
        except (OSError, ValueError):
            return False
        meta = data.get("meta") if isinstance(data, dict) else None
        if not isinstance(meta, dict):
            return False
        clean = {i: _clean_meta(m) for i, m in meta.items()}
        if clean == meta:
            return False
        data["meta"] = clean
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(target)
            raise
        os.unlink(self.sidecar_path)
        return True

    def _drop_legacy_tmp(self):
        if self._legacy_tmp_checked:
            return
        self._legacy_tmp_checked = True
        legacy = self.sidecar_path.with_suffix(_LEGACY_TMP_SUFFIX)
        try:
            if time.time() - os.lstat(legacy).st_mtime >= _LEGACY_TMP_MIN_AGE:
                os.unlink(legacy)
                print(f"[vectors] removed {legacy}, left behind by an interrupted write of "
                      f"an older version", file=sys.stderr)
        except FileNotFoundError:
            pass
        except OSError as e:
            print(f"[vectors] could not remove {legacy} ({e}); it may hold plaintext "
                  f"memory excerpts - delete it by hand", file=sys.stderr)

    def _write(self, ids: List[str], vectors: List[np.ndarray], meta: Dict[str, dict],
               embedder: Optional[str] = None):
        parent = self.sidecar_path.parent
        parent.mkdir(parents=True, exist_ok=True)
        matrix = (np.stack(vectors).astype("<f4", copy=False) if vectors
                  else np.empty((0, self.dim), dtype="<f4"))
        doc = {
            "format": SIDECAR_FORMAT,
            "dim": self.dim,
            "ids": ids,
            "vectors_b64": base64.b64encode(matrix.tobytes()).decode("ascii"),
            "meta": meta,
        }
        if embedder:
            doc["embedder"] = embedder
        payload = json.dumps(doc)
        # mkstemp creates the file 0600 whatever the umask; the embeddings are
        # derived from content and are not for other users of the machine.
        fd, tmp = tempfile.mkstemp(dir=parent, prefix=self.sidecar_path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                if hasattr(os, "fchmod"):
                    os.fchmod(f.fileno(), 0o600)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
                sig = _sig(os.fstat(f.fileno()))   # rename keeps inode and mtime
            os.replace(tmp, self.sidecar_path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        with contextlib.suppress(OSError, AttributeError):
            dfd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        return sig

    def _flush_sync(self) -> bool:
        """Write this process's queued operations. True if another process's
        changes were merged in, so the view has to be rebuilt."""
        with self._io_lock:
            with self._ops_lock:
                ops, self._pending = self._pending, {}
                self._inflight = ops
            if not ops:
                return False
            try:
                with self._file_lock():
                    base, sig, emb = self._base, self._base_sig, self._base_embedder
                    needs_rewrite, foreign = self._base_needs_rewrite, False
                    cur = self._stat()
                    if cur is not None and cur != sig:
                        try:
                            found = self._read_disk()
                        except _Unusable as e:
                            self._set_aside(e)
                            found, needs_rewrite, emb = None, True, None
                        if found is not None:
                            sig, base, needs_rewrite, emb = found
                            foreign = True
                    elif cur is None and sig is not None:
                        needs_rewrite = True   # someone removed it; put it back
                    ids, vectors, meta = list(base[0]), list(base[1]), dict(base[2])
                    pos = {m: i for i, m in enumerate(ids)}
                    changed = False
                    for mem_id, op in ops.items():
                        changed |= _apply_op(ids, vectors, meta, pos, mem_id, op)
                    # A caller that names no embedder keeps whatever the file recorded.
                    emb = self.embedder_id or emb
                    if changed or needs_rewrite:
                        sig = self._write(ids, vectors, meta, emb)
                        needs_rewrite = False
                    self._drop_legacy_tmp()
            except BaseException:
                with self._ops_lock:
                    for mem_id, op in ops.items():
                        self._pending.setdefault(mem_id, op)   # a newer op for the id wins
                    self._inflight = {}
                raise
            with self._ops_lock:
                self._base, self._base_sig = (ids, vectors, meta), sig
                self._base_needs_rewrite, self._base_embedder = needs_rewrite, emb
                self._ignored_sig = None
                self._inflight = {}
            return foreign

    async def _flush(self):
        # The write runs in a thread so fsync and file I/O don't stall every
        # other request on the event loop. Concurrent callers coalesce: the
        # first thread to get the lock writes everything queued so far.
        if await asyncio.to_thread(self._flush_sync):
            self._rebuild_view()

    async def add(self, mem_id: str, embedding: List[float], meta: Optional[dict] = None):
        vec, clean = self._prepare(mem_id, embedding, meta)
        self._queue(mem_id, (vec, clean))
        await self._flush()

    async def add_many(self, rows: Iterable[Tuple[str, List[float], Optional[dict]]]) -> int:
        """Upsert many vectors with one sidecar write instead of one per row."""
        prepared = [(mem_id, self._prepare(mem_id, emb, meta)) for mem_id, emb, meta in rows]
        for mem_id, op in prepared:
            self._queue(mem_id, op)
        await self._flush()
        return len(prepared)

    async def delete(self, mem_id: str):
        # Queued and written even when this process has never seen the id:
        # another process may have added it, and the merge removes it there.
        self._queue(mem_id, None)
        await self._flush()

    async def delete_many(self, mem_ids: Iterable[str]) -> int:
        """Delete many vectors with one sidecar write. Returns how many of them
        this process's view held; the others are still removed wherever the
        file on disk has them."""
        removed = sum(self._queue(mem_id, None) for mem_id in list(mem_ids))
        await self._flush()
        return removed

    # --- searching ------------------------------------------------------------

    def _rebuild_index(self):
        """Rebuild from the lists, so labels and positions can never drift apart."""
        self._dirty = False
        if not HAS_HNSW or not self.ids:
            self.index = None if not HAS_HNSW else self.index
            return
        index = hnswlib.Index(space="cosine", dim=self.dim)
        index.init_index(max_elements=max(1024, len(self.ids) * 2), ef_construction=200, M=16)
        index.add_items(np.stack(self.vectors), np.arange(len(self.ids)))
        index.set_ef(max(32, min(200, len(self.ids))))
        self.index = index

    async def search(self, query_embedding: List[float], k: int = 10) -> List[Tuple[str, float]]:
        cur = self._stat()
        if cur is not None and cur != self._base_sig and cur != self._ignored_sig:
            if await asyncio.to_thread(self._refresh_sync):
                self._rebuild_view()
        if not self.vectors:
            return []
        q = np.array(query_embedding, dtype=np.float32)
        if q.shape[0] != self.dim:
            print(f"[vectors] query is {q.shape[0]}-dim, store is {self.dim}-dim - no results",
                  file=sys.stderr)
            return []

        if HAS_HNSW:
            if self._dirty:
                self._rebuild_index()
            if self.index is not None:
                try:
                    labels, distances = self.index.knn_query(q, k=min(k, len(self.ids)))
                    return [(self.ids[label], float(1 - dist))
                            for label, dist in zip(labels[0], distances[0])]
                except Exception as e:  # noqa: BLE001 - fall back to exact search
                    print(f"[vectors] HNSW query failed ({e}), using exact search",
                          file=sys.stderr)

        if self._matrix is None:
            self._matrix = np.stack(self.vectors)
        matrix = self._matrix
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q) + 1e-9
        sims = (matrix @ q) / norms
        order = np.argsort(sims)[::-1][:k]
        return [(self.ids[i], float(sims[i])) for i in order]
