"""Audit group B: the vector sidecar.

Covers the plaintext leak (F1, R18), multi-process safety (R1, R7), write cost
on the event loop (F40), the dim-mismatch overwrite (F67) and structural
validation on load (R11). Everything runs in tmp_path.
"""

import asyncio
import json
import os
import stat
import subprocess
import sys
import textwrap

import pytest

from memcore_memory.storage import vector_store as vs_mod
from memcore_memory.storage.vector_store import VectorStore


def _sidecar(path):
    return path.with_suffix(VectorStore.SIDECAR_SUFFIX)


def _disk_ids(path):
    return VectorStore(path, dim=4).ids


def _unit(i, dim=4):
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# --- F1 / R18: no plaintext, 0600 ------------------------------------------

async def test_content_never_reaches_the_sidecar(tmp_path):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    await vs.add("a", _unit(0), {"tier": "working", "content": "CANARY-7731 hunter2"})

    raw = _sidecar(path).read_bytes()
    assert b"CANARY-7731" not in raw and b"hunter2" not in raw
    assert stat.S_IMODE(_sidecar(path).stat().st_mode) == 0o600
    assert vs.id_to_meta["a"] == {"tier": "working"}
    assert VectorStore(path, dim=4).id_to_meta["a"] == {"tier": "working"}


async def test_old_format_sidecar_loads_and_loses_its_plaintext_on_next_write(tmp_path):
    """Format 1, exactly as the previous code wrote it: float lists, content in
    meta, default umask."""
    path = tmp_path / "v.hnsw"
    side = _sidecar(path)
    side.write_text(json.dumps({
        "dim": 4, "ids": ["old1", "old2"],
        "vectors": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        "meta": {"old1": {"tier": "episodic", "content": "CANARY-legacy secret"},
                 "old2": {"tier": "working", "content": "CANARY-legacy other"}},
    }))
    os.chmod(side, 0o644)

    vs = VectorStore(path, dim=4)
    assert vs.ids == ["old1", "old2"]
    assert vs.id_to_meta == {"old1": {"tier": "episodic"}, "old2": {"tier": "working"}}
    assert [m for m, _ in await vs.search([0.0, 1.0, 0.0, 0.0], k=1)] == ["old2"]

    await vs.add("new", _unit(2), {"tier": "working"})
    raw = side.read_bytes()
    assert b"CANARY-legacy" not in raw
    assert stat.S_IMODE(side.stat().st_mode) == 0o600
    assert _disk_ids(path) == ["old1", "old2", "new"]


async def test_a_no_op_delete_still_purges_an_old_format_file(tmp_path):
    path = tmp_path / "v.hnsw"
    _sidecar(path).write_text(json.dumps({
        "dim": 4, "ids": ["x"], "vectors": [[1.0, 0.0, 0.0, 0.0]],
        "meta": {"x": {"tier": "working", "content": "CANARY-noop"}}}))
    await VectorStore(path, dim=4).delete("not-there")
    assert b"CANARY-noop" not in _sidecar(path).read_bytes()
    assert _disk_ids(path) == ["x"]


async def test_no_temp_files_are_left_behind(tmp_path):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    for i in range(5):
        await vs.add(f"m{i}", _unit(i), {})
    await vs.delete("m0")
    assert sorted(p.name for p in tmp_path.glob("v.*")) == sorted(
        [_sidecar(path).name, _sidecar(path).name + ".lock"])


# --- R1: last writer no longer wins -----------------------------------------

async def test_a_long_lived_process_keeps_what_others_wrote(tmp_path):
    path = tmp_path / "v.hnsw"
    a = VectorStore(path, dim=4)            # e.g. the stdio MCP server
    await a.add("m1", _unit(0), {})
    b = VectorStore(path, dim=4)            # e.g. a CLI command
    await b.add("m2", _unit(1), {})

    await a.add("m3", _unit(2), {})
    assert set(_disk_ids(path)) == {"m1", "m2", "m3"}
    assert [m for m, _ in await a.search(_unit(1), k=1)] == ["m2"]


async def test_deletes_from_one_process_are_not_undone_by_another(tmp_path):
    path = tmp_path / "v.hnsw"
    a = VectorStore(path, dim=4)
    await a.add("m1", _unit(0), {})
    await a.add("m2", _unit(1), {})
    b = VectorStore(path, dim=4)
    await b.delete("m1")

    await a.add("m3", _unit(2), {})          # a still has m1 in memory
    assert set(_disk_ids(path)) == {"m2", "m3"}
    assert "m1" not in [m for m, _ in await a.search(_unit(0), k=5)]


async def test_delete_reaches_a_vector_this_process_never_saw(tmp_path):
    path = tmp_path / "v.hnsw"
    a = VectorStore(path, dim=4)
    b = VectorStore(path, dim=4)
    await b.add("from-b", _unit(0), {})
    await a.delete("from-b")
    assert _disk_ids(path) == []


_WRITER = textwrap.dedent("""
    import asyncio, sys
    from pathlib import Path
    from memcore_memory.storage.vector_store import VectorStore

    async def main(path, tag, n):
        vs = VectorStore(Path(path), dim=4)
        for i in range(n):
            await vs.add(f"{tag}{i}", [float(i + 1), 1.0, 0.0, 0.0], {"tier": "working"})

    asyncio.run(main(sys.argv[1], sys.argv[2], int(sys.argv[3])))
""")


def test_concurrent_processes_neither_lose_vectors_nor_crash(tmp_path):
    """R7: every process used the same '.tmp' name, so writers interleaved
    bytes, renamed a corrupt file into place, and the loser's add() raised."""
    path = tmp_path / "v.hnsw"
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(path), tag, "100"],
                              stderr=subprocess.PIPE, text=True)
             for tag in ("a", "b", "c")]
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    data = json.loads(_sidecar(path).read_text())
    assert data["format"] == 2
    assert set(_disk_ids(path)) == {f"{t}{i}" for t in "abc" for i in range(100)}
    assert not list(tmp_path.glob("*.tmp"))


async def test_a_failed_write_keeps_the_operation_for_the_next_one(tmp_path, monkeypatch):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    real = vs_mod.os.replace
    monkeypatch.setattr(vs_mod.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        await vs.add("lost?", _unit(0), {})
    monkeypatch.setattr(vs_mod.os, "replace", real)
    await vs.add("next", _unit(1), {})
    assert _disk_ids(path) == ["lost?", "next"]
    assert not list(tmp_path.glob("*.tmp"))


# --- F40: fewer, cheaper writes ----------------------------------------------

async def test_add_many_writes_the_sidecar_once(tmp_path, monkeypatch):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    calls = []
    real = VectorStore._write
    monkeypatch.setattr(VectorStore, "_write", lambda self, *a: calls.append(1) or real(self, *a))
    assert await vs.add_many((f"m{i}", _unit(i), {"tier": "working"}) for i in range(200)) == 200
    assert len(calls) == 1
    assert len(_disk_ids(path)) == 200


async def test_concurrent_adds_coalesce_and_all_land(tmp_path, monkeypatch):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    calls = []
    real = VectorStore._write
    monkeypatch.setattr(VectorStore, "_write", lambda self, *a: calls.append(1) or real(self, *a))
    await asyncio.gather(*(vs.add(f"m{i}", _unit(i), {}) for i in range(50)))
    assert len(calls) < 50
    assert len(_disk_ids(path)) == 50


async def test_deleting_an_unknown_id_does_not_write(tmp_path):
    path = tmp_path / "v.hnsw"
    await VectorStore(path, dim=4).delete("nothing")
    assert not _sidecar(path).exists()


async def test_the_write_does_not_run_on_the_event_loop(tmp_path, monkeypatch):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    seen = []
    real = VectorStore._write

    def spy(self, *a):
        try:
            asyncio.get_running_loop()
            seen.append("loop")
        except RuntimeError:
            seen.append("thread")
        return real(self, *a)

    monkeypatch.setattr(VectorStore, "_write", spy)
    await vs.add("a", _unit(0), {})
    assert seen == ["thread"]


# --- F67: a different embedder's vectors are kept, not overwritten -----------

async def test_dim_mismatch_moves_the_old_file_aside_instead_of_overwriting(tmp_path, capsys):
    path = tmp_path / "v.hnsw"
    old = VectorStore(path, dim=8)
    for i in range(3):
        await old.add(f"old{i}", _unit(i, 8), {})

    vs = VectorStore(path, dim=4)
    assert vs.ids == []
    assert "reindex-vectors" in capsys.readouterr().err
    assert json.loads(_sidecar(path).read_text())["dim"] == 8   # untouched by opening

    await vs.add("n", _unit(0), {})
    backup = path.with_name("v.vectors.dim8.bak.json")
    bak = json.loads(backup.read_text())
    assert bak["dim"] == 8 and bak["ids"] == ["old0", "old1", "old2"]
    live = json.loads(_sidecar(path).read_text())
    assert live["dim"] == 4 and live["ids"] == ["n"]


# --- R11: structurally invalid sidecars --------------------------------------

@pytest.mark.parametrize("content", [
    "[]",
    "null",
    "42",
    '{"dim": 4, "ids": ["a", "a"], "vectors": [[1,0,0,0],[0,1,0,0]]}',
    '{"dim": 4, "ids": ["a", "b"], "vectors": [[1,0,0,0],[0,1]]}',
    '{"dim": 4, "ids": ["a"], "vectors": [[1,0,0,"x"]]}',
    '{"dim": 4, "ids": ["a"], "vectors": [[NaN,0,0,0]]}',
    '{"dim": 4, "ids": "a", "vectors": []}',
    '{"dim": 4, "ids": ["a"], "vectors": [[1,0,0,0]], "meta": []}',
    '{"format": 2, "dim": 4, "ids": ["a"], "vectors_b64": "AAAA"}',
    "{not json",
])
async def test_an_invalid_sidecar_starts_empty_and_is_kept_aside(tmp_path, content, capsys):
    path = tmp_path / "v.hnsw"
    _sidecar(path).write_text(content)

    vs = VectorStore(path, dim=4)             # must not raise
    assert vs.ids == []
    assert await vs.search(_unit(0), k=3) == []
    assert "reindex-vectors" in capsys.readouterr().err

    await vs.add("fresh", _unit(0), {})
    corrupt = list(tmp_path.glob("v.vectors.json.corrupt-*"))
    assert len(corrupt) == 1 and corrupt[0].read_text() == content
    assert _disk_ids(path) == ["fresh"]


async def test_non_finite_embeddings_are_refused(tmp_path):
    vs = VectorStore(tmp_path / "v.hnsw", dim=4)
    with pytest.raises(ValueError, match="NaN"):
        await vs.add("bad", [float("nan"), 0.0, 0.0, 0.0], {})
    assert vs.ids == []
