"""Audit group B, round 2: the vector sidecar.

Embedder identity next to the dimension (R6), set-asides that never overwrite
an earlier one and carry no plaintext (F67), removal of the pre-flock writer's
fixed-name temp file (F1) and a single-write delete_many. Everything runs in
tmp_path.
"""

import importlib.util
import json
import os
import time

import pytest

from memcore_memory.embeddings.local import LocalHashEmbedder
from memcore_memory.storage import vector_store as vs_mod
from memcore_memory.storage.vector_store import VectorStore, embedder_identity


def _sidecar(path):
    return path.with_suffix(VectorStore.SIDECAR_SUFFIX)


def _unit(i, dim=4):
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


def _old_format(path, dim=4, meta=None, embedder=None):
    """A sidecar as the pre-round-1 code wrote it: float lists, no format key."""
    doc = {"dim": dim, "ids": ["m1", "m2"], "vectors": [_unit(0, dim), _unit(1, dim)],
           "meta": meta if meta is not None else {}}
    if embedder is not None:
        doc["embedder"] = embedder
    _sidecar(path).write_text(json.dumps(doc))


# --- R6: the embedder is part of the sidecar's identity ----------------------

async def test_a_sidecar_from_another_embedder_of_the_same_dim_is_set_aside(tmp_path, capsys):
    path = tmp_path / "v.hnsw"
    old = VectorStore(path, dim=4, embedder_id="local-hash-4")
    await old.add("old", _unit(0), {"tier": "working"})
    assert json.loads(_sidecar(path).read_text())["embedder"] == "local-hash-4"

    vs = VectorStore(path, dim=4, embedder_id="BAAI/bge-small-en-v1.5")
    assert vs.ids == []
    assert await vs.search(_unit(0), k=3) == []
    err = capsys.readouterr().err
    assert "reindex-vectors --reembed" in err
    assert json.loads(_sidecar(path).read_text())["ids"] == ["old"]   # opening moves nothing

    await vs.add("new", _unit(1), {})
    backup = json.loads(path.with_name("v.vectors.emb-local-hash-4.bak.json").read_text())
    assert backup["ids"] == ["old"] and backup["embedder"] == "local-hash-4"
    live = json.loads(_sidecar(path).read_text())
    assert live["ids"] == ["new"] and live["embedder"] == "BAAI/bge-small-en-v1.5"


@pytest.mark.parametrize("fmt", ["old", "current"])
async def test_a_sidecar_without_the_field_counts_as_the_hash_fallback(tmp_path, fmt):
    """Every sidecar on disk today predates the field and came from the hash
    embedder, so it must load for the hash embedder and get the field stamped."""
    path = tmp_path / "v.hnsw"
    if fmt == "old":
        _old_format(path)
    else:
        await VectorStore(path, dim=4).add_many([("m1", _unit(0), {}), ("m2", _unit(1), {})])
        assert "embedder" not in json.loads(_sidecar(path).read_text())

    vs = VectorStore(path, dim=4, embedder_id="local-hash-4")
    assert vs.ids == ["m1", "m2"]
    await vs.add("m3", _unit(2), {})
    data = json.loads(_sidecar(path).read_text())
    assert data["embedder"] == "local-hash-4" and data["ids"] == ["m1", "m2", "m3"]


async def test_a_legacy_sidecar_is_not_taken_for_a_real_model(tmp_path):
    path = tmp_path / "v.hnsw"
    _old_format(path)
    assert VectorStore(path, dim=4, embedder_id="BAAI/bge-small-en-v1.5").ids == []


async def test_an_unnamed_caller_keeps_what_the_file_records(tmp_path):
    path = tmp_path / "v.hnsw"
    await VectorStore(path, dim=4, embedder_id="some-model").add("a", _unit(0), {})
    vs = VectorStore(path, dim=4)
    assert vs.ids == ["a"]
    await vs.add("b", _unit(1), {})
    assert json.loads(_sidecar(path).read_text())["embedder"] == "some-model"


async def test_a_non_string_embedder_field_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "v.hnsw"
    _old_format(path, embedder=42)
    vs = VectorStore(path, dim=4)
    assert vs.ids == []
    await vs.add("fresh", _unit(0), {})
    assert len(list(tmp_path.glob("v.vectors.json.corrupt-*"))) == 1


def test_the_dim_mismatch_message_points_at_reindex(tmp_path, capsys):
    path = tmp_path / "v.hnsw"
    _old_format(path, dim=8)
    VectorStore(path, dim=4)
    err = capsys.readouterr().err
    assert "memcore system reindex-vectors" in err
    assert "cannot do that" not in err


def test_embedder_identity():
    assert embedder_identity(None) is None
    assert embedder_identity(LocalHashEmbedder(dim=4)) == "local-hash-4"

    class Loaded:
        model_name, dim, _model = "BAAI/bge-small-en-v1.5", 384, object()

    class FellBack(Loaded):
        _model = None

    class Binary(Loaded):
        binary = True

    assert embedder_identity(Loaded()) == "BAAI/bge-small-en-v1.5"
    assert embedder_identity(FellBack()) == "local-hash-384"
    assert embedder_identity(Binary()) == "BAAI/bge-small-en-v1.5-binary"


@pytest.mark.skipif(importlib.util.find_spec("sentence_transformers") is not None,
                    reason="only meaningful where BGEEmbedder falls back to the hash")
def test_a_bge_embedder_that_fell_back_is_identified_as_the_hash(capsys):
    from memcore_memory.embeddings.bge import BGEEmbedder
    emb = BGEEmbedder(dim=4)
    assert emb.model_name == "BAAI/bge-small-en-v1.5"
    assert embedder_identity(emb) == "local-hash-4"


# --- F67: set-asides are never overwritten and carry no plaintext ------------

async def test_switching_back_and_forth_keeps_every_set_aside_file(tmp_path):
    path = tmp_path / "v.hnsw"
    a = VectorStore(path, dim=8)
    await a.add_many([(f"a{i}", _unit(i, 8), {}) for i in range(3)])
    await VectorStore(path, dim=4).add("b", _unit(0), {})     # sets aside the 3 a-vectors
    await VectorStore(path, dim=8).add("a-again", _unit(0, 8), {})
    await VectorStore(path, dim=4).add("b-again", _unit(1), {})

    first = json.loads(path.with_name("v.vectors.dim8.bak.json").read_text())
    second = json.loads(path.with_name("v.vectors.dim8.bak.1.json").read_text())
    assert first["ids"] == ["a0", "a1", "a2"]
    assert second["ids"] == ["a-again"]
    assert json.loads(path.with_name("v.vectors.dim4.bak.json").read_text())["ids"] == ["b"]


async def test_two_corruptions_in_the_same_second_are_both_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(vs_mod.time, "time", lambda: 1_700_000_000.0)
    path = tmp_path / "v.hnsw"
    for content in ("{broken one", "{broken two"):
        _sidecar(path).write_text(content)
        await VectorStore(path, dim=4).add("x", _unit(0), {})
    kept = sorted(p.read_text() for p in tmp_path.glob("v.vectors.json.corrupt-*"))
    assert kept == ["{broken one", "{broken two"]


async def test_a_foreign_old_format_file_is_set_aside_without_its_plaintext(tmp_path):
    path = tmp_path / "v.hnsw"
    secret = "the wifi password is hunter2"
    _old_format(path, dim=8, meta={"m1": {"content": secret, "tier": "working"},
                                   "m2": {"content": secret}})
    await VectorStore(path, dim=4).add("n", _unit(0), {})

    backup = path.with_name("v.vectors.dim8.bak.json")
    assert secret not in backup.read_text()
    data = json.loads(backup.read_text())
    assert data["ids"] == ["m1", "m2"] and len(data["vectors"]) == 2
    assert data["meta"] == {"m1": {"tier": "working"}, "m2": {}}
    assert oct(os.stat(backup).st_mode & 0o777) == "0o600"
    assert not _sidecar(path).read_text().count(secret)


# --- F1: the pre-flock writer's fixed temp name ------------------------------

async def test_a_stale_legacy_tmp_file_is_removed_on_the_first_write(tmp_path):
    path = tmp_path / "v.hnsw"
    legacy = path.with_name("v.vectors.tmp")
    legacy.write_text(json.dumps({"meta": {"m": {"content": "plaintext"}}}))
    old = time.time() - 3600
    os.utime(legacy, (old, old))

    vs = VectorStore(path, dim=4)
    assert legacy.exists()                    # opening does not touch the directory
    await vs.add("a", _unit(0), {})
    assert not legacy.exists()


async def test_a_fresh_legacy_tmp_file_is_left_for_the_writer_that_owns_it(tmp_path):
    path = tmp_path / "v.hnsw"
    legacy = path.with_name("v.vectors.tmp")
    legacy.write_text("{}")
    await VectorStore(path, dim=4).add("a", _unit(0), {})
    assert legacy.exists()


# --- delete_many -------------------------------------------------------------

async def test_delete_many_writes_once(tmp_path, monkeypatch):
    path = tmp_path / "v.hnsw"
    vs = VectorStore(path, dim=4)
    await vs.add_many([(f"m{i}", _unit(i), {}) for i in range(5)])

    writes = []
    real = VectorStore._write

    def spy(self, *a, **kw):
        writes.append(1)
        return real(self, *a, **kw)

    monkeypatch.setattr(VectorStore, "_write", spy)
    assert await vs.delete_many(["m0", "m2", "m4", "never-seen"]) == 3
    assert len(writes) == 1
    assert vs.ids == ["m1", "m3"]
    assert VectorStore(path, dim=4).ids == ["m1", "m3"]


async def test_delete_many_removes_ids_only_another_process_wrote(tmp_path):
    path = tmp_path / "v.hnsw"
    mine = VectorStore(path, dim=4)
    await VectorStore(path, dim=4).add_many([("x", _unit(0), {}), ("y", _unit(1), {})])
    assert await mine.delete_many(["x"]) == 0     # not in this view, removed on disk anyway
    assert VectorStore(path, dim=4).ids == ["y"]
