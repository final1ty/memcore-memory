"""Audit group C, round 2: review follow-ups on core/, config.py and the package init.

Every test runs against the per-test data dir from conftest's autouse
``isolate_data_dir``; subprocess tests get their own MNEM_DATA_DIR.
"""

import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from memcore_memory import create_memory_system
from memcore_memory.config import Settings, settings
from memcore_memory.core.ebbinghaus import ForgettingCurve
from memcore_memory.core.memory import embedder_identity
from memcore_memory.core.tiers import TIER_BASE_STRENGTH, Tier, validate_importance
from memcore_memory.graph.kg import KnowledgeGraphUnreadable

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
async def mem():
    return await create_memory_system()


async def age(mem, item_id, seconds):
    item = await mem.store.get(item_id)
    item.timestamp -= seconds
    item.forgetting.last_access -= seconds
    await mem.store.put(item)


def _clean_env(**extra):
    env = {k: v for k, v in os.environ.items()
           if not k.upper().startswith("MNEM_") and k != "MEMCORE_ENV"}
    env.update(extra)
    return env


# --- R14 / H-1: the unknown-variable warning -------------------------------

def _mnem_names_in_use():
    """Every MNEM_* name the code, scripts and deployment manifests set or read."""
    token = re.compile(r"(?<![A-Za-z0-9_])MNEM_[A-Z0-9_]*[A-Z0-9]")
    roots = [REPO / "src", REPO / "scripts", REPO / "k8s", REPO / "examples"]
    files = [p for r in roots if r.exists() for p in r.rglob("*")
             if p.is_file() and p.suffix in {".py", ".yaml", ".yml", ".sh", ".json"}]
    files += [REPO / n for n in ("docker-compose.yml", "Dockerfile", ".mcp.json", "Makefile")
              if (REPO / n).exists()]
    names = set()
    for f in files:
        names |= set(token.findall(f.read_text(errors="replace")))
    return names


def test_every_mnem_variable_in_use_is_known(tmp_path, monkeypatch, capsys):
    names = _mnem_names_in_use()
    assert {"MNEM_ENV", "MNEM_DATA_DIR", "MNEM_MASTER_PASSWORD", "MNEM_REMOTE_URL"} <= names
    for name in names:
        field = Settings.model_fields.get(name[len("MNEM_"):].lower())
        default = field.get_default(call_default_factory=True) if field else None
        if isinstance(default, bool):
            value = str(default).lower()
        elif isinstance(default, list):
            value = ""
        elif default is None:
            value = str(tmp_path / "x")
        else:
            value = str(default)
        monkeypatch.setenv(name, value)
    Settings(data_dir=tmp_path)
    err = capsys.readouterr().err
    assert "unknown environment variable" not in err, err


def test_mnem_env_is_not_reported_as_ignored(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MNEM_ENV", "prod")
    Settings(data_dir=tmp_path)
    assert "MNEM_ENV" not in capsys.readouterr().err


def test_a_misspelt_variable_still_warns(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MNEM_DATADIR", "/x")
    Settings(data_dir=tmp_path)
    assert "MNEM_DATADIR" in capsys.readouterr().err


def test_invalid_environment_fails_with_one_line_naming_the_variable(tmp_path):
    out = subprocess.run([sys.executable, "-c", "import memcore_memory"], cwd=REPO,
                         capture_output=True, text=True,
                         env=_clean_env(MNEM_DATA_DIR=str(tmp_path), MNEM_BACKEND="mysql"))
    assert out.returncode != 0
    last = out.stderr.strip().splitlines()[-1]
    assert last.startswith("memcore_memory.config.ConfigError") and "MNEM_BACKEND" in last
    assert out.stdout == ""


# --- F39: switches for modules nothing runs --------------------------------

def test_unwired_feature_switches_default_off(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.reranker_enabled is False
    assert s.matryoshka_enabled is False
    assert s.binary_quantization is False


# --- F36: one importance rule ----------------------------------------------

@pytest.mark.parametrize("bad", [True, False])
async def test_bool_importance_is_rejected(mem, bad):
    with pytest.raises(ValueError):
        validate_importance(bad)
    with pytest.raises(ValueError):
        await mem.add("flagged", importance=bad)


# --- F22: expired working memories are demoted by the next add -------------

async def test_add_demotes_expired_working_memories(mem):
    old = await mem.add("an hour-old note")
    fresh = await mem.add("a fresh note")
    await age(mem, old.id, 3600)
    assert (await mem.store.get(old.id)).forgetting.retention() < 0.2

    await mem.add("the next note")
    loaded = await mem.store.get(old.id)
    assert loaded.tier == Tier.EPISODIC
    assert loaded.forgetting.strength >= TIER_BASE_STRENGTH[Tier.EPISODIC]
    assert loaded.forgetting.retention() > 0.9
    assert (await mem.store.get(fresh.id)).tier == Tier.WORKING


# --- F79: lower ranks strengthen but do not count toward promotion ---------

def test_rehearse_can_strengthen_without_counting():
    fc = ForgettingCurve(strength=7.0)
    fc.rehearse(feedback=0.5, count=False)
    assert fc.rehearsals == 0 and fc.strength > 7.0
    fc.rehearse()
    assert fc.rehearsals == 1


async def test_repeated_second_place_hits_do_not_consolidate(mem, monkeypatch):
    top = await mem.add("the answer", tier="episodic", importance=0.95)
    runner_up = await mem.add("near the answer", tier="episodic", importance=0.95)

    async def fake_search(query, k=10, tier_filter=None):
        return [{"id": top.id, "retention": 1.0}, {"id": runner_up.id, "retention": 1.0}]

    monkeypatch.setattr(mem.retriever, "search", fake_search)
    for _ in range(settings.semantic_consolidation_threshold):
        await mem.recall("something loosely related")
    assert (await mem.store.get(runner_up.id)).forgetting.rehearsals == 0
    assert await mem.consolidate() == 1
    assert (await mem.store.get(top.id)).tier == Tier.SEMANTIC
    assert (await mem.store.get(runner_up.id)).tier == Tier.EPISODIC


# --- R26: rows storage cannot decode are reported -------------------------

async def test_lifecycle_pass_reports_rows_the_store_cannot_decode(mem):
    good = await mem.add("fine", tier="episodic")
    bad = await mem.add("corrupt curve", tier="episodic")
    con = sqlite3.connect(settings.db_path)
    con.execute("UPDATE memories SET forgetting_json=? WHERE id=?",
                ('{"strength": "bad"}', bad.id))
    con.commit()
    con.close()
    report = await mem.lifecycle_pass()
    assert [e["id"] for e in report["errors"]] == [bad.id]
    assert await mem.store.get(good.id) is not None


# --- contract 3: get() and a concurrent delete ------------------------------

async def test_get_returns_none_when_the_memory_is_deleted_mid_read(mem, monkeypatch):
    item = await mem.add("short-lived")
    real = mem.store.update_forgetting

    async def delete_first(memory_id, forgetting):
        await mem.store.delete(memory_id)
        return await real(memory_id, forgetting)

    monkeypatch.setattr(mem.store, "update_forgetting", delete_first)
    assert await mem.get(item.id) is None
    assert await mem.store.get(item.id) is None


# --- R6: the vector store learns which embedder wrote its vectors ----------

async def test_vector_store_is_told_the_embedder_identity(mem):
    assert mem.vectors.embedder_id == embedder_identity(mem.embedder)


def test_embedder_identity_uses_the_loaded_model_not_the_label():
    class Hash:
        dim = 384
        model_name = "BAAI/bge-small-en-v1.5"
        _model = None  # BGEEmbedder after the fallback keeps its label

    class Real(Hash):
        _model = object()

    assert embedder_identity(Hash()) == "local-hash-384"
    assert embedder_identity(Real()) == "BAAI/bge-small-en-v1.5"


# --- R12: the graph is backfilled, and a broken one does not stop startup --

async def test_deleted_graph_is_rebuilt_from_the_memories(capsys):
    mem = await create_memory_system()
    item = await mem.add("Jellyfin runs natively", entities=["Jellyfin"], tier="episodic")
    kg_path = settings.db_path.with_suffix(".kg.db")
    kg_path.unlink()

    mem = await create_memory_system()
    assert "created empty" in capsys.readouterr().err
    assert await mem.kg.get_related_memories("Jellyfin") == [item.id]
    assert mem.kg.needs_link_backfill is False


async def test_missing_links_are_backfilled_at_startup():
    mem = await create_memory_system()
    item = await mem.add("Jellyfin runs natively", entities=["Jellyfin"], tier="episodic")
    # What a graph migrated from v1 looks like: nodes, but no memory links and
    # no completion marker.
    con = sqlite3.connect(settings.db_path.with_suffix(".kg.db"))
    con.execute("DELETE FROM memory_entities")
    con.execute("DELETE FROM kg_meta WHERE k='links_complete'")
    con.commit()
    con.close()

    mem = await create_memory_system()
    assert await mem.kg.get_related_memories("Jellyfin") == [item.id]


async def test_unreadable_graph_leaves_the_memories_usable(capsys):
    mem = await create_memory_system()
    kept = await mem.add("SkyNAS runs Docker", entities=["SkyNAS"], tier="episodic")
    kg_path = settings.db_path.with_suffix(".kg.db")
    kg_path.write_bytes(b"this is not a sqlite database" * 100)

    mem = await create_memory_system()
    assert "WITHOUT the knowledge graph" in capsys.readouterr().err
    assert mem.kg_error and str(kg_path) in mem.kg_error
    assert (await mem.get(kept.id)).content == "SkyNAS runs Docker"
    added = await mem.add("Docker on SkyNAS again", entities=["Docker"], tier="episodic")
    assert any(r["id"] == kept.id for r in await mem.recall("SkyNAS"))
    with pytest.raises(KnowledgeGraphUnreadable):
        await mem.kg.traverse("SkyNAS")
    assert await mem.delete(added.id) is True
    # The damaged file is left exactly as it was.
    assert kg_path.read_bytes() == b"this is not a sqlite database" * 100


async def test_healthy_start_has_no_graph_error(mem):
    assert mem.kg_error is None


# --- F89: opt-in entity extraction -----------------------------------------

async def test_entities_are_extracted_only_when_enabled(mem, monkeypatch):
    plain = await mem.add("Jellyfin runs on SkyNAS")
    assert plain.entities == []
    monkeypatch.setattr(settings, "auto_extract_entities", True)
    auto = await mem.add("Jellyfin runs on SkyNAS")
    assert set(auto.entities) == {"Jellyfin", "SkyNAS"}
    given = await mem.add("Jellyfin runs on SkyNAS", entities=["Media"])
    assert given.entities == ["Media"]
