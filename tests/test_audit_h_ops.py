"""Ops/packaging checks for the 2026-09-24 audit (group H).

These files are never imported by the test suite otherwise, so every one of the
defects here - a dependency floor that let the package crash on import, probes
pointed at a full-table decrypt, an example that pruned the user's real store -
was invisible to the tests. The checks are textual on purpose: no docker, no
kubectl, no network.
"""
import inspect
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _pyproject():
    return tomllib.loads(_read("pyproject.toml"))


def _live_lines(rel):
    """Non-comment lines, so explanatory comments can mention what was removed."""
    return [l for l in _read(rel).splitlines() if not l.lstrip().startswith("#")]


# --- pyproject (R17, F94, R31) ------------------------------------------------

def _dep(deps, name):
    return next(d for d in deps if re.match(rf"{re.escape(name)}\b", d))


def test_cryptography_floor_has_argon2id():
    deps = _pyproject()["project"]["dependencies"]
    assert _dep(deps, "cryptography").startswith("cryptography>=44")
    assert not any(d.startswith("argon2-cffi") for d in deps)
    # And the code really does need it.
    from cryptography.hazmat.primitives.kdf.argon2 import Argon2id  # noqa: F401


def test_mcp_floor_matches_the_server_api_used():
    deps = _pyproject()["project"]["dependencies"]
    assert _dep(deps, "mcp").startswith("mcp>=2.2")
    from mcp.server.lowlevel import Server
    params = inspect.signature(Server.__init__).parameters
    assert "on_list_tools" in params and "on_call_tool" in params


def test_no_unused_or_optional_only_core_deps():
    project = _pyproject()["project"]
    deps = project["dependencies"]
    for name in ("websockets", "sqlalchemy", "faiss-cpu"):
        assert not any(d.startswith(name) for d in deps), name
    extras = project["optional-dependencies"]
    assert not any("faiss" in d for group in extras.values() for d in group)
    # The alias must survive: deleting it breaks existing `[excellent]` installs.
    assert "excellent" in extras
    assert any(d.startswith("sqlalchemy") for d in extras["postgres"])
    assert any(d.startswith("sqlalchemy") for d in extras["all"])


def test_summary_claims_nothing_unshipped():
    project = _pyproject()["project"]
    desc = project["description"]
    for claim in ("mTLS", "MRR", "reranker", "audit log", "power-law", "excellent", "blind index"):
        assert claim.lower() not in desc.lower(), claim
    for kw in ("audit-log", "blind-index", "excellent", "bge"):
        assert kw not in project["keywords"], kw


def test_mcp_manifest_matches_tool_schema():
    from memcore_memory.mcp.tools import get_tools_schema
    manifest = json.loads(_read("mcp_manifest.json"))
    schema = {t["name"] for t in get_tools_schema()}
    listed = {t["name"] if isinstance(t, dict) else t for t in manifest["tools"]}
    assert listed == schema
    assert manifest["tools_count"] == len(schema)


# --- .gitignore / .dockerignore (F99) -----------------------------------------

IGNORED = [
    "master.key", "keys/other.key", ".master-password", "k8s/secrets.yaml",
    "memory.db", "memory.db-wal", "memory.db-shm", "memory.db-journal",
    "vectors.vectors.json", "memory.vectors.json", "audit.log", "working_buffer.jsonl",
    "backups/x.tar.gz", "container-rescue-1.json", "sub/container-rescue-2.json",
    "memcore-full-audit-2026-09-24_wf_951355b1-d57/agentek-nyers/agent-1.jsonl",
]


@pytest.mark.parametrize("path", IGNORED)
def test_gitignore_covers_secrets_and_store_files(path):
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not ignored"


def test_gitignore_does_not_hide_tracked_files():
    r = subprocess.run(["git", "ls-files", "-ci", "--exclude-standard"], cwd=ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0
    # k8s/secrets.yaml is still in the index until the rename to secrets.example.yaml
    # is committed; only files that exist on disk would really be hidden.
    hidden = [p for p in r.stdout.split() if (ROOT / p).exists()]
    assert hidden == [], hidden


def test_dockerignore_keeps_key_material_out_of_the_context():
    lines = set(_live_lines(".dockerignore"))
    for pattern in ("**/*.key", "**/.master-password", "**/*.db", "**/container-rescue-*.json"):
        assert pattern in lines, pattern


# --- Dockerfile / compose (F101, group notes) ---------------------------------

def test_dockerfile_probes_livez_and_exposes_rest_only():
    live = "\n".join(_live_lines("Dockerfile"))
    health = re.search(r"HEALTHCHECK.*?CMD (.*)", live, re.S).group(1)
    assert "/livez" in health and "/health" not in health
    assert re.search(r"^EXPOSE 8000\s*$", live, re.M)
    assert "7742" not in live


def test_dockerfile_postgres_backend_installs_its_extras():
    live = "\n".join(_live_lines("Dockerfile"))
    assert "ARG BACKEND=sqlite" in live
    assert '.[postgres]' in live and "import asyncpg, pgvector.sqlalchemy" in live
    assert "MNEM_BACKEND=${BACKEND}" in live


def test_compose_keeps_live_names_and_drops_p2p_port():
    live = "\n".join(_live_lines("docker-compose.yml"))
    # The project name is what makes the volume memcore-memory-100_mnem_data.
    assert re.search(r"^name: memcore-memory-100$", live, re.M)
    assert "container_name: mnemosyne" in live
    assert '"8000:8000"' in live
    assert "mnem_data:/data" in live
    assert "7742" not in live


# --- k8s (F17, F18, F52, F53, F54, F100, R30) ---------------------------------

def test_k8s_deployment_env_is_read_by_config():
    live = "\n".join(_live_lines("k8s/mnemosyne-deployment.yaml"))
    names = re.findall(r"- name: (\S+)\n\s+value", live) + re.findall(r"- name: (\S+)\n\s+valueFrom", live)
    for name in names:
        # POSTGRES_PASSWORD only feeds $(POSTGRES_PASSWORD) in the DSN below it.
        assert name.startswith("MNEM_") or name == "POSTGRES_PASSWORD", name
    assert "MNEM_DATABASE_URL" in names
    assert names.index("POSTGRES_PASSWORD") < names.index("MNEM_DATABASE_URL")
    assert "MNEM_EMBEDDING_PROVIDER" not in names
    assert "- name: MNEM_KEY_PATH\n          value: /keys/master.key" in live


def test_k8s_deployment_is_startable_and_single_writer():
    live = "\n".join(_live_lines("k8s/mnemosyne-deployment.yaml"))
    assert re.search(r"^\s+replicas: 1$", live, re.M)
    assert "HorizontalPodAutoscaler" not in live
    assert "type: Recreate" in live
    assert "imagePullPolicy: Always" not in live
    assert re.search(r"kind: PersistentVolumeClaim\nmetadata:\n\s+name: mnemosyne-data", live)
    assert "prometheus.io" not in live
    probes = re.findall(r"Probe:\n\s+httpGet:\n\s+path: (\S+)", live)
    assert probes == ["/livez", "/livez"]
    # The key mount carries the key file only, not the DB password.
    assert "items:\n          - key: master.key" in live


def test_k8s_kustomization_orders_namespace_and_leaves_secret_out():
    live = "\n".join(_live_lines("k8s/kustomization.yaml"))
    resources = re.findall(r"^\s+- (\S+\.yaml)$", live, re.M)
    assert resources[0] == "namespace.yaml"
    assert not any("secret" in r or "monitoring" in r for r in resources)
    for r in resources:
        assert (ROOT / "k8s" / r).exists(), r
    assert not (ROOT / "k8s" / "monitoring.yaml").exists()


def test_k8s_postgres_image_pinned_and_pgdata_subdir():
    live = "\n".join(_live_lines("k8s/postgres.yaml"))
    assert "image: pgvector/pgvector:pg16" in live
    assert "value: /var/lib/postgresql/data/pgdata" in live
    assert ":latest" not in live


def test_k8s_secret_example_has_no_dsn_copy_of_the_password():
    live = "\n".join(_live_lines("k8s/secrets.example.yaml"))
    assert "database-url" not in live
    assert "master.key:" in live


# --- Makefile (F55, F56, F95) -------------------------------------------------

def test_makefile_targets_are_runnable():
    mk = _read("Makefile")
    test_recipe = re.search(r"^test:\n((?:\t.*\n)+)", mk, re.M).group(1)
    assert "--cov" not in test_recipe
    assert "--cov=memcore_memory" in mk and "--cov=mnemosyne" not in mk
    assert "docker-compose" not in mk and "docker compose up" in mk
    assert "rufflehog" not in mk and "|| true" not in mk
    assert not re.search(r"^docs:", mk, re.M)
    assert "system stats" not in mk
    phony = set(re.search(r"^\.PHONY: (.*)$", mk, re.M).group(1).split())
    targets = set(re.findall(r"^([a-z][\w-]*):", mk, re.M))
    assert targets <= phony, targets - phony


# --- examples (F60, R16, R29) -------------------------------------------------

def _run_example(name, tmp_path, *args, extra_env=None):
    sentinel = tmp_path / "must-stay-empty"
    sentinel.mkdir()
    home = tmp_path / "home"
    home.mkdir()
    env = {**os.environ, "HOME": str(home), "TMPDIR": str(tmp_path),
           "MNEM_DATA_DIR": str(sentinel), "MNEM_DB_PATH": str(sentinel / "memory.db"),
           **(extra_env or {})}
    r = subprocess.run([sys.executable, str(ROOT / "examples" / name), *args], env=env,
                       capture_output=True, text=True, timeout=180, cwd=tmp_path)
    return r, sentinel, home


@pytest.mark.parametrize("name", ["quickstart.py", "mcp_demo.py"])
def test_store_examples_never_touch_a_configured_store(name, tmp_path):
    r, sentinel, home = _run_example(name, tmp_path)
    assert r.returncode == 0, r.stderr
    # Neither the store MNEM_DATA_DIR pointed at nor ~/.memcore was opened.
    assert list(sentinel.iterdir()) == []
    assert list(home.iterdir()) == []
    assert "Demo store:" in r.stdout


def test_mcp_demo_states_the_real_tool_count():
    from memcore_memory.mcp.tools import get_tools_schema
    head = _read("examples/mcp_demo.py").splitlines()[0]
    assert f"{len(get_tools_schema())} tools" in head


def test_rest_demo_has_no_default_target(tmp_path):
    src = _read("examples/rest_api_demo.py")
    assert "localhost:8000" not in "\n".join(_live_lines("examples/rest_api_demo.py"))
    assert "client.delete(" in src
    r, _, _ = _run_example("rest_api_demo.py", tmp_path)
    assert r.returncode != 0
    assert "usage:" in r.stderr


def test_p2p_demo_opens_no_socket(tmp_path):
    r, sentinel, home = _run_example("p2p_demo.py", tmp_path)
    assert r.returncode == 0, r.stderr
    assert "mem-2" in r.stdout
    assert "P2PNode" not in "\n".join(_live_lines("examples/p2p_demo.py"))
