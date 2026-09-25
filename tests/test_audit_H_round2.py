"""Round-2 ops/packaging checks for the 2026-09-24 audit (group H).

Textual like test_audit_h_ops.py - no docker, no kubectl, no network - plus two key
loads in a tmp dir that back up what the k8s manifest promises about
MNEM_MASTER_PASSWORD.
"""
import json
import re
import subprocess
import tomllib
from pathlib import Path

import pytest

from memcore_memory.crypto.key_manager import KeyManager, MasterPasswordRequired

ROOT = Path(__file__).resolve().parent.parent


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _live_lines(rel):
    return [l for l in _read(rel).splitlines() if not l.lstrip().startswith("#")]


def _deps():
    return tomllib.loads(_read("pyproject.toml"))["project"]["dependencies"]


# --- pyproject (F88, F9) -------------------------------------------------------

def test_networkx_is_not_a_dependency_and_nothing_imports_it():
    assert not any(d.startswith("networkx") for d in _deps())
    src = ROOT / "src"
    importers = [p for p in src.rglob("*.py")
                 if re.search(r"^\s*(import|from)\s+networkx\b", p.read_text(encoding="utf-8"), re.M)]
    assert importers == [], importers


def test_jsonschema_is_declared_because_the_mcp_server_imports_it():
    assert any(re.match(r"jsonschema>=4", d) for d in _deps())
    assert re.search(r"^from jsonschema import", _read("src/memcore_memory/mcp/server.py"), re.M)


# --- .gitignore / .dockerignore (H-3, H-4) --------------------------------------

@pytest.mark.parametrize("path", ["20260925-volume-before-deploy.tgz", "sub/dir/x.tgz", "sub/x.tar.gz"])
def test_gitignore_covers_tgz_backups(path):
    r = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=ROOT)
    assert r.returncode == 0, f"{path} is not ignored"


def test_dockerignore_matches_gitignore_store_and_backup_files_at_any_depth():
    lines = set(_live_lines(".dockerignore"))
    for pattern in ("**/*.hnsw", "**/working_buffer.jsonl", "**/backups/", "**/*.tar.gz", "**/*.tgz"):
        assert pattern in lines, pattern
    # A root-only "backups/" next to "**/backups/" would just be noise.
    assert "backups/" not in lines


# --- k8s (R13, H-5) ------------------------------------------------------------

def _env_block(manifest, name):
    m = re.search(rf"- name: {name}\n((?:\s{{10,}}.*\n)+)", manifest)
    assert m, f"{name} not in the Deployment env"
    return m.group(1)


def test_deployment_wires_an_optional_master_password():
    dep = "\n".join(_live_lines("k8s/mnemosyne-deployment.yaml"))
    block = _env_block(dep + "\n", "MNEM_MASTER_PASSWORD")
    assert "name: mnemosyne-secrets" in block
    assert "key: master-password" in block
    # Optional: a raw key must keep working with no password in the Secret.
    assert "optional: true" in block
    assert re.search(r"- name: MNEM_ENV\n\s+value: prod", dep)


def test_deployment_keys_mount_has_a_volume_and_no_dead_secret_keys():
    dep = "\n".join(_live_lines("k8s/mnemosyne-deployment.yaml"))
    assert re.search(r"- name: keys\n\s+mountPath: /keys", dep)
    assert re.search(r"- name: keys\n\s+secret:\n\s+secretName: mnemosyne-secrets", dep)
    both = dep + _read("k8s/secrets.example.yaml")
    assert "master-key-b64" not in both


def test_secret_example_documents_the_wrapped_key_path():
    text = _read("k8s/secrets.example.yaml")
    assert "master-password" in text and "MasterPasswordRequired" in text


def test_prod_is_not_defaulted_outside_k8s():
    # SkyNAS runs on a raw key it must keep loading; prod there is the owner's call.
    for rel in ("Dockerfile", "docker-compose.yml"):
        assert not re.search(r"MNEM_ENV\s*[=:]\s*\"?prod", "\n".join(_live_lines(rel))), rel


def test_raw_key_still_loads_when_the_optional_password_is_set(tmp_path, monkeypatch):
    key = tmp_path / "master.key"
    key.write_bytes(bytes(range(32)))
    monkeypatch.setenv("MNEM_MASTER_PASSWORD", "unused")
    assert KeyManager(key).load_or_create(db_path=tmp_path / "memory.db") == bytes(range(32))
    assert key.read_bytes() == bytes(range(32))


def test_wrapped_key_needs_the_password_the_manifest_supplies(tmp_path, monkeypatch):
    key = tmp_path / "master.key"
    monkeypatch.setenv("MNEM_MASTER_PASSWORD", "s3cret")
    created = KeyManager(key).load_or_create(db_path=tmp_path / "memory.db")
    assert set(json.loads(key.read_text())) >= {"salt", "nonce", "ct"}
    assert KeyManager(key).load_or_create(db_path=tmp_path / "memory.db") == created
    # Without the Secret key the pod gets exactly this, not a new key.
    monkeypatch.delenv("MNEM_MASTER_PASSWORD")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False, raising=False)
    before = key.read_bytes()
    with pytest.raises(MasterPasswordRequired):
        KeyManager(key).load_or_create(db_path=tmp_path / "memory.db")
    assert key.read_bytes() == before


# --- examples (F32) ------------------------------------------------------------

def test_p2p_demo_says_p2p_is_not_implemented_and_opens_no_socket():
    text = _read("examples/p2p_demo.py")
    head = text.split("import", 1)[0]
    assert "NOT implemented" in head and "NotImplementedError" in head
    code = "\n".join(_live_lines("examples/p2p_demo.py"))
    assert "P2PNode" not in code and "socket" not in code
