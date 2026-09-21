"""Shared test fixtures.

The important one is ``isolate_data_dir``: without it the suite ran against the real
``~/.memcore``, so ``create_memory_system()`` in test_memory.py was writing test
memories into the user's live encrypted store (and, once that store gained a master
password, failing outright). Tests must never touch it.
"""

import pytest

from memcore_memory import config
from memcore_memory.config import _DERIVED_PATHS


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Point every path in the global settings object at a per-test directory."""
    data_dir = tmp_path / "memcore"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config.settings, "data_dir", data_dir)
    for field, filename in _DERIVED_PATHS.items():
        monkeypatch.setattr(config.settings, field, data_dir / filename)
    monkeypatch.delenv("MNEM_MASTER_PASSWORD", raising=False)
    return data_dir
