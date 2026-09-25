"""Shared test fixtures.

The important one is ``isolate_data_dir``: without it the suite ran against the real
``~/.memcore``, so ``create_memory_system()`` in test_memory.py was writing test
memories into the user's live encrypted store (and, once that store gained a master
password, failing outright). Tests must never touch it.
"""

import os

import pytest

from memcore_memory import config
from memcore_memory.config import _DERIVED_PATHS

# test_postgres.py opts in through these at collection time and needs them again at
# run time, so scrubbing them would silently turn an intended postgres run into a skip.
_KEEP_ENV = {"MNEM_BACKEND", "MNEM_DATABASE_URL"}


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Point every path in the global settings object at a per-test directory.

    The shell environment is scrubbed too. With MEMCORE_ENV=prod exported - the
    setting the Docker deployment is meant to run with - every fixture that creates
    an unprotected key failed, and any stray MNEM_* variable leaks into a freshly
    constructed Settings() and quietly overrides what the test asked for.
    """
    data_dir = tmp_path / "memcore"
    data_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config.settings, "data_dir", data_dir)
    for field, filename in _DERIVED_PATHS.items():
        monkeypatch.setattr(config.settings, field, data_dir / filename)
    monkeypatch.delenv("MEMCORE_ENV", raising=False)
    fields = type(config.settings).model_fields
    for name in list(os.environ):
        if not name.upper().startswith("MNEM_") or name.upper() in _KEEP_ENV:
            continue
        monkeypatch.delenv(name)
        # The global object was built at import, before this ran, so it already
        # absorbed the variable (MNEM_API_KEY would lock every REST test out).
        field = name[len("MNEM_"):].lower()
        if field in fields and field != "data_dir" and field not in _DERIVED_PATHS:
            monkeypatch.setattr(config.settings, field,
                                fields[field].get_default(call_default_factory=True))
    return data_dir
