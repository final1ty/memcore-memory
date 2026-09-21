"""Regression tests for MNEM_* environment handling.

Two bugs lived here and between them meant a containerised deployment silently wrote
its data somewhere other than the mounted volume:

1. ``env_prefix`` was ``MEMCORE_`` while every Dockerfile/compose/doc sets ``MNEM_*``,
   so the env vars were ignored outright.
2. ``db_path``/``key_path``/... were declared as ``data_dir / "..."`` at class scope,
   which pydantic evaluates once at class-creation time against the *default*
   ``data_dir``. Even with the prefix fixed, ``MNEM_DATA_DIR`` moved ``data_dir`` only,
   leaving every derived path pointing at ``~/.memcore``.

Both are silent failures - the app starts fine and serves requests - so they need
tests rather than a code comment.
"""

import subprocess
import sys
import json
import textwrap

from memcore_memory.config import Settings, _DERIVED_PATHS


def test_env_prefix_is_mnem():
    assert Settings.model_config["env_prefix"] == "MNEM_"


def test_data_dir_env_var_is_honoured(tmp_path):
    s = Settings(data_dir=tmp_path)
    assert s.data_dir == tmp_path


def test_derived_paths_follow_data_dir(tmp_path):
    """The bug: these used to stay under ~/.memcore no matter what data_dir said."""
    s = Settings(data_dir=tmp_path)
    for field, filename in _DERIVED_PATHS.items():
        assert getattr(s, field) == tmp_path / filename, f"{field} did not follow data_dir"


def test_explicit_path_overrides_data_dir(tmp_path):
    custom = tmp_path / "elsewhere" / "custom.db"
    s = Settings(data_dir=tmp_path, db_path=custom)
    assert s.db_path == custom
    assert s.key_path == tmp_path / "master.key"  # siblings still derive


def test_mnem_env_vars_reach_settings_in_a_fresh_process(tmp_path):
    """End-to-end: the prefix and the path derivation together, via real env vars."""
    script = textwrap.dedent("""
        import json
        from memcore_memory.config import settings
        print(json.dumps({
            "data_dir": str(settings.data_dir),
            "db_path": str(settings.db_path),
            "key_path": str(settings.key_path),
        }))
    """)
    target = tmp_path / "datadir"
    out = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, check=True,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path / "home"),
             "MNEM_DATA_DIR": str(target)},
    )
    paths = json.loads(out.stdout.strip().splitlines()[-1])
    assert paths["data_dir"] == str(target)
    assert paths["db_path"] == str(target / "memory.db")
    assert paths["key_path"] == str(target / "master.key")
