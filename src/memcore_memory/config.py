from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict
from pydantic import Field, ValidationError, field_validator, model_validator
from pathlib import Path
from typing import Annotated, Literal, Optional
import json, os, sys

# Paths derived from data_dir, and the filename each one gets inside it.
_DERIVED_PATHS = {
    "db_path": "memory.db",
    "vector_path": "vectors.hnsw",
    "key_path": "master.key",
    "audit_log_path": "audit.log",
    "working_buffer_path": "working_buffer.jsonl",
}

# MNEM_* variables that are read directly rather than through Settings. Each one
# left out here is reported as "ignored" while something does read it - for
# MNEM_ENV that told operators the production key guard was off when it was on.
_EXTERNAL_MNEM_VARS = {
    "MNEM_MASTER_PASSWORD",  # crypto/key_manager.py
    "MNEM_ENV",              # crypto/key_manager.py: refuses an unprotected key in prod
    "MNEM_REMOTE_URL",       # cli/main.py: `server mcp --remote`
}

_BACKEND_ALIASES = {"postgresql": "postgres", "pg": "postgres", "pgvector": "postgres", "sqlite3": "sqlite"}


def normalise_backend(v):
    if isinstance(v, str):
        v = v.strip().lower()
        return _BACKEND_ALIASES.get(v, v)
    return v


class Settings(BaseSettings):
    # Every Dockerfile, compose file and doc in this project sets MNEM_* - the prefix
    # must match or the whole environment is silently ignored. extra="forbid" only
    # affects keyword arguments (env vars never reach model_extra), so a typo such as
    # Settings(working_capcity=3) fails instead of being kept and never read.
    model_config = SettingsConfigDict(env_prefix="MNEM_", extra="forbid")

    data_dir: Path = Path.home() / ".memcore"
    # Left as None so they can be resolved against the *effective* data_dir below.
    # Defining them as `data_dir / "..."` would freeze them to the default at class
    # creation time, silently ignoring MNEM_DATA_DIR. Set any of them explicitly
    # (e.g. MNEM_DB_PATH) to override just that one.
    db_path: Optional[Path] = None
    vector_path: Optional[Path] = None
    key_path: Optional[Path] = None
    audit_log_path: Optional[Path] = None
    working_buffer_path: Optional[Path] = None  # reserved: nothing reads it yet
    p2p_port: int = 7742
    # NoDecode: pydantic-settings otherwise demands JSON for a list, and a plain
    # MNEM_P2P_PEERS=http://a:7742 crashed every command at import, --help included.
    p2p_peers: Annotated[list[str], NoDecode] = []

    embedding_provider: str = "bge-small"
    embedding_dim: int = 384
    embedding_device: Optional[str] = None
    # Reserved: parsed but not wired to anything. The provider above picks the
    # model; changing these has no effect. The Matryoshka embedder and the
    # cross-encoder reranker are experimental modules nothing instantiates, so
    # their switches are off: True here was reported by config_get as a running
    # feature. Kept as fields so existing MNEM_* settings stay valid.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    matryoshka_enabled: bool = False
    binary_quantization: bool = False
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-large"

    # A closed set: "postgresql" used to be accepted and then quietly open SQLite.
    backend: Literal["sqlite", "postgres"] = "sqlite"
    database_url: str = "postgresql+asyncpg://memcore:memcore@localhost:5432/memcore"

    # REST authentication. Unset keeps the API open, as it has always been; set,
    # every route except the liveness/health probes needs it as a Bearer token or
    # X-API-Key header. Kept out of repr so it does not end up in logs.
    api_key: Optional[str] = Field(default=None, repr=False)

    encryption_enabled: bool = True
    blind_index_enabled: bool = True
    audit_log_enabled: bool = True
    pii_filter_enabled: bool = True
    pii_filter_action: str = "warn"
    rate_limit_enabled: bool = True
    mtls_enabled: bool = False
    tls_cert_path: str = ""
    tls_key_path: str = ""
    tls_ca_path: str = ""

    log_level: str = "INFO"
    sensory_ttl_seconds: float = 30.0
    working_capacity: int = 7
    working_ttl_seconds: float = 1200.0
    episodic_decay_rate: float = 0.1  # reserved: decay comes from the forgetting curve
    semantic_consolidation_threshold: int = 3
    # Episodic -> semantic also needs importance above this; see TierManager.
    semantic_min_importance: float = 0.8
    # Decay model for newly created memories. Stored memories keep their own.
    forgetting_model: Literal["exponential", "power_law"] = "exponential"
    # Extract entity names from the text when an add passes none. Off by default:
    # it is a capitalised-word heuristic, and turning it on changes graph ranking.
    auto_extract_entities: bool = False

    @field_validator("data_dir", "db_path", "vector_path", "key_path",
                     "audit_log_path", "working_buffer_path", mode="after")
    @classmethod
    def _expand_user(cls, v):
        # JSON env blocks (.mcp.json, Claude Desktop, k8s, systemd) do not expand
        # "~", so MNEM_DATA_DIR=~/.memcore created a literal ./~ directory - store
        # and master key included - in whatever directory the process started in.
        return v.expanduser() if v is not None else v

    @field_validator("backend", mode="before")
    @classmethod
    def _normalise_backend(cls, v):
        return normalise_backend(v)

    @field_validator("p2p_peers", mode="before")
    @classmethod
    def _split_peers(cls, v):
        if isinstance(v, str):
            s = v.strip()
            if s.startswith("["):
                return json.loads(s)
            return [p.strip() for p in s.split(",") if p.strip()]
        return v

    @model_validator(mode="after")
    def _resolve_paths(self):
        for field, filename in _DERIVED_PATHS.items():
            if getattr(self, field) is None:
                object.__setattr__(self, field, self.data_dir / filename)
        return self

    @model_validator(mode="after")
    def _warn_unknown_env(self):
        # A misspelt MNEM_DATA_DIR is how every memory once ended up outside the
        # volume, and pydantic-settings ignores unknown variables without a word.
        # Warn rather than raise: Settings() runs at import in every entry point.
        # stderr only - stdout carries the stdio MCP stream.
        known = {"MNEM_" + f.upper() for f in type(self).model_fields} | _EXTERNAL_MNEM_VARS
        for name in sorted(os.environ):
            if name.upper().startswith("MNEM_") and name.upper() not in known:
                print(f"[config] warning: unknown environment variable {name} is ignored",
                      file=sys.stderr)
        return self

    def ensure_data_dir(self) -> Path:
        """Create data_dir (0700 when new) for the code paths that open a store.

        This used to run at import, so `server mcp --remote`, which needs no data
        dir at all, failed on an unwritable one and left an empty ~/.memcore on
        bridge-only hosts. An existing directory's mode is left alone.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return self.data_dir


class ConfigError(ValueError):
    """An MNEM_* variable holds a value Settings rejects. The message names it."""


def _load_settings() -> Settings:
    # This runs at import in every entry point, so a pydantic traceback here was
    # the whole output of `memcore --help` with, say, MNEM_BACKEND=mysql. Failing
    # is right - a wrong backend must not quietly open another store - but the
    # message should say which variable to fix.
    try:
        return Settings()
    except ValidationError as e:
        problems = "; ".join(
            f"MNEM_{str(err['loc'][0]).upper() if err.get('loc') else '?'}: {err['msg']}"
            for err in e.errors())
        raise ConfigError(f"invalid configuration from the environment - {problems}") from None


settings = _load_settings()
