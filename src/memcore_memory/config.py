from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator
from pathlib import Path
from typing import Optional

# Paths derived from data_dir, and the filename each one gets inside it.
_DERIVED_PATHS = {
    "db_path": "memory.db",
    "vector_path": "vectors.hnsw",
    "key_path": "master.key",
    "audit_log_path": "audit.log",
    "working_buffer_path": "working_buffer.jsonl",
}

class Settings(BaseSettings):
    # Every Dockerfile, compose file and doc in this project sets MNEM_* - the prefix
    # must match or the whole environment is silently ignored.
    model_config = SettingsConfigDict(env_prefix="MNEM_", extra="allow")

    data_dir: Path = Path.home() / ".memcore"
    # Left as None so they can be resolved against the *effective* data_dir below.
    # Defining them as `data_dir / "..."` would freeze them to the default at class
    # creation time, silently ignoring MNEM_DATA_DIR. Set any of them explicitly
    # (e.g. MNEM_DB_PATH) to override just that one.
    db_path: Optional[Path] = None
    vector_path: Optional[Path] = None
    key_path: Optional[Path] = None
    audit_log_path: Optional[Path] = None
    working_buffer_path: Optional[Path] = None
    p2p_port: int = 7742
    p2p_peers: list[str] = []

    embedding_provider: str = "bge-small"
    embedding_dim: int = 384
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: Optional[str] = None
    matryoshka_enabled: bool = True
    binary_quantization: bool = False
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-large"

    backend: str = "sqlite"
    database_url: str = "postgresql+asyncpg://memcore:memcore@localhost:5432/memcore"

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
    episodic_decay_rate: float = 0.1
    semantic_consolidation_threshold: int = 3
    forgetting_model: str = "exponential"

    @model_validator(mode="after")
    def _resolve_paths(self):
        for field, filename in _DERIVED_PATHS.items():
            if getattr(self, field) is None:
                object.__setattr__(self, field, self.data_dir / filename)
        return self

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
