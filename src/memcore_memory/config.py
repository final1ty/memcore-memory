from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    data_dir: Path = Path.home() / ".memcore"
    db_path: Path = data_dir / "memory.db"
    vector_path: Path = data_dir / "vectors.hnsw"
    key_path: Path = data_dir / "master.key"
    audit_log_path: Path = data_dir / "audit.log"
    working_buffer_path: Path = data_dir / "working_buffer.jsonl"
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

    class Config:
        env_prefix = "MEMCORE_"
        extra = "allow"

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
