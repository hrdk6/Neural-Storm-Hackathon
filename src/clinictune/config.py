"""Runtime settings, read from environment variables prefixed ``CLINICTUNE_``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CLINICTUNE_", env_file=".env", extra="ignore")

    bundle_dir: Path = Field(Path("models/demo"), description="Model bundle directory (manifest.json inside).")
    data_dir: Path = Field(Path("var"), description="Writable dir for SQLite DB and site embeddings.")
    api_key: str | None = Field(None, description="If set, every /v1 request needs header X-API-Key.")
    cors_origins: list[str] = Field(default_factory=list)
    max_upload_mb: float = 25.0
    max_onboard_zip_mb: float = 600.0
    max_onboard_images: int = 2000
    ort_threads: int = Field(0, description="ONNX Runtime intra-op threads; 0 lets ORT decide.")
    onboard_workers: int = 1
    log_level: str = "INFO"
    log_json: bool = True
    store_audit_log: bool = True

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clinictune.sqlite3"


@lru_cache
def get_settings() -> Settings:
    return Settings()
