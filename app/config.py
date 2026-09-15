"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables and `.env`."""

    app_name: str = "Agentic RAG Research Assistant"
    app_version: str = "2.0.0"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    ollama_base_url: str = "http://127.0.0.1:11434"
    primary_llm_model: str = "qwen3:8b"
    fallback_llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"
    disable_thinking: bool = True
    llm_temperature: float = 0.2
    ollama_connect_timeout_seconds: float = 8.0
    ollama_generation_timeout_seconds: float = 300.0

    chunk_size: int = 900
    chunk_overlap: int = 140
    retrieval_top_k: int = Field(default=8, ge=2, le=20)
    web_max_results: int = Field(default=6, ge=1, le=20)
    web_request_timeout_seconds: float = 12.0
    max_file_bytes: int = 15 * 1024 * 1024
    max_source_chars: int = 30_000
    collection_name: str = "agentic_rag_research_v2"

    evaluation_revision_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    max_revision_passes: int = Field(default=1, ge=0, le=3)

    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="RAG_",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create runtime directories when they do not exist."""

        for path in (self.upload_dir, self.chroma_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    settings = Settings()
    settings.ensure_directories()
    return settings
