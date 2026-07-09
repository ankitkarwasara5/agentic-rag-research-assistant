"""Application configuration.

Defaults are tuned for a 16 GB Apple Silicon MacBook. Ollama uses Apple's
Metal acceleration automatically on macOS when available; the settings below
keep context and generation sizes practical for qwen3:8b on an M3 machine.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime settings loaded from defaults and optional environment vars."""

    app_name: str = "Agentic RAG Research Assistant"
    api_host: str = "127.0.0.1"
    api_port: int = 8000

    ollama_base_url: str = "http://localhost:11434"
    primary_llm_model: str = "qwen3:8b"
    fallback_llm_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text"

    llm_temperature: float = 0.2
    ollama_num_ctx: int = 4096
    ollama_num_predict: int = 1400
    ollama_num_thread: int = 8
    ollama_num_gpu: int = 1
    ollama_keep_alive: str = "10m"

    chunk_size: int = 800
    chunk_overlap: int = 120
    retrieval_top_k: int = 6
    web_max_results: int = 5
    web_request_timeout: int = 12
    max_file_bytes: int = 15 * 1024 * 1024
    collection_name: str = "agentic_rag_research"
    user_agent: str = (
        "AgenticRAGResearchAssistant/1.0 "
        "(local research tool; +https://localhost)"
    )

    upload_dir: Path = Field(default=PROJECT_ROOT / "data" / "uploads")
    chroma_dir: Path = Field(default=PROJECT_ROOT / "data" / "chroma_db")
    reports_dir: Path = Field(default=PROJECT_ROOT / "reports")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="RAG_",
        extra="ignore",
    )

    def ensure_directories(self) -> None:
        """Create runtime directories if they are missing."""

        for path in (self.upload_dir, self.chroma_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings and ensure required directories exist."""

    settings = Settings()
    settings.ensure_directories()
    return settings

