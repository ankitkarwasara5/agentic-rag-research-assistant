"""Pydantic models shared across API, agents, and services."""

from typing import Literal

from pydantic import BaseModel, Field

SourceType = Literal["web", "document"]


class UploadedFilePayload(BaseModel):
    """In-memory representation of an uploaded research document."""

    file_name: str
    content_type: str | None = None
    content: bytes


class SearchResult(BaseModel):
    """Normalized web search result."""

    title: str
    url: str
    snippet: str = ""


class RawSource(BaseModel):
    """Collected source before chunking and retrieval."""

    source_id: str
    source_type: SourceType
    title: str
    text: str
    url: str | None = None
    file_name: str | None = None


class RetrievedContext(BaseModel):
    """Retrieved evidence chunk."""

    source_id: str
    source_type: SourceType
    title: str
    text: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_id: str
    chunk_id: str
    url: str | None = None
    file_name: str | None = None


class Citation(BaseModel):
    """Source metadata exposed to the user."""

    citation_id: str
    title: str
    source_type: SourceType
    url: str | None = None
    file_name: str | None = None
    snippet: str = ""


class EvaluationMetrics(BaseModel):
    """Heuristic quality metrics for a completed research report."""

    citation_validity: float = Field(ge=0.0, le=1.0)
    citation_coverage: float = Field(ge=0.0, le=1.0)
    source_diversity: float = Field(ge=0.0, le=1.0)
    retrieval_quality: float = Field(ge=0.0, le=1.0)
    groundedness_proxy: float = Field(ge=0.0, le=1.0)
    overall_score: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


class ResearchRequest(BaseModel):
    """JSON research request."""

    query: str = Field(min_length=3, max_length=2000)


class ResearchResponse(BaseModel):
    """Complete research result returned by the backend."""

    run_id: str
    query: str
    research_plan: list[str]
    refined_questions: list[str]
    retrieved_context: list[RetrievedContext]
    final_report: str
    citations: list[Citation]
    evaluation: EvaluationMetrics
    warnings: list[str] = Field(default_factory=list)
    model_used: str | None = None
    retrieval_mode: str
    revision_count: int = 0


class HealthResponse(BaseModel):
    """Dependency-aware backend health response."""

    status: Literal["healthy", "degraded"]
    app_name: str
    version: str
    ollama_reachable: bool
    installed_models: list[str]
    primary_model: str
    embedding_model: str
    primary_model_available: bool
    embedding_model_available: bool
