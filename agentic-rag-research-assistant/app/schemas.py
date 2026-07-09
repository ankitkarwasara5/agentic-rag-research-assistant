"""Pydantic schemas and shared data models."""

from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


SourceType = Literal["web", "document"]


class UploadedFilePayload(BaseModel):
    """In-memory representation of an uploaded research file."""

    file_name: str
    content_type: str | None = None
    content: bytes


class SearchResult(BaseModel):
    """Normalized web search result."""

    title: str
    url: str
    snippet: str = ""


class RawSource(BaseModel):
    """Full text collected from a document or web page before chunking."""

    source_id: str
    source_type: SourceType
    title: str
    text: str
    url: str | None = None
    file_name: str | None = None


class RetrievedContext(BaseModel):
    """A retrieved chunk with source metadata and citation identity."""

    source_id: str
    source_type: SourceType
    title: str
    text: str
    score: float = 0.0
    citation_id: str
    chunk_id: str
    url: str | None = None
    file_name: str | None = None


class Citation(BaseModel):
    """A source used in the final report."""

    citation_id: str
    title: str
    source_type: SourceType
    url: str | None = None
    file_name: str | None = None
    snippet: str = ""


class ResearchRequest(BaseModel):
    """JSON-compatible research request used by tests and API clients."""

    query: str = Field(min_length=3)


class ResearchResponse(BaseModel):
    """Structured response returned by the FastAPI backend."""

    query: str
    research_plan: list[str]
    retrieved_context: list[RetrievedContext]
    final_report: str
    citations: list[Citation]
    confidence_score: float = Field(ge=0.0, le=1.0)
    refined_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    ollama_base_url: str
    primary_model: str
    embedding_model: str

