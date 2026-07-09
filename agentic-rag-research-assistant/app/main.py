"""FastAPI backend for the Agentic RAG Research Assistant."""

from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.graph import run_research
from app.schemas import HealthResponse, ResearchRequest, ResearchResponse, UploadedFilePayload


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Fully local Agentic RAG research backend using Ollama, LangGraph, and ChromaDB.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return backend health and model configuration."""

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        ollama_base_url=settings.ollama_base_url,
        primary_model=settings.primary_llm_model,
        embedding_model=settings.embedding_model,
    )


async def _read_uploads(files: list[UploadFile] | None) -> list[UploadedFilePayload]:
    uploads: list[UploadedFilePayload] = []
    for file in files or []:
        content = await file.read()
        if len(content) > settings.max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"{file.filename} exceeds the configured upload size limit.",
            )
        uploads.append(
            UploadedFilePayload(
                file_name=file.filename or "uploaded_document",
                content_type=file.content_type,
                content=content,
            )
        )
    return uploads


@app.post("/research", response_model=ResearchResponse)
async def research(
    request: Request,
    query: str | None = Form(default=None),
    files: list[UploadFile] | None = File(default=None),
) -> ResearchResponse:
    """Run research from multipart form data or a JSON body."""

    uploads: list[UploadedFilePayload] = []
    if query is None:
        try:
            payload = ResearchRequest.model_validate(await request.json())
            query = payload.query
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=422,
                detail="Provide a research query in form field `query` or JSON body.",
            ) from exc
    else:
        uploads = await _read_uploads(files)

    try:
        return run_research(query=query, uploaded_files=uploads)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

