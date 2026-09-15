"""FastAPI backend for the Agentic RAG Research Assistant."""

from __future__ import annotations

from typing import Annotated

import anyio
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import Settings, get_settings
from app.graph import run_research
from app.schemas import (
    HealthResponse,
    ResearchRequest,
    ResearchResponse,
    UploadedFilePayload,
)
from app.services.ollama import OllamaService
from app.services.report_writer import write_report


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the API app with explicit settings for testability."""

    settings = settings or get_settings()
    settings.ensure_directories()
    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Local-first agentic RAG research API with LangGraph orchestration, "
            "Ollama inference, Chroma retrieval, and deterministic evaluation."
        ),
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root() -> dict[str, str]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "health": "/health",
            "research": "POST /research",
        }

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        reachable, models = OllamaService(settings).health()

        def model_available(configured: str) -> bool:
            return any(
                model == configured or model.startswith(f"{configured}:")
                for model in models
            )

        primary_available = model_available(settings.primary_llm_model)
        embedding_available = model_available(settings.embedding_model)
        healthy = reachable and primary_available and embedding_available
        return HealthResponse(
            status="healthy" if healthy else "degraded",
            app_name=settings.app_name,
            version=__version__,
            ollama_reachable=reachable,
            installed_models=models,
            primary_model=settings.primary_llm_model,
            embedding_model=settings.embedding_model,
            primary_model_available=primary_available,
            embedding_model_available=embedding_available,
        )

    async def read_uploads(
        files: list[UploadFile] | None,
    ) -> list[UploadedFilePayload]:
        uploads: list[UploadedFilePayload] = []
        for file in files or []:
            content = await file.read()
            if len(content) > settings.max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"{file.filename} exceeds the upload size limit.",
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
        query: Annotated[str | None, Form()] = None,
        files: Annotated[list[UploadFile] | None, File()] = None,
    ) -> ResearchResponse:
        uploads: list[UploadedFilePayload] = []
        if query is None:
            try:
                payload = ResearchRequest.model_validate(await request.json())
                query = payload.query
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(
                    status_code=422,
                    detail="Provide `query` as form data or a JSON body.",
                ) from exc
        else:
            uploads = await read_uploads(files)

        try:
            response = await anyio.to_thread.run_sync(
                lambda: run_research(query=query or "", uploaded_files=uploads)
            )
            write_report(response, settings)
            return response
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
