"""Document Ingestion Agent."""

from __future__ import annotations

from app.schemas import RawSource, UploadedFilePayload
from app.services.document_loader import (
    DocumentLoadError,
    extract_text_upload,
    save_uploaded_file,
)


def ingest_documents(
    uploads: list[UploadedFilePayload],
    warnings: list[str],
) -> list[RawSource]:
    """Extract text from uploaded files and keep processing on partial failure."""

    sources: list[RawSource] = []
    for upload in uploads:
        try:
            source = extract_text_upload(upload)
            if source is None:
                warnings.append(f"Skipped empty file: {upload.file_name}")
                continue
            save_uploaded_file(upload)
            sources.append(source)
        except DocumentLoadError as exc:
            warnings.append(str(exc))
    return sources
