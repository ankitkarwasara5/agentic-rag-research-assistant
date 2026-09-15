"""Document extraction for PDF, TXT, and Markdown uploads."""

from __future__ import annotations

from pathlib import Path

from app.schemas import RawSource, UploadedFilePayload
from app.utils.text import normalize_whitespace, stable_id


class DocumentLoadError(RuntimeError):
    """Raised when an uploaded document cannot be extracted safely."""


def sanitize_filename(file_name: str) -> str:
    """Remove path components and unsafe NUL characters."""

    name = Path(file_name).name.replace("\x00", "").strip()
    return name or "uploaded_document"


def extract_pdf_text(content: bytes) -> str:
    """Extract plain text from a PDF byte stream."""

    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("PyMuPDF is not installed.") from exc

    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            text = "\n\n".join(page.get_text("text") for page in document)
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Could not read PDF: {exc}") from exc
    return normalize_whitespace(text)


def extract_text_upload(upload: UploadedFilePayload) -> RawSource | None:
    """Convert an uploaded supported file to a normalized source."""

    file_name = sanitize_filename(upload.file_name)
    suffix = Path(file_name).suffix.lower()
    if not upload.content:
        return None

    if suffix == ".pdf":
        text = extract_pdf_text(upload.content)
    elif suffix in {".txt", ".md", ".markdown"}:
        text = normalize_whitespace(upload.content.decode("utf-8", errors="replace"))
    else:
        raise DocumentLoadError(
            f"Unsupported file type for {file_name}. Use PDF, TXT, or MD."
        )

    if not text:
        return None
    return RawSource(
        source_id=stable_id(f"document:{file_name}:{len(upload.content)}"),
        source_type="document",
        title=file_name,
        file_name=file_name,
        text=text,
    )
