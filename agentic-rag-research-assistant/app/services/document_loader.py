"""Local document extraction for PDF, TXT, and Markdown uploads."""

from __future__ import annotations

from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import RawSource, UploadedFilePayload
from app.utils.text_utils import normalize_whitespace, stable_id


class DocumentLoadError(RuntimeError):
    """Raised when an uploaded document cannot be extracted."""


def sanitize_filename(file_name: str) -> str:
    """Return a safe file name without path components."""

    name = Path(file_name).name.strip().replace("\x00", "")
    return name or "uploaded_document"


def save_uploaded_file(
    upload: UploadedFilePayload, settings: Settings | None = None
) -> Path:
    """Persist an uploaded file under data/uploads."""

    settings = settings or get_settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / sanitize_filename(upload.file_name)
    target.write_bytes(upload.content)
    return target


def extract_pdf_text(content: bytes) -> str:
    """Extract text from a PDF byte stream with PyMuPDF."""

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError("PyMuPDF is not installed.") from exc

    try:
        with fitz.open(stream=content, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Could not read PDF: {exc}") from exc
    return normalize_whitespace("\n".join(pages))


def extract_text_upload(upload: UploadedFilePayload) -> RawSource | None:
    """Extract a supported uploaded file into a RawSource.

    Empty files return None so the caller can keep processing other inputs.
    """

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
            f"Unsupported file type for {file_name}. Use PDF, TXT, or MD files."
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

