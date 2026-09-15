import pytest

from app.schemas import UploadedFilePayload
from app.services.document_loader import (
    DocumentLoadError,
    extract_text_upload,
    sanitize_filename,
)


def test_sanitize_filename_removes_directories():
    assert sanitize_filename("../../notes.md") == "notes.md"


def test_extract_markdown_document():
    source = extract_text_upload(
        UploadedFilePayload(
            file_name="notes.md",
            content=b"# Heading\nUseful evidence",
        )
    )
    assert source is not None
    assert source.source_type == "document"
    assert "Useful evidence" in source.text


def test_rejects_unsupported_upload():
    with pytest.raises(DocumentLoadError):
        extract_text_upload(
            UploadedFilePayload(file_name="image.png", content=b"not an image")
        )
