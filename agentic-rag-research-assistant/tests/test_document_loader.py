from app.schemas import UploadedFilePayload
from app.services.document_loader import extract_text_upload, sanitize_filename


def test_sanitize_filename_removes_paths() -> None:
    assert sanitize_filename("../notes.md") == "notes.md"


def test_extract_text_upload_reads_markdown() -> None:
    upload = UploadedFilePayload(
        file_name="notes.md",
        content_type="text/markdown",
        content=b"# Agentic RAG\n\nLocal retrieval and planning.",
    )

    source = extract_text_upload(upload)

    assert source is not None
    assert source.source_type == "document"
    assert source.file_name == "notes.md"
    assert "Local retrieval" in source.text


def test_extract_text_upload_skips_empty_file() -> None:
    upload = UploadedFilePayload(file_name="empty.txt", content=b"")

    assert extract_text_upload(upload) is None

