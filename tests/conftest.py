from pathlib import Path

import pytest

from app.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        reports_dir=tmp_path / "reports",
    )
