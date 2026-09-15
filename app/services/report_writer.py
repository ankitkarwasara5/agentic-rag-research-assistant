"""Persist generated Markdown reports and metadata."""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import ResearchResponse


def write_report(
    response: ResearchResponse,
    settings: Settings | None = None,
) -> tuple[Path, Path]:
    """Save Markdown and JSON metadata for a research run."""

    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = settings.reports_dir / f"{response.run_id}.md"
    metadata_path = settings.reports_dir / f"{response.run_id}.json"
    markdown_path.write_text(response.final_report, encoding="utf-8")
    metadata_path.write_text(
        json.dumps(response.model_dump(mode="json"), indent=2),
        encoding="utf-8",
    )
    return markdown_path, metadata_path
