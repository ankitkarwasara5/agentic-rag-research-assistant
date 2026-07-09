"""Markdown report rendering and persistence."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from app.config import Settings, get_settings
from app.schemas import Citation, ResearchResponse
from app.utils.text_utils import stable_id


def render_markdown_report(response: ResearchResponse) -> str:
    """Render a complete downloadable Markdown report."""

    citation_lines = []
    for citation in response.citations:
        locator = citation.url or citation.file_name or "local source"
        citation_lines.append(
            f"- [{citation.citation_id}] {citation.title} ({citation.source_type}): {locator}"
        )

    plan_lines = [f"- {item}" for item in response.research_plan]
    warning_lines = [f"- {warning}" for warning in response.warnings]

    return "\n".join(
        [
            f"# Research Report: {response.query}",
            "",
            f"Generated: {datetime.now(UTC).isoformat()}",
            f"Confidence score: {response.confidence_score:.2f}",
            "",
            "## Research Plan",
            "\n".join(plan_lines) if plan_lines else "No plan generated.",
            "",
            "## Final Report",
            response.final_report,
            "",
            "## Citations",
            "\n".join(citation_lines) if citation_lines else "No citations available.",
            "",
            "## Warnings",
            "\n".join(warning_lines) if warning_lines else "None.",
            "",
        ]
    )


def write_report(
    response: ResearchResponse, settings: Settings | None = None
) -> Path:
    """Write a Markdown report under the reports directory."""

    settings = settings or get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    report_id = stable_id(response.query, prefix="report")
    output_path = settings.reports_dir / f"{report_id}.md"
    output_path.write_text(render_markdown_report(response), encoding="utf-8")
    return output_path

