"""Citation Validation Agent."""

from __future__ import annotations

from statistics import mean

from app.schemas import Citation, RetrievedContext
from app.utils.text_utils import safe_truncate


def build_citations(contexts: list[RetrievedContext]) -> list[Citation]:
    """Build one citation entry per unique citation id."""

    citations: list[Citation] = []
    seen: set[str] = set()
    for context in contexts:
        if context.citation_id in seen:
            continue
        seen.add(context.citation_id)
        citations.append(
            Citation(
                citation_id=context.citation_id,
                title=context.title,
                source_type=context.source_type,
                url=context.url,
                file_name=context.file_name,
                snippet=safe_truncate(context.text, 280),
            )
        )
    return citations


def validate_citations(
    contexts: list[RetrievedContext],
    warnings: list[str],
) -> tuple[list[Citation], float, str]:
    """Estimate confidence from source count, retrieval scores, and warnings."""

    citations = build_citations(contexts)
    if not contexts:
        return citations, 0.15, "No retrieved context was available to support claims."

    unique_sources = len({context.citation_id for context in contexts})
    avg_score = mean(max(0.0, min(1.0, context.score)) for context in contexts)
    confidence = 0.25 + (0.1 * min(unique_sources, 4)) + (0.25 * avg_score)

    if any("fallback" in warning.lower() for warning in warnings):
        confidence -= 0.1
    if any("search failed" in warning.lower() for warning in warnings):
        confidence -= 0.08
    if unique_sources < 2:
        confidence -= 0.12

    confidence = max(0.1, min(0.92, confidence))
    notes = (
        f"Validated {len(contexts)} retrieved chunks across {unique_sources} "
        f"unique cited source(s). Claims should use the listed citation labels only."
    )
    return citations, round(confidence, 2), notes

