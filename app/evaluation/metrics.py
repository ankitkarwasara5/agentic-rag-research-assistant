"""Transparent heuristic metrics for RAG report quality."""

from __future__ import annotations

from statistics import mean

from app.schemas import EvaluationMetrics, RetrievedContext
from app.utils.text import extract_citation_ids, significant_terms


def _cited_paragraphs(report: str) -> list[str]:
    return [
        paragraph.strip()
        for paragraph in report.split("\n\n")
        if len(significant_terms(paragraph)) >= 6
    ]


def _groundedness_proxy(report: str, contexts: list[RetrievedContext]) -> float:
    """Estimate lexical evidence overlap for paragraphs containing citations."""

    by_citation: dict[str, str] = {}
    for context in contexts:
        by_citation.setdefault(context.citation_id, "")
        by_citation[context.citation_id] += " " + context.text

    scores: list[float] = []
    for paragraph in _cited_paragraphs(report):
        citation_ids = extract_citation_ids(paragraph)
        if not citation_ids:
            continue
        evidence = " ".join(by_citation.get(cid, "") for cid in citation_ids)
        paragraph_terms = significant_terms(paragraph)
        evidence_terms = significant_terms(evidence)
        if paragraph_terms:
            scores.append(len(paragraph_terms & evidence_terms) / len(paragraph_terms))
    return round(mean(scores), 4) if scores else 0.0


def evaluate_report(
    report: str,
    contexts: list[RetrievedContext],
) -> EvaluationMetrics:
    """Score citation hygiene, source mix, retrieval, and lexical grounding.

    These are deterministic engineering diagnostics, not claims of factual truth.
    """

    valid_ids = {context.citation_id for context in contexts if context.citation_id}
    used_ids = extract_citation_ids(report)

    citation_validity = (
        sum(1 for item in used_ids if item in valid_ids) / len(used_ids)
        if used_ids
        else 0.0
    )

    paragraphs = _cited_paragraphs(report)
    citation_coverage = (
        sum(1 for paragraph in paragraphs if extract_citation_ids(paragraph))
        / len(paragraphs)
        if paragraphs
        else 0.0
    )

    unique_sources = len({context.source_id for context in contexts})
    source_diversity = min(1.0, unique_sources / 3.0)
    retrieval_quality = mean(context.score for context in contexts) if contexts else 0.0
    groundedness = _groundedness_proxy(report, contexts)

    overall = (
        0.25 * citation_validity
        + 0.25 * citation_coverage
        + 0.20 * source_diversity
        + 0.15 * retrieval_quality
        + 0.15 * groundedness
    )

    notes: list[str] = []
    if not contexts:
        notes.append("No retrieved evidence was available.")
    if citation_validity < 1.0:
        notes.append("One or more citation labels are missing or invalid.")
    if citation_coverage < 0.6:
        notes.append("Many substantive paragraphs are not explicitly cited.")
    if unique_sources < 2:
        notes.append("Source diversity is limited; add independent evidence.")
    if groundedness < 0.35 and contexts:
        notes.append(
            "Low lexical overlap suggests a manual grounding review is useful."
        )

    return EvaluationMetrics(
        citation_validity=round(citation_validity, 4),
        citation_coverage=round(citation_coverage, 4),
        source_diversity=round(source_diversity, 4),
        retrieval_quality=round(retrieval_quality, 4),
        groundedness_proxy=round(groundedness, 4),
        overall_score=round(max(0.0, min(1.0, overall)), 4),
        notes=notes,
    )
