"""Citation-backed report generation and revision."""

from app.schemas import Citation, EvaluationMetrics, RetrievedContext
from app.services.ollama import OllamaError, OllamaService
from app.utils.text import safe_truncate


def build_citations(contexts: list[RetrievedContext]) -> list[Citation]:
    """Return one citation entry per retrieved source."""

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
                snippet=safe_truncate(context.text, 320),
            )
        )
    return citations


def _fallback_report(
    query: str,
    synthesis: str,
    contexts: list[RetrievedContext],
) -> str:
    if not contexts:
        return (
            "# Research Report\n\n"
            "## Executive Summary\n\n"
            "No source-backed answer could be produced because no usable evidence "
            "was retrieved.\n\n"
            "## Limitations\n\n"
            "- Add documents or retry web research."
        )
    evidence = "\n".join(
        f"- [{item.citation_id}] **{item.title}** — {safe_truncate(item.text, 260)}"
        for item in contexts[:8]
    )
    return f"""# Research Report

## Executive Summary

This fallback report addresses **{query}** using the retrieved evidence below.

## Evidence Synthesis

{synthesis}

## Evidence Snapshot

{evidence}

## Limitations

- Local model generation was unavailable, so this report avoids adding claims
  beyond the retrieved evidence.
"""


def draft_report(
    query: str,
    plan: list[str],
    synthesis: str,
    contexts: list[RetrievedContext],
    ollama: OllamaService,
    warnings: list[str],
) -> tuple[str, str | None]:
    """Write a structured source-grounded Markdown report."""

    if not contexts:
        return _fallback_report(query, synthesis, contexts), None

    evidence = "\n\n".join(
        f"[{item.citation_id}] {item.title}\n{safe_truncate(item.text, 1300)}"
        for item in contexts
    )
    prompt = f"""Write a careful research report in Markdown.

Rules:
- Use only the supplied evidence.
- Every substantive factual paragraph must include at least one supplied
  citation such as [Source 1].
- Never invent sources, URLs, dates, statistics, or citation labels.
- Clearly distinguish evidence from inference.
- Discuss disagreements and limitations when evidence is incomplete.
- Use sections: Executive Summary, Research Approach, Key Findings,
  Source Analysis, Limitations, Conclusion.

Query: {query}

Research plan:
{chr(10).join(f"- {item}" for item in plan)}

Evidence synthesis:
{synthesis}

Retrieved evidence:
{evidence}"""
    try:
        result = ollama.chat(
            prompt,
            system="You write rigorous, source-grounded research reports.",
        )
        return result.content, result.model
    except OllamaError as exc:
        warnings.append(str(exc))
        return _fallback_report(query, synthesis, contexts), None


def revise_report(
    query: str,
    report: str,
    contexts: list[RetrievedContext],
    evaluation: EvaluationMetrics,
    ollama: OllamaService,
    warnings: list[str],
) -> tuple[str, str | None]:
    """Perform one evidence-focused revision when evaluation is weak."""

    evidence = "\n\n".join(
        f"[{item.citation_id}] {item.title}\n{safe_truncate(item.text, 1000)}"
        for item in contexts
    )
    notes = "\n".join(f"- {item}" for item in evaluation.notes)
    prompt = f"""Revise the report to improve citation validity, citation coverage,
and evidence grounding. Do not add any unsupported claim or new citation.

Query: {query}

Evaluation notes:
{notes or "- Improve evidence coverage."}

Current report:
{report}

Allowed evidence:
{evidence}

Return only the revised Markdown report."""
    try:
        result = ollama.chat(
            prompt,
            system="You revise research reports for grounding and citation quality.",
        )
        return result.content, result.model
    except OllamaError as exc:
        warnings.append(f"Report revision skipped: {exc}")
        return report, None
