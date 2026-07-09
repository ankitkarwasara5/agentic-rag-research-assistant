"""Final Report Agent."""

from __future__ import annotations

from app.llm import LocalLLM
from app.schemas import Citation, RetrievedContext
from app.utils.text_utils import safe_truncate


def _fallback_report(
    query: str,
    summary: str,
    contexts: list[RetrievedContext],
    citations: list[Citation],
    confidence_score: float,
) -> str:
    if not contexts:
        return (
            "## Executive Summary\n"
            "No source-backed answer could be produced because no usable web or "
            "document context was retrieved.\n\n"
            "## Key Findings\n"
            "- Insufficient evidence was available for a citation-backed report.\n\n"
            "## Limitations\n"
            "- Add PDF/TXT/MD documents or retry web search, then run the research again.\n"
            f"- Confidence score: {confidence_score:.2f}"
        )

    citation_text = ", ".join(f"[{citation.citation_id}]" for citation in citations)
    evidence_lines = "\n".join(
        f"- [{context.citation_id}] {context.title}: {safe_truncate(context.text, 220)}"
        for context in contexts[:6]
    )
    return (
        "## Executive Summary\n"
        f"This report addresses: {query}. The available evidence comes from "
        f"{len(citations)} source(s): {citation_text}.\n\n"
        "## Key Findings\n"
        f"{summary}\n\n"
        "## Evidence Snapshot\n"
        f"{evidence_lines}\n\n"
        "## Limitations\n"
        "- This fallback report uses retrieved excerpts directly because local LLM "
        "generation was unavailable.\n"
        f"- Confidence score: {confidence_score:.2f}"
    )


def generate_final_report(
    query: str,
    research_plan: list[str],
    summary: str,
    contexts: list[RetrievedContext],
    citations: list[Citation],
    confidence_score: float,
    validation_notes: str,
    llm: LocalLLM,
    warnings: list[str],
) -> str:
    """Generate a structured citation-backed Markdown report."""

    context_block = "\n\n".join(
        (
            f"[{context.citation_id}] {context.title}\n"
            f"{safe_truncate(context.text, 1000)}"
        )
        for context in contexts
    )
    citation_block = "\n".join(
        f"- [{citation.citation_id}] {citation.title} ({citation.url or citation.file_name})"
        for citation in citations
    )
    fallback = _fallback_report(query, summary, contexts, citations, confidence_score)
    prompt = f"""
Write a polished research report in Markdown.

Rules:
- Use only the retrieved context.
- Cite factual claims with labels like [Source 1].
- If evidence is incomplete, say so.
- Do not invent citations, URLs, statistics, or dates.
- Include these sections: Executive Summary, Research Plan, Key Findings,
  Source Analysis, Limitations, Confidence.

Query: {query}

Research plan:
{chr(10).join(f"- {item}" for item in research_plan)}

Evidence summary:
{summary}

Validation notes:
{validation_notes}

Citations:
{citation_block}

Retrieved context:
{context_block}

Confidence score: {confidence_score:.2f}
"""
    return llm.generate_or_fallback(
        prompt=prompt,
        fallback=fallback,
        warnings=warnings,
        system="You write careful source-grounded research reports.",
    )

