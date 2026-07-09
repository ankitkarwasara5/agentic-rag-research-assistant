"""Summarization Agent."""

from __future__ import annotations

from app.llm import LocalLLM
from app.schemas import RetrievedContext
from app.utils.text_utils import safe_truncate


def summarize_context(
    query: str,
    contexts: list[RetrievedContext],
    llm: LocalLLM,
    warnings: list[str],
) -> str:
    """Summarize retrieved chunks with source references."""

    if not contexts:
        return "No relevant context was retrieved."

    context_block = "\n\n".join(
        (
            f"[{context.citation_id}] {context.title}\n"
            f"{safe_truncate(context.text, 1200)}"
        )
        for context in contexts
    )
    fallback = "\n".join(
        f"- [{context.citation_id}] {context.title}: {safe_truncate(context.text, 260)}"
        for context in contexts
    )
    prompt = f"""
Summarize the retrieved evidence for the query below. Use citation labels exactly
as provided, such as [Source 1]. Keep claims tied to citations.

Query: {query}

Retrieved context:
{context_block}
"""
    return llm.generate_or_fallback(
        prompt=prompt,
        fallback=fallback,
        warnings=warnings,
        system="You summarize research evidence without adding unsupported claims.",
    )

