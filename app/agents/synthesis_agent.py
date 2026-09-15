"""Evidence synthesis agent."""

from app.schemas import RetrievedContext
from app.services.ollama import OllamaError, OllamaService
from app.utils.text import safe_truncate


def synthesize_evidence(
    query: str,
    contexts: list[RetrievedContext],
    ollama: OllamaService,
    warnings: list[str],
) -> tuple[str, str | None]:
    """Synthesize retrieved evidence without inventing new sources."""

    if not contexts:
        return "No source-backed evidence was retrieved.", None
    evidence = "\n\n".join(
        f"[{item.citation_id}] {item.title}\n{safe_truncate(item.text, 1400)}"
        for item in contexts
    )
    fallback = "\n".join(
        f"- [{item.citation_id}] {item.title}: {safe_truncate(item.text, 280)}"
        for item in contexts
    )
    prompt = f"""Synthesize the evidence for the research query below.
Use only the supplied evidence. Preserve citation labels exactly. Explicitly
note source agreement, disagreement, uncertainty, and missing information.

Query: {query}

Evidence:
{evidence}"""
    try:
        result = ollama.chat(
            prompt,
            system="You synthesize evidence without adding unsupported claims.",
        )
        return result.content, result.model
    except OllamaError as exc:
        warnings.append(str(exc))
        return fallback, None
