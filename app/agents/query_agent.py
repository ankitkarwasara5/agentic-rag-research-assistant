"""Query decomposition agent."""

from app.services.ollama import OllamaError, OllamaService
from app.utils.text import parse_list


def understand_query(
    query: str,
    ollama: OllamaService,
    warnings: list[str],
) -> tuple[list[str], str | None]:
    """Generate focused research questions."""

    fallback = [
        query,
        f"What reliable evidence directly answers: {query}?",
        f"What disagreements, risks, or limitations matter for: {query}?",
    ]
    prompt = f"""Decompose the research request into 3-5 precise source-seeking
questions. Cover facts, evidence, counterpoints, and limitations. Return only
bullets.

Research request: {query}"""
    try:
        result = ollama.chat(
            prompt=prompt,
            system=(
                "You turn research requests into precise evidence-seeking questions."
            ),
        )
        return parse_list(result.content, fallback, 5), result.model
    except OllamaError as exc:
        warnings.append(str(exc))
        return fallback, None
