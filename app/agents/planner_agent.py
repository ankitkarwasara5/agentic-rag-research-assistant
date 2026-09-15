"""Research planning agent."""

from app.services.ollama import OllamaError, OllamaService
from app.utils.text import parse_list


def create_plan(
    query: str,
    questions: list[str],
    ollama: OllamaService,
    warnings: list[str],
) -> tuple[list[str], str | None]:
    """Create an execution-oriented research plan."""

    fallback = [
        "Clarify the scope and identify claims that require evidence.",
        "Collect independent web sources and any uploaded documents.",
        "Retrieve the most relevant evidence chunks for the query.",
        "Compare agreement, disagreement, and limitations across sources.",
        "Draft a citation-backed report and evaluate its evidence coverage.",
    ]
    prompt = f"""Create a 5-7 step research plan. Keep each step actionable.

Query: {query}
Questions:
{chr(10).join(f"- {item}" for item in questions)}

Return only bullets."""
    try:
        result = ollama.chat(prompt, system="You are a rigorous research planner.")
        return parse_list(result.content, fallback, 7), result.model
    except OllamaError as exc:
        warnings.append(str(exc))
        return fallback, None
