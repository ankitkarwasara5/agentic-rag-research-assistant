"""Query Understanding Agent."""

from __future__ import annotations

from app.llm import LocalLLM
from app.utils.text_utils import parse_list


def understand_query(query: str, llm: LocalLLM, warnings: list[str]) -> list[str]:
    """Generate focused search questions from the user's query."""

    fallback = [
        query,
        f"What are the key facts and recent developments about {query}?",
        f"What evidence, sources, and limitations matter for {query}?",
    ]
    prompt = f"""
Turn this research query into 3 to 5 precise search questions.
Return only a short bullet list.

Query: {query}
"""
    text = llm.generate_or_fallback(
        prompt=prompt,
        fallback="\n".join(f"- {item}" for item in fallback),
        warnings=warnings,
        system="You refine research queries into concise, source-seeking questions.",
    )
    return parse_list(text, fallback=fallback, limit=5)

