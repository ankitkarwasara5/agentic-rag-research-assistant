"""Research Planning Agent."""

from __future__ import annotations

from app.llm import LocalLLM
from app.utils.text_utils import parse_list


def create_research_plan(
    query: str,
    refined_questions: list[str],
    llm: LocalLLM,
    warnings: list[str],
) -> list[str]:
    """Break the research task into executable steps."""

    fallback = [
        "Clarify the research scope and important subtopics.",
        "Collect web and uploaded-document sources relevant to the query.",
        "Retrieve the highest-signal chunks from the local vector database.",
        "Summarize findings and compare source agreement.",
        "Write a citation-backed report with limitations and confidence.",
    ]
    prompt = f"""
Create a practical research plan for this query.

Query: {query}
Refined questions:
{chr(10).join(f"- {question}" for question in refined_questions)}

Return 5 to 7 concise bullet points. No preamble.
"""
    text = llm.generate_or_fallback(
        prompt=prompt,
        fallback="\n".join(f"- {item}" for item in fallback),
        warnings=warnings,
        system="You are a research planning agent.",
    )
    return parse_list(text, fallback=fallback, limit=7)

