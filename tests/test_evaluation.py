from app.evaluation.metrics import evaluate_report
from app.schemas import RetrievedContext


def context(
    source_id: str,
    citation: str,
    score: float,
    text: str,
) -> RetrievedContext:
    return RetrievedContext(
        source_id=source_id,
        source_type="document",
        title=source_id,
        text=text,
        score=score,
        citation_id=citation,
        chunk_id=f"{source_id}:0",
    )


def test_evaluation_rewards_valid_cited_report():
    contexts = [
        context(
            "a",
            "Source 1",
            0.9,
            "RAG retrieves external evidence before generating an answer.",
        ),
        context(
            "b",
            "Source 2",
            0.8,
            "Agentic RAG can plan retrieval steps and revise its workflow.",
        ),
    ]
    report = (
        "Retrieval augmented generation uses external evidence before response "
        "generation [Source 1].\n\n"
        "Agentic RAG can add planning and iterative workflow steps [Source 2]."
    )
    metrics = evaluate_report(report, contexts)
    assert metrics.citation_validity == 1.0
    assert metrics.citation_coverage == 1.0
    assert metrics.overall_score > 0.6


def test_evaluation_detects_invalid_citation():
    contexts = [context("a", "Source 1", 0.7, "Grounded evidence.")]
    metrics = evaluate_report(
        "A substantive unsupported paragraph cites an unknown source [Source 9].",
        contexts,
    )
    assert metrics.citation_validity == 0.0
    assert metrics.notes
