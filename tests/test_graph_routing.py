from app.config import get_settings
from app.graph import route_after_evaluation
from app.schemas import EvaluationMetrics


def metrics(score: float) -> EvaluationMetrics:
    return EvaluationMetrics(
        citation_validity=score,
        citation_coverage=score,
        source_diversity=score,
        retrieval_quality=score,
        groundedness_proxy=score,
        overall_score=score,
    )


def test_low_score_routes_to_revision(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "evaluation_revision_threshold", 0.65)
    monkeypatch.setattr(settings, "max_revision_passes", 1)
    state = {"evaluation": metrics(0.4), "revision_count": 0}
    assert route_after_evaluation(state) == "revise"


def test_revision_limit_routes_to_finish(monkeypatch):
    settings = get_settings()
    monkeypatch.setattr(settings, "evaluation_revision_threshold", 0.65)
    monkeypatch.setattr(settings, "max_revision_passes", 1)
    state = {"evaluation": metrics(0.4), "revision_count": 1}
    assert route_after_evaluation(state) == "finish"
