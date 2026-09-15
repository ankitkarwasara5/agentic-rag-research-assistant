"""LangGraph orchestration for the agentic RAG workflow."""

from __future__ import annotations

from typing import Any, TypedDict
from uuid import uuid4

from app.agents.planner_agent import create_plan
from app.agents.query_agent import understand_query
from app.agents.report_agent import build_citations, draft_report, revise_report
from app.agents.retrieval_agent import retrieve_evidence
from app.agents.synthesis_agent import synthesize_evidence
from app.config import get_settings
from app.evaluation.metrics import evaluate_report
from app.schemas import (
    Citation,
    EvaluationMetrics,
    RawSource,
    ResearchResponse,
    RetrievedContext,
    UploadedFilePayload,
)
from app.services.document_loader import DocumentLoadError, extract_text_upload
from app.services.ollama import OllamaService
from app.services.web_search import WebSearchError, result_to_source, search_web


class ResearchState(TypedDict, total=False):
    run_id: str
    query: str
    uploaded_files: list[UploadedFilePayload]
    refined_questions: list[str]
    research_plan: list[str]
    sources: list[RawSource]
    retrieved_context: list[RetrievedContext]
    synthesis: str
    final_report: str
    citations: list[Citation]
    evaluation: EvaluationMetrics
    warnings: list[str]
    model_used: str | None
    retrieval_mode: str
    revision_count: int


def _warnings(state: ResearchState) -> list[str]:
    return list(state.get("warnings", []))


def _set_model(state: ResearchState, model: str | None) -> str | None:
    return model or state.get("model_used")


def query_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    questions, model = understand_query(
        state["query"],
        OllamaService(),
        warnings,
    )
    return {
        "refined_questions": questions,
        "warnings": warnings,
        "model_used": _set_model(state, model),
    }


def planning_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    plan, model = create_plan(
        state["query"],
        state.get("refined_questions", [state["query"]]),
        OllamaService(),
        warnings,
    )
    return {
        "research_plan": plan,
        "warnings": warnings,
        "model_used": _set_model(state, model),
    }


def source_collection_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    sources: list[RawSource] = []
    settings = get_settings()

    try:
        results = search_web(state.get("refined_questions", [state["query"]]), settings)
        sources.extend(result_to_source(result, settings) for result in results)
    except WebSearchError as exc:
        warnings.append(str(exc))

    for upload in state.get("uploaded_files", []):
        try:
            source = extract_text_upload(upload)
            if source:
                sources.append(source)
        except DocumentLoadError as exc:
            warnings.append(f"{upload.file_name}: {exc}")

    if not sources:
        warnings.append("No usable web or uploaded-document sources were collected.")
    return {"sources": sources, "warnings": warnings}


def retrieval_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    contexts, mode = retrieve_evidence(
        state.get("sources", []),
        state["query"],
        state["run_id"],
        get_settings(),
        warnings,
    )
    return {
        "retrieved_context": contexts,
        "retrieval_mode": mode,
        "warnings": warnings,
    }


def synthesis_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    synthesis, model = synthesize_evidence(
        state["query"],
        state.get("retrieved_context", []),
        OllamaService(),
        warnings,
    )
    return {
        "synthesis": synthesis,
        "warnings": warnings,
        "model_used": _set_model(state, model),
    }


def report_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    report, model = draft_report(
        state["query"],
        state.get("research_plan", []),
        state.get("synthesis", ""),
        state.get("retrieved_context", []),
        OllamaService(),
        warnings,
    )
    contexts = state.get("retrieved_context", [])
    return {
        "final_report": report,
        "citations": build_citations(contexts),
        "warnings": warnings,
        "model_used": _set_model(state, model),
    }


def evaluation_node(state: ResearchState) -> ResearchState:
    evaluation = evaluate_report(
        state.get("final_report", ""),
        state.get("retrieved_context", []),
    )
    return {"evaluation": evaluation}


def revision_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    report, model = revise_report(
        state["query"],
        state.get("final_report", ""),
        state.get("retrieved_context", []),
        state["evaluation"],
        OllamaService(),
        warnings,
    )
    return {
        "final_report": report,
        "revision_count": state.get("revision_count", 0) + 1,
        "warnings": warnings,
        "model_used": _set_model(state, model),
    }


def route_after_evaluation(state: ResearchState) -> str:
    settings = get_settings()
    score = state["evaluation"].overall_score
    revisions = state.get("revision_count", 0)
    if (
        score < settings.evaluation_revision_threshold
        and revisions < settings.max_revision_passes
    ):
        return "revise"
    return "finish"


def build_research_graph():
    """Compile the LangGraph workflow lazily so unit tests remain lightweight."""

    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("langgraph is not installed") from exc

    graph = StateGraph(ResearchState)
    graph.add_node("understand_query", query_node)
    graph.add_node("plan", planning_node)
    graph.add_node("collect_sources", source_collection_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("synthesize", synthesis_node)
    graph.add_node("draft_report", report_node)
    graph.add_node("evaluate", evaluation_node)
    graph.add_node("revise", revision_node)

    graph.add_edge(START, "understand_query")
    graph.add_edge("understand_query", "plan")
    graph.add_edge("plan", "collect_sources")
    graph.add_edge("collect_sources", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "draft_report")
    graph.add_edge("draft_report", "evaluate")
    graph.add_conditional_edges(
        "evaluate",
        route_after_evaluation,
        {"revise": "revise", "finish": END},
    )
    graph.add_edge("revise", "evaluate")
    return graph.compile()


def run_research(
    query: str,
    uploaded_files: list[UploadedFilePayload] | None = None,
) -> ResearchResponse:
    """Run the full research workflow."""

    cleaned = query.strip()
    if len(cleaned) < 3:
        raise ValueError("Query must be at least 3 characters long.")

    initial: ResearchState = {
        "run_id": uuid4().hex,
        "query": cleaned,
        "uploaded_files": uploaded_files or [],
        "warnings": [],
        "revision_count": 0,
        "retrieval_mode": "none",
    }
    final: dict[str, Any] = build_research_graph().invoke(initial)
    return ResearchResponse(
        run_id=final["run_id"],
        query=cleaned,
        research_plan=final.get("research_plan", []),
        refined_questions=final.get("refined_questions", []),
        retrieved_context=final.get("retrieved_context", []),
        final_report=final.get("final_report", ""),
        citations=final.get("citations", []),
        evaluation=final.get("evaluation")
        or evaluate_report("", final.get("retrieved_context", [])),
        warnings=final.get("warnings", []),
        model_used=final.get("model_used"),
        retrieval_mode=final.get("retrieval_mode", "none"),
        revision_count=final.get("revision_count", 0),
    )
