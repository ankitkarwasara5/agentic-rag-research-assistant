"""LangGraph workflow for the agentic RAG research process."""

from __future__ import annotations

from typing import TypedDict
from uuid import uuid4

from app.agents.document_agent import ingest_documents
from app.agents.planner_agent import create_research_plan
from app.agents.query_agent import understand_query
from app.agents.report_agent import generate_final_report
from app.agents.retrieval_agent import retrieve_context
from app.agents.summarizer_agent import summarize_context
from app.agents.validator_agent import validate_citations
from app.agents.web_search_agent import WebSearchError, search_web
from app.config import get_settings
from app.llm import get_local_llm
from app.schemas import RawSource, ResearchResponse, SearchResult, UploadedFilePayload
from app.services.report_writer import write_report
from app.services.web_loader import search_result_to_source

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]


class ResearchState(TypedDict, total=False):
    """Mutable state passed between graph nodes."""

    run_id: str
    query: str
    uploaded_files: list[UploadedFilePayload]
    refined_questions: list[str]
    research_plan: list[str]
    search_results: list[SearchResult]
    web_sources: list[RawSource]
    document_sources: list[RawSource]
    sources: list[RawSource]
    retrieved_context: list
    summary: str
    validation_notes: str
    citations: list
    confidence_score: float
    final_report: str
    warnings: list[str]


def _warnings(state: ResearchState) -> list[str]:
    return list(state.get("warnings", []))


def query_understanding_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    refined_questions = understand_query(
        query=state["query"],
        llm=get_local_llm(),
        warnings=warnings,
    )
    return {"refined_questions": refined_questions, "warnings": warnings}


def research_planning_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    research_plan = create_research_plan(
        query=state["query"],
        refined_questions=state.get("refined_questions", [state["query"]]),
        llm=get_local_llm(),
        warnings=warnings,
    )
    return {"research_plan": research_plan, "warnings": warnings}


def web_search_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    settings = get_settings()
    questions = state.get("refined_questions") or [state["query"]]
    search_results: list[SearchResult] = []
    web_sources: list[RawSource] = []

    try:
        search_results = search_web(questions=questions, settings=settings)
        for result in search_results:
            web_sources.append(search_result_to_source(result, settings=settings))
    except WebSearchError as exc:
        warnings.append(str(exc))

    if not web_sources:
        warnings.append("Web search returned no usable sources.")

    return {
        "search_results": search_results,
        "web_sources": web_sources,
        "warnings": warnings,
    }


def document_ingestion_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    document_sources = ingest_documents(
        uploads=state.get("uploaded_files", []),
        warnings=warnings,
    )
    sources = [*state.get("web_sources", []), *document_sources]
    return {
        "document_sources": document_sources,
        "sources": sources,
        "warnings": warnings,
    }


def retrieval_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    contexts = retrieve_context(
        sources=state.get("sources", []),
        query=state["query"],
        run_id=state["run_id"],
        warnings=warnings,
    )
    return {"retrieved_context": contexts, "warnings": warnings}


def summarization_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    summary = summarize_context(
        query=state["query"],
        contexts=state.get("retrieved_context", []),
        llm=get_local_llm(),
        warnings=warnings,
    )
    return {"summary": summary, "warnings": warnings}


def citation_validation_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    citations, confidence_score, validation_notes = validate_citations(
        contexts=state.get("retrieved_context", []),
        warnings=warnings,
    )
    return {
        "citations": citations,
        "confidence_score": confidence_score,
        "validation_notes": validation_notes,
        "warnings": warnings,
    }


def final_report_node(state: ResearchState) -> ResearchState:
    warnings = _warnings(state)
    report = generate_final_report(
        query=state["query"],
        research_plan=state.get("research_plan", []),
        summary=state.get("summary", ""),
        contexts=state.get("retrieved_context", []),
        citations=state.get("citations", []),
        confidence_score=state.get("confidence_score", 0.0),
        validation_notes=state.get("validation_notes", ""),
        llm=get_local_llm(),
        warnings=warnings,
    )
    return {"final_report": report, "warnings": warnings}


def build_research_graph():
    """Compile the LangGraph workflow."""

    if StateGraph is None or END is None:
        raise RuntimeError("langgraph is not installed. Run `pip install -r requirements.txt`.")

    graph = StateGraph(ResearchState)
    graph.add_node("query_understanding", query_understanding_node)
    graph.add_node("research_planning", research_planning_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("document_ingestion", document_ingestion_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("citation_validation", citation_validation_node)
    graph.add_node("final_report", final_report_node)

    graph.set_entry_point("query_understanding")
    graph.add_edge("query_understanding", "research_planning")
    graph.add_edge("research_planning", "web_search")
    graph.add_edge("web_search", "document_ingestion")
    graph.add_edge("document_ingestion", "retrieval")
    graph.add_edge("retrieval", "summarization")
    graph.add_edge("summarization", "citation_validation")
    graph.add_edge("citation_validation", "final_report")
    graph.add_edge("final_report", END)
    return graph.compile()


def run_research(
    query: str,
    uploaded_files: list[UploadedFilePayload] | None = None,
) -> ResearchResponse:
    """Execute the complete research workflow and return a response model."""

    query = query.strip()
    if len(query) < 3:
        raise ValueError("Query must be at least 3 characters long.")

    initial_state: ResearchState = {
        "run_id": uuid4().hex,
        "query": query,
        "uploaded_files": uploaded_files or [],
        "warnings": [],
    }
    graph = build_research_graph()
    final_state = graph.invoke(initial_state)

    response = ResearchResponse(
        query=query,
        refined_questions=final_state.get("refined_questions", []),
        research_plan=final_state.get("research_plan", []),
        retrieved_context=final_state.get("retrieved_context", []),
        final_report=final_state.get("final_report", ""),
        citations=final_state.get("citations", []),
        confidence_score=final_state.get("confidence_score", 0.0),
        warnings=final_state.get("warnings", []),
    )
    write_report(response)
    return response

