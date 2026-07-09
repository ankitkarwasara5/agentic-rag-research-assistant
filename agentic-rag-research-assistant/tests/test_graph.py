import pytest


def test_graph_compiles_when_langgraph_is_installed() -> None:
    pytest.importorskip("langgraph")

    from app.graph import build_research_graph

    graph = build_research_graph()

    assert graph is not None

