"""Streamlit UI for the Agentic RAG Research Assistant."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    page_icon="🔎",
    layout="wide",
)
st.title("🔎 Agentic RAG Research Assistant")
st.caption(
    "Local-first research with agent planning, web/document retrieval, citations, "
    "and transparent RAG quality metrics."
)

with st.sidebar:
    st.subheader("Runtime")
    st.code(API_URL, language="text")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        st.metric("Backend", health.get("status", "unknown"))
        st.write("Primary model:", health.get("primary_model", "unknown"))
        st.write("Embedding model:", health.get("embedding_model", "unknown"))
    except Exception:
        st.warning("Backend is not reachable yet.")

query = st.text_area(
    "Research question",
    placeholder=(
        "Compare agentic RAG with standard RAG for enterprise knowledge systems."
    ),
    height=120,
)
files = st.file_uploader(
    "Optional evidence documents",
    type=["pdf", "txt", "md", "markdown"],
    accept_multiple_files=True,
)

if st.button("Run research", type="primary", use_container_width=True):
    if len(query.strip()) < 3:
        st.error("Enter a research question with at least 3 characters.")
        st.stop()

    multipart = [("files", (file.name, file.getvalue(), file.type)) for file in files]
    try:
        with st.spinner("Running the research graph…"):
            response = requests.post(
                f"{API_URL}/research",
                data={"query": query},
                files=multipart,
                timeout=600,
            )
            response.raise_for_status()
            result = response.json()
    except Exception as exc:
        st.error(f"Research request failed: {exc}")
        st.stop()

    evaluation = result.get("evaluation", {})
    cols = st.columns(6)
    cols[0].metric("Overall", f"{evaluation.get('overall_score', 0):.2f}")
    cols[1].metric("Citation validity", f"{evaluation.get('citation_validity', 0):.2f}")
    cols[2].metric("Citation coverage", f"{evaluation.get('citation_coverage', 0):.2f}")
    cols[3].metric("Source diversity", f"{evaluation.get('source_diversity', 0):.2f}")
    cols[4].metric("Retrieval", f"{evaluation.get('retrieval_quality', 0):.2f}")
    cols[5].metric("Grounding proxy", f"{evaluation.get('groundedness_proxy', 0):.2f}")

    st.subheader("Research report")
    st.markdown(result.get("final_report", ""))
    st.download_button(
        "Download Markdown report",
        result.get("final_report", ""),
        file_name=f"research-{result.get('run_id', 'report')}.md",
        mime="text/markdown",
    )

    with st.expander("Research plan"):
        for item in result.get("research_plan", []):
            st.write(f"- {item}")

    with st.expander("Evidence and citations"):
        for citation in result.get("citations", []):
            label = citation.get("citation_id", "Source")
            st.markdown(f"**[{label}] {citation.get('title', 'Untitled')}**")
            if citation.get("url"):
                st.write(citation["url"])
            st.caption(citation.get("snippet", ""))

    with st.expander("Diagnostics"):
        st.write("Model used:", result.get("model_used"))
        st.write("Retrieval mode:", result.get("retrieval_mode"))
        st.write("Revision passes:", result.get("revision_count"))
        if evaluation.get("notes"):
            st.write("Evaluation notes:")
            for note in evaluation["notes"]:
                st.write(f"- {note}")
        if result.get("warnings"):
            st.write("Runtime warnings:")
            for warning in result["warnings"]:
                st.write(f"- {warning}")
