"""Streamlit frontend for the local research assistant."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.schemas import ResearchResponse
from app.services.report_writer import render_markdown_report


settings = get_settings()
DEFAULT_API_URL = os.getenv(
    "RAG_API_URL",
    f"http://{settings.api_host}:{settings.api_port}",
)


st.set_page_config(
    page_title="Agentic RAG Research Assistant",
    layout="wide",
)

st.title("Agentic RAG Research Assistant")

with st.sidebar:
    api_url = st.text_input("Backend URL", value=DEFAULT_API_URL)

query = st.text_area(
    "Research query",
    value="Compare RAG and Agentic RAG",
    height=120,
)
uploaded_files = st.file_uploader(
    "Documents",
    type=["pdf", "txt", "md"],
    accept_multiple_files=True,
)

run_button = st.button("Run Research", type="primary", use_container_width=True)

if run_button:
    if not query.strip():
        st.error("Enter a research query.")
        st.stop()

    files_payload = [
        ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
        for file in uploaded_files
    ]

    with st.spinner("Running research workflow..."):
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/research",
                data={"query": query},
                files=files_payload or None,
                timeout=900,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            st.error(f"Research request failed: {exc}")
            st.stop()

    result = ResearchResponse.model_validate(response.json())
    report_md = render_markdown_report(result)

    st.session_state["last_result"] = result
    st.session_state["last_report_md"] = report_md

if "last_result" in st.session_state:
    result: ResearchResponse = st.session_state["last_result"]
    report_md: str = st.session_state["last_report_md"]

    confidence_pct = int(result.confidence_score * 100)
    st.metric("Confidence", f"{confidence_pct}%")
    st.progress(result.confidence_score)

    tabs = st.tabs(["Plan", "Sources", "Report", "Citations", "Warnings"])

    with tabs[0]:
        st.subheader("Research Plan")
        for item in result.research_plan:
            st.markdown(f"- {item}")
        st.subheader("Refined Questions")
        for item in result.refined_questions:
            st.markdown(f"- {item}")

    with tabs[1]:
        st.subheader("Retrieved Sources")
        if not result.retrieved_context:
            st.info("No retrieved context.")
        for context in result.retrieved_context:
            with st.expander(f"[{context.citation_id}] {context.title}"):
                st.write(context.text)
                st.caption(context.url or context.file_name or context.source_type)
                st.caption(f"Score: {context.score:.2f} | Chunk: {context.chunk_id}")

    with tabs[2]:
        st.subheader("Final Report")
        st.markdown(result.final_report)
        st.download_button(
            label="Download Markdown",
            data=report_md,
            file_name="research_report.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with tabs[3]:
        st.subheader("Citations")
        if not result.citations:
            st.info("No citations available.")
        for citation in result.citations:
            locator = citation.url or citation.file_name or "local source"
            st.markdown(f"- **[{citation.citation_id}] {citation.title}**")
            st.caption(locator)
            st.write(citation.snippet)

    with tabs[4]:
        st.subheader("Warnings")
        if result.warnings:
            for warning in result.warnings:
                st.warning(warning)
        else:
            st.success("No warnings.")
