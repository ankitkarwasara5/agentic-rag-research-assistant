from app.agents.retrieval_agent import lexical_retrieve
from app.config import Settings
from app.schemas import RawSource
from app.utils.text_utils import split_text


def test_split_text_uses_overlap() -> None:
    chunks = split_text("abcdefghijklmnopqrstuvwxyz", chunk_size=10, overlap=2)

    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("ij")


def test_lexical_retrieval_ranks_relevant_source(tmp_path) -> None:
    settings = Settings(
        chunk_size=120,
        chunk_overlap=10,
        retrieval_top_k=2,
        upload_dir=tmp_path / "uploads",
        chroma_dir=tmp_path / "chroma",
        reports_dir=tmp_path / "reports",
    )
    sources = [
        RawSource(
            source_id="src_1",
            source_type="document",
            title="RAG Notes",
            file_name="rag.md",
            text="Agentic RAG uses planning, retrieval, validation, and citations.",
        ),
        RawSource(
            source_id="src_2",
            source_type="document",
            title="Cooking Notes",
            file_name="cook.md",
            text="A recipe needs flour, water, and heat.",
        ),
    ]

    results = lexical_retrieve(sources, "Agentic RAG validation", settings=settings)

    assert results
    assert results[0].source_id == "src_1"
    assert results[0].citation_id == "Source 1"

