from app.schemas import RawSource
from app.services.vector_store import lexical_retrieve


def test_lexical_retrieval_prefers_relevant_source(settings):
    sources = [
        RawSource(
            source_id="a",
            source_type="document",
            title="RAG notes",
            text=(
                "Retrieval augmented generation uses retrieved evidence and citations."
            ),
        ),
        RawSource(
            source_id="b",
            source_type="document",
            title="Cooking",
            text="Bake bread with flour and yeast in a warm oven.",
        ),
    ]
    contexts = lexical_retrieve(sources, "retrieval evidence RAG", settings)
    assert contexts
    assert contexts[0].source_id == "a"
    assert contexts[0].citation_id == "Source 1"
