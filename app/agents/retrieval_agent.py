"""Evidence retrieval agent."""

from app.config import Settings
from app.schemas import RawSource, RetrievedContext
from app.services.vector_store import (
    VectorStoreError,
    VectorStoreService,
    lexical_retrieve,
)


def retrieve_evidence(
    sources: list[RawSource],
    query: str,
    run_id: str,
    settings: Settings,
    warnings: list[str],
) -> tuple[list[RetrievedContext], str]:
    """Prefer semantic retrieval and fall back deterministically."""

    if not sources:
        warnings.append("No sources were available for retrieval.")
        return [], "none"
    try:
        contexts = VectorStoreService(settings).retrieve(sources, query, run_id)
        return contexts, "semantic"
    except VectorStoreError as exc:
        warnings.append(f"{exc} Using lexical retrieval fallback.")
        return lexical_retrieve(sources, query, settings), "lexical"
