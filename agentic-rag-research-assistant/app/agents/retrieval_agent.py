"""Retrieval Agent for vector search with lexical fallback."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.schemas import RawSource, RetrievedContext
from app.services.vector_store import VectorStoreError, VectorStoreService
from app.utils.text_utils import split_text, term_overlap_score


def _assign_citation_ids(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    citation_map: dict[str, str] = {}
    next_idx = 1
    assigned: list[RetrievedContext] = []

    for context in contexts:
        key = context.source_id or context.url or context.file_name or context.title
        if key not in citation_map:
            citation_map[key] = f"Source {next_idx}"
            next_idx += 1
        assigned.append(context.model_copy(update={"citation_id": citation_map[key]}))
    return assigned


def lexical_retrieve(
    sources: list[RawSource],
    query: str,
    settings: Settings | None = None,
) -> list[RetrievedContext]:
    """Retrieve chunks with a deterministic keyword-overlap scorer."""

    settings = settings or get_settings()
    candidates: list[RetrievedContext] = []

    for source in sources:
        chunks = split_text(source.text, settings.chunk_size, settings.chunk_overlap)
        for idx, chunk in enumerate(chunks):
            score = term_overlap_score(query, chunk)
            candidates.append(
                RetrievedContext(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    title=source.title,
                    text=chunk,
                    score=score,
                    citation_id="",
                    chunk_id=f"{source.source_id}:{idx}",
                    url=source.url,
                    file_name=source.file_name,
                )
            )

    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    top = ranked[: settings.retrieval_top_k]
    return _assign_citation_ids(top)


def retrieve_context(
    sources: list[RawSource],
    query: str,
    run_id: str,
    warnings: list[str],
    settings: Settings | None = None,
) -> list[RetrievedContext]:
    """Run vector retrieval and fall back to lexical retrieval if needed."""

    settings = settings or get_settings()
    if not sources:
        warnings.append("No web or document sources were available for retrieval.")
        return []

    try:
        contexts = VectorStoreService(settings).index_and_retrieve(
            sources=sources,
            query=query,
            run_id=run_id,
            top_k=settings.retrieval_top_k,
        )
        return _assign_citation_ids(contexts)
    except VectorStoreError as exc:
        warnings.append(f"{exc} Used lexical retrieval fallback.")
        return lexical_retrieve(sources=sources, query=query, settings=settings)

