"""Vector retrieval with Chroma and deterministic lexical fallback."""

from __future__ import annotations

from app.config import Settings, get_settings
from app.schemas import RawSource, RetrievedContext
from app.services.ollama import OllamaService
from app.utils.text import split_text, term_overlap_score


class VectorStoreError(RuntimeError):
    """Raised when semantic retrieval cannot complete."""


def _assign_citations(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
    labels: dict[str, str] = {}
    next_number = 1
    output: list[RetrievedContext] = []
    for context in contexts:
        key = context.source_id
        if key not in labels:
            labels[key] = f"Source {next_number}"
            next_number += 1
        output.append(context.model_copy(update={"citation_id": labels[key]}))
    return output


def lexical_retrieve(
    sources: list[RawSource],
    query: str,
    settings: Settings | None = None,
) -> list[RetrievedContext]:
    """Retrieve evidence using deterministic keyword overlap."""

    settings = settings or get_settings()
    candidates: list[RetrievedContext] = []
    for source in sources:
        for index, chunk in enumerate(
            split_text(source.text, settings.chunk_size, settings.chunk_overlap)
        ):
            candidates.append(
                RetrievedContext(
                    source_id=source.source_id,
                    source_type=source.source_type,
                    title=source.title,
                    text=chunk,
                    score=term_overlap_score(query, chunk),
                    citation_id="",
                    chunk_id=f"{source.source_id}:{index}",
                    url=source.url,
                    file_name=source.file_name,
                )
            )
    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
    return _assign_citations(ranked[: settings.retrieval_top_k])


class VectorStoreService:
    """Index source chunks in Chroma using Ollama embeddings."""

    def __init__(
        self,
        settings: Settings | None = None,
        ollama: OllamaService | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.ollama = ollama or OllamaService(self.settings)

    def retrieve(
        self,
        sources: list[RawSource],
        query: str,
        run_id: str,
    ) -> list[RetrievedContext]:
        """Index this run's sources and return semantically similar chunks."""

        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("chromadb is not installed") from exc

        chunks: list[tuple[RawSource, int, str]] = []
        for source in sources:
            parts = split_text(
                source.text,
                self.settings.chunk_size,
                self.settings.chunk_overlap,
            )
            chunks.extend((source, index, chunk) for index, chunk in enumerate(parts))
        if not chunks:
            return []

        documents = [item[2] for item in chunks]
        try:
            embeddings = self.ollama.embed(documents)
            query_embedding = self.ollama.embed([query])[0]
            client = chromadb.PersistentClient(path=str(self.settings.chroma_dir))
            collection = client.get_or_create_collection(
                name=self.settings.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            ids = [
                f"{run_id}:{source.source_id}:{index}" for source, index, _ in chunks
            ]
            metadatas = [
                {
                    "run_id": run_id,
                    "source_id": source.source_id,
                    "source_type": source.source_type,
                    "title": source.title,
                    "url": source.url or "",
                    "file_name": source.file_name or "",
                    "chunk_id": f"{source.source_id}:{index}",
                }
                for source, index, _ in chunks
            ]
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=min(self.settings.retrieval_top_k, len(documents)),
                where={"run_id": run_id},
                include=["metadatas", "documents", "distances"],
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Semantic retrieval failed: {exc}") from exc

        contexts: list[RetrievedContext] = []
        result_docs = (result.get("documents") or [[]])[0]
        result_meta = (result.get("metadatas") or [[]])[0]
        result_distances = (result.get("distances") or [[]])[0]
        for document, metadata, distance in zip(
            result_docs,
            result_meta,
            result_distances,
            strict=False,
        ):
            similarity = max(0.0, min(1.0, 1.0 - float(distance or 0.0)))
            contexts.append(
                RetrievedContext(
                    source_id=str(metadata.get("source_id", "")),
                    source_type=metadata.get("source_type", "document"),
                    title=str(metadata.get("title", "Untitled source")),
                    text=str(document),
                    score=similarity,
                    citation_id="",
                    chunk_id=str(metadata.get("chunk_id", "")),
                    url=str(metadata.get("url") or "") or None,
                    file_name=str(metadata.get("file_name") or "") or None,
                )
            )
        return _assign_citations(contexts)
