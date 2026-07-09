"""ChromaDB vector storage and retrieval service."""

from __future__ import annotations

from uuid import uuid4

from app.config import Settings, get_settings
from app.schemas import RawSource, RetrievedContext
from app.utils.text_utils import split_text

try:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # pragma: no cover - exercised only without dependencies
    Chroma = None  # type: ignore[assignment]
    Document = None  # type: ignore[assignment]
    OllamaEmbeddings = None  # type: ignore[assignment]
    RecursiveCharacterTextSplitter = None  # type: ignore[assignment]


class VectorStoreError(RuntimeError):
    """Raised when vector indexing or retrieval fails."""


class VectorStoreService:
    """Store source chunks in ChromaDB and retrieve relevant context."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def _require_dependencies(self) -> None:
        if Chroma is None or Document is None or OllamaEmbeddings is None:
            raise VectorStoreError(
                "LangChain Chroma/Ollama packages are not installed. "
                "Run `pip install -r requirements.txt`."
            )

    def _embedding_function(self):
        self._require_dependencies()
        return OllamaEmbeddings(
            model=self.settings.embedding_model,
            base_url=self.settings.ollama_base_url,
        )

    def _vector_store(self):
        self._require_dependencies()
        return Chroma(
            collection_name=self.settings.collection_name,
            embedding_function=self._embedding_function(),
            persist_directory=str(self.settings.chroma_dir),
        )

    def chunk_sources(self, sources: list[RawSource], run_id: str):
        """Convert raw sources into LangChain documents with metadata."""

        self._require_dependencies()
        documents = []

        for source in sources:
            if RecursiveCharacterTextSplitter is not None:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                chunks = splitter.split_text(source.text)
            else:
                chunks = split_text(
                    source.text,
                    self.settings.chunk_size,
                    self.settings.chunk_overlap,
                )

            for idx, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "run_id": run_id,
                            "source_id": source.source_id,
                            "source_type": source.source_type,
                            "title": source.title,
                            "url": source.url or "",
                            "file_name": source.file_name or "",
                            "chunk_id": f"{source.source_id}:{idx}",
                        },
                    )
                )
        return documents

    def index_and_retrieve(
        self,
        sources: list[RawSource],
        query: str,
        run_id: str,
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        """Index source chunks and retrieve the most relevant chunks."""

        documents = self.chunk_sources(sources, run_id=run_id)
        if not documents:
            return []

        try:
            vector_store = self._vector_store()
            ids = [f"{run_id}:{uuid4().hex}" for _ in documents]
            vector_store.add_documents(documents, ids=ids)
            results = vector_store.similarity_search_with_relevance_scores(
                query,
                k=top_k or self.settings.retrieval_top_k,
                filter={"run_id": run_id},
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Chroma/Ollama retrieval failed: {exc}") from exc

        contexts: list[RetrievedContext] = []
        for doc, score in results:
            metadata = doc.metadata
            contexts.append(
                RetrievedContext(
                    source_id=str(metadata.get("source_id", "")),
                    source_type=metadata.get("source_type", "document"),
                    title=str(metadata.get("title", "Untitled source")),
                    text=doc.page_content,
                    score=float(score or 0.0),
                    citation_id="",
                    chunk_id=str(metadata.get("chunk_id", "")),
                    url=str(metadata.get("url") or "") or None,
                    file_name=str(metadata.get("file_name") or "") or None,
                )
            )
        return contexts

