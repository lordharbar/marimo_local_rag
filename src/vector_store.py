"""Vector database interface for ChromaDB."""

from typing import TypeAlias, Self
from dataclasses import dataclass, field
from pathlib import Path
import chromadb
import numpy as np

from .config import (
    VECTOR_DB_DIR,
    COLLECTION_NAME,
    TOP_K_RESULTS,
    VECTOR_DB_BATCH_SIZE,
    METADATA_SOURCE_KEY,
    METADATA_PAGES_KEY,
    METADATA_CHUNK_INDEX_KEY,
)
from .pdf_processor import TextChunk
from .embeddings import Embedding

# Type aliases
QueryResult: TypeAlias = dict[str, list[str] | list[dict] | list[float]]


@dataclass
class VectorStore:
    """Interface for ChromaDB vector storage."""

    persist_dir: Path = VECTOR_DB_DIR
    collection_name: str = COLLECTION_NAME
    top_k: int = TOP_K_RESULTS
    _client: chromadb.ClientAPI | None = None
    _collection: chromadb.Collection | None = None

    def __post_init__(self):
        """Initialize the ChromaDB client."""
        self.persist_dir.mkdir(parents=True, exist_ok=True)

    @property
    def client(self) -> chromadb.ClientAPI:
        """Lazy load ChromaDB client."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        return self._client

    @property
    def collection(self) -> chromadb.Collection:
        """Get or create the collection."""
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        return self._collection

    def add_chunks(self, chunks_with_embeddings: list[tuple[TextChunk, Embedding]]) -> None:
        """Add chunks and their embeddings to the vector store."""
        if not chunks_with_embeddings:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for chunk, embedding in chunks_with_embeddings:
            ids.append(chunk.chunk_id)
            embeddings.append(embedding.tolist())
            documents.append(chunk.content)

            metadata = {
                METADATA_SOURCE_KEY: chunk.source_file,
                METADATA_PAGES_KEY: ",".join(map(str, chunk.page_numbers)),
                METADATA_CHUNK_INDEX_KEY: chunk.metadata.get("chunk_index", 0),
            }
            metadatas.append(metadata)

        # Add to collection in batches
        for i in range(0, len(ids), VECTOR_DB_BATCH_SIZE):
            end = min(i + VECTOR_DB_BATCH_SIZE, len(ids))
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end]
            )

        print(f"Added {len(ids)} chunks to vector store")

    def search(self, query_embedding: Embedding, top_k: int | None = None) -> QueryResult:
        """Search for similar chunks."""
        if top_k is None:
            top_k = self.top_k

        if self.count == 0:
            return {"documents": [], "metadatas": [], "distances": []}

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.count),
            include=["documents", "metadatas", "distances"]
        )

        # --- BUG FIX ---
        # Robustly handle empty results, which can have different structures.
        if not results or not results.get("documents") or not results["documents"][0]:
            return {"documents": [], "metadatas": [], "distances": []}

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }

    def get_all_sources(self) -> set[str]:
        """Get all unique source files in the collection."""
        results = self.collection.get(include=["metadatas"])
        sources = set()

        if results and results.get("metadatas"):
            for metadata in results["metadatas"]:
                if metadata and METADATA_SOURCE_KEY in metadata:
                    sources.add(metadata[METADATA_SOURCE_KEY])

        return sources

    def delete_source(self, source_file: str) -> None:
        """Delete all chunks from a specific source file."""
        self.collection.delete(where={METADATA_SOURCE_KEY: source_file})
        print(f"Deleted all chunks from {source_file}")

    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(name=self.collection_name)
        self._collection = None
        print(f"Cleared collection: {self.collection_name}")

    @property
    def count(self) -> int:
        """Get the number of chunks in the collection."""
        return self.collection.count()


@dataclass
class SearchResult:
    """Represents a search result with metadata."""

    content: str
    source_file: str
    page_numbers: list[int]
    distance: float
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_query_result(cls, document: str, metadata: dict, distance: float) -> Self:
        """Create SearchResult from query result."""
        page_numbers_str = metadata.get(METADATA_PAGES_KEY, "")
        page_numbers = [int(p) for p in page_numbers_str.split(",") if p]

        return cls(
            content=document,
            source_file=metadata.get(METADATA_SOURCE_KEY, "Unknown"),
            page_numbers=page_numbers,
            distance=distance,
            metadata=metadata
        )


def parse_search_results(query_result: QueryResult) -> list[SearchResult]:
    """Parse query results into SearchResult objects."""
    results: list[SearchResult] = []

    for doc, meta, dist in zip(
        query_result.get("documents", []),
        query_result.get("metadatas", []),
        query_result.get("distances", [])
    ):
        result = SearchResult.from_query_result(doc, meta, dist)
        results.append(result)

    return results