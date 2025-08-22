"""Embedding generation utilities."""

from typing import TypeAlias
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import EMBEDDING_MODEL
from .pdf_processor import TextChunk

# Type aliases
Embedding: TypeAlias = np.ndarray
EmbeddingsList: TypeAlias = list[Embedding]


@dataclass
class EmbeddingGenerator:
    """Handles text embedding generation."""
    
    model_name: str = EMBEDDING_MODEL
    _model: SentenceTransformer | None = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def embed_text(self, text: str | list[str]) -> Embedding | EmbeddingsList:
        """Generate embeddings for text."""
        if isinstance(text, str):
            return self.model.encode(text, convert_to_numpy=True)
        else:
            return self.model.encode(text, convert_to_numpy=True, show_progress_bar=True)
    
    def embed_chunks(self, chunks: list[TextChunk]) -> list[tuple[TextChunk, Embedding]]:
        """Generate embeddings for all chunks."""
        if not chunks:
            return []
            
        print(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batch
        embeddings = self.embed_text(texts)
        
        # Ensure embeddings is a list
        if isinstance(embeddings, np.ndarray) and len(embeddings.shape) == 1:
            embeddings = [embeddings]
        
        # Pair chunks with their embeddings
        return list(zip(chunks, embeddings))
    
    def embed_query(self, query: str) -> Embedding:
        """Generate embedding for a query."""
        return self.embed_text(query)
    
    @property
    def embedding_dimension(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model.get_sentence_embedding_dimension()
