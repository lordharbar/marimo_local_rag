"""Helper functions for the RAG system UI."""

from pathlib import Path
from typing import TypeAlias
from dataclasses import dataclass, field
import traceback

from src.pdf_processor import PDFProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore, parse_search_results, SearchResult
from src.llm_interface import LLMInterface, ConversationManager
from src.config import PDF_DIR

# Type aliases
ProcessResult: TypeAlias = tuple[bool, str, int | None]
AnswerResult: TypeAlias = tuple[bool, str, list[dict[str, str | list[int]]] | None]
ClearResult: TypeAlias = tuple[bool, str]


@dataclass
class Statistics:
    """Statistics about indexed documents."""
    success: bool
    num_documents: int
    total_chunks: int
    documents: list[str] = field(default_factory=list)


@dataclass
class RAGSystem:
    """Encapsulates the RAG system functionality."""
    
    pdf_processor: PDFProcessor = field(default_factory=PDFProcessor)
    embedding_generator: EmbeddingGenerator = field(default_factory=EmbeddingGenerator)
    vector_store: VectorStore = field(default_factory=VectorStore)
    llm_interface: LLMInterface = field(default_factory=LLMInterface)
    conversation_manager: ConversationManager = field(default_factory=ConversationManager)
    
    def process_pdf(self, file_name: str, file_contents: bytes) -> ProcessResult:
        """
        Process a PDF file and add it to the vector store.
        
        Args:
            file_name: Name of the PDF file
            file_contents: Binary contents of the PDF
            
        Returns:
            Tuple of (success, message, chunk_count)
        """
        try:
            # Save the file using Path methods
            pdf_path = PDF_DIR / file_name
            pdf_path.write_bytes(file_contents)
            print(f"Saved PDF to {pdf_path}")
            
            # Extract and chunk text
            print("Extracting text from PDF...")
            chunks = self.pdf_processor.process_pdf(pdf_path)
            print(f"Created {len(chunks)} chunks")
            
            if not chunks:
                return False, "No text could be extracted from the PDF", None
            
            # Generate embeddings
            print("Generating embeddings...")
            chunks_with_embeddings = self.embedding_generator.embed_chunks(chunks)
            print(f"Generated embeddings for {len(chunks_with_embeddings)} chunks")
            
            # Store in vector database
            print("Storing in vector database...")
            self.vector_store.add_chunks(chunks_with_embeddings)
            print(f"Successfully stored chunks in vector database")
            
            return True, f"Successfully processed {len(chunks)} chunks from {file_name}", len(chunks)
            
        except Exception as e:
            error_msg = f"Error processing PDF: {e!s}"
            print(error_msg)
            print(traceback.format_exc())
            return False, error_msg, None
    
    def answer_question(self, question: str) -> AnswerResult:
        """
        Answer a question using the RAG system.
        
        Args:
            question: The user's question
            
        Returns:
            Tuple of (success, answer/error_message, sources)
        """
        try:
            question = question.strip()
            if not question:
                return False, "Please enter a question", None
            
            print(f"Processing question: {question}")
            
            # Check if we have any documents
            if self.vector_store.count == 0:
                return False, "No documents have been indexed yet. Please upload and process a PDF first.", None
            
            # Generate query embedding
            print("Generating query embedding...")
            query_embedding = self.embedding_generator.embed_query(question)
            
            # Search vector store
            print("Searching vector store...")
            search_results_raw = self.vector_store.search(query_embedding)
            search_results = parse_search_results(search_results_raw)
            
            if not search_results:
                return False, "No relevant information found in the indexed documents.", None
            
            print(f"Found {len(search_results)} relevant chunks")
            
            # Generate response
            print("Generating answer with LLM...")
            answer = self.llm_interface.generate_response(question, search_results)
            
            # Add to conversation history
            self.conversation_manager.add_exchange(question, answer)
            
            # Extract source information
            sources = [
                {
                    "file": result.source_file,
                    "pages": list(result.page_numbers)
                }
                for result in search_results
            ]
            
            return True, answer, sources
            
        except Exception as e:
            error_msg = f"Error generating answer: {e!s}"
            print(error_msg)
            print(traceback.format_exc())
            return False, error_msg, None
    
    def get_statistics(self) -> Statistics:
        """
        Get statistics about the indexed documents.
        
        Returns:
            Statistics dataclass with document information
        """
        try:
            sources = self.vector_store.get_all_sources()
            total_chunks = self.vector_store.count
            
            return Statistics(
                success=True,
                num_documents=len(sources),
                total_chunks=total_chunks,
                documents=sorted(sources) if sources else []
            )
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return Statistics(
                success=False,
                num_documents=0,
                total_chunks=0,
                documents=[]
            )
    
    def clear_database(self) -> ClearResult:
        """
        Clear the vector database.
        
        Returns:
            Tuple of (success, message)
        """
        try:
            # Reset the vector store
            self.vector_store = VectorStore()
            # Clear conversation history
            self.conversation_manager.clear()
            return True, "Database cleared successfully"
        except Exception as e:
            return False, f"Error clearing database: {e!s}"
    
    def get_conversation_history(self) -> str:
        """
        Get formatted conversation history.
        
        Returns:
            Formatted conversation history string
        """
        return self.conversation_manager.get_formatted_history()


# Create a singleton instance
rag_system = RAGSystem()
