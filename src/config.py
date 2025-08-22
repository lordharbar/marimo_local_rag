"""Configuration settings for the RAG system."""

from pathlib import Path
from typing import TypeAlias

# Type aliases
ModelName: TypeAlias = str
ChunkSize: TypeAlias = int

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
VECTOR_DB_DIR = DATA_DIR / "vectordb"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)

# PDF Processing
CHUNK_SIZE: ChunkSize = 1000  # characters
CHUNK_OVERLAP: ChunkSize = 200  # characters
MIN_CHUNK_SIZE: ChunkSize = 100  # minimum chunk size to keep

# Embedding settings
EMBEDDING_MODEL: ModelName = "all-MiniLM-L6-v2"  # Fast and lightweight
# Alternative: "all-mpnet-base-v2" for better quality

# Vector database settings
VECTOR_DB_NAME = "astrobase_rag"
COLLECTION_NAME = "documents"
TOP_K_RESULTS = 5  # Number of chunks to retrieve

# LLM settings
LLM_MODEL: ModelName = "llama3"  # Ollama model name
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 2000
LLM_CONTEXT_WINDOW = 4096

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"

# Prompt templates
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context. 
Always cite the specific parts of the context that support your answer.
If the context doesn't contain enough information to answer the question, say so clearly."""

QUESTION_PROMPT_TEMPLATE = """Context information is below:
---------------------
{context}
---------------------

Given the context information above, please answer the following question:
{question}

If the context contains relevant information, cite the specific parts that support your answer.
If the context doesn't contain the answer, please state that clearly.

Answer:"""
