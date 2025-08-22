# Astrobase Marimo RAG System

A local PDF-based Retrieval-Augmented Generation (RAG) system built with Marimo notebooks and open-source LLMs.

## Features

- PDF upload and text extraction
- Semantic search using vector embeddings
- Local LLM integration for question answering
- Interactive Marimo interface
- Source citations in responses

## Setup

1. Install dependencies:
```bash
uv sync
```

2. Download and set up Ollama (for local LLM):
```bash
# macOS
brew install ollama

# Start Ollama service
ollama serve

# Pull a model (e.g., Llama 3)
ollama pull llama3
```

3. Run the Marimo notebook:
```bash
uv run marimo run notebooks/rag_system.py
```

## Project Structure

```
astrobase_marimo_rag/
├── notebooks/
│   └── rag_system.py      # Main Marimo notebook
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py   # PDF text extraction
│   ├── embeddings.py      # Text embedding generation
│   ├── vector_store.py    # Vector database interface
│   ├── llm_interface.py   # LLM interaction
│   └── config.py          # Configuration settings
├── data/
│   ├── pdfs/             # Uploaded PDF storage
│   └── vectordb/         # Persistent vector storage
├── requirements.txt
└── README.md
```

## Usage

1. Open the Marimo notebook in your browser
2. Upload a PDF using the file upload widget
3. Wait for processing (embedding generation)
4. Ask questions about the document
5. View answers with source citations

## Configuration

Edit `src/config.py` to modify:
- Chunk size and overlap
- Embedding model selection
- LLM model and parameters
- Vector database settings
