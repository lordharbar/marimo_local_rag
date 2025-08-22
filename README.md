# Astrobase Marimo RAG System

A local PDF-based Retrieval-Augmented Generation (RAG) system built with Marimo notebooks and open-source LLMs.

## Features

- PDF upload and text extraction
- Semantic search using vector embeddings
- Local LLM integration for question answering
- Interactive Marimo interface
- Source citations in responses
- Persistent vector storage using ChromaDB

## Prerequisites

- Python 3.11+
- UV package manager
- Ollama for local LLM

## Setup

### 1. Install UV (if not already installed)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### 2. Clone and set up the project
```bash
cd wherever_you_want/astrobase_marimo_rag

# Install dependencies with UV
uv sync
```

### 3. Install and configure Ollama
```bash
# macOS
brew install ollama

# Start Ollama service (keep this running in a terminal)
ollama serve

# In another terminal, pull the Llama 3 model
ollama pull llama3
```

### 4. Test the setup
```bash
# Test ChromaDB
uv run python test_chromadb.py

# Test Ollama connection
uv run python test_ollama.py

# Run diagnostics if needed
uv run python diagnose_ollama.py
```

### 5. Run the RAG system
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
│   ├── config.py          # Configuration settings
│   ├── pdf_processor.py   # PDF text extraction & chunking
│   ├── embeddings.py      # Text embedding generation
│   ├── vector_store.py    # ChromaDB interface
│   └── llm_interface.py   # Ollama LLM interface
├── data/
│   ├── pdfs/              # Uploaded PDF storage
│   └── vectordb/          # Persistent vector storage
├── test_chromadb.py       # ChromaDB test script
├── test_ollama.py         # Ollama connection test
├── diagnose_ollama.py     # Ollama diagnostic tool
├── requirements.txt       # Python dependencies
├── pyproject.toml         # UV/Python project configuration
└── README.md              # This file
```

## Usage

1. Start Ollama service (if not running):
   ```bash
   ollama serve
   ```

2. Run the Marimo notebook:
   ```bash
   uv run marimo run notebooks/rag_system.py
   ```

3. In your browser:
   - Upload a PDF using the file upload widget
   - Click "Process PDF" to generate embeddings
   - Ask questions about the document
   - View answers with source citations

## Configuration

Edit `src/config.py` to modify:
- **Chunk settings**: Size (1000 chars) and overlap (200 chars)
- **Embedding model**: Default is `all-MiniLM-L6-v2`
- **LLM model**: Default is `llama3`
- **Generation parameters**: Temperature (0.7), max tokens (2000)
- **Search settings**: Top-K results (5)

## Common Commands

```bash
# Install/update dependencies
uv sync

# Add a new package
uv add package_name

# Run the main application
uv run marimo run notebooks/rag_system.py

# Test Ollama connection
uv run python test_ollama.py

# Test ChromaDB
uv run python test_chromadb.py

# Check Ollama models
ollama list

# Pull a different model
ollama pull mistral

# Clear vector database (if needed)
rm -rf data/vectordb/*

# Run any Python script in the UV environment
uv run python script_name.py
```

## Troubleshooting

### Ollama not connecting
1. Ensure Ollama is running: `ollama serve`
2. Check if model is downloaded: `ollama list`
3. Run diagnostics: `uv run python diagnose_ollama.py`

### ChromaDB issues
1. Clear the database: `rm -rf data/vectordb/*`
2. Test ChromaDB: `uv run python test_chromadb.py`

### Package issues
1. Update all packages: `uv sync --refresh`
2. Clear UV cache: `uv cache clean`

### Marimo not loading
1. Check UV environment: `uv pip list`
2. Reinstall Marimo: `uv add marimo --force-reinstall`

## Performance Tips

- For large PDFs, increase chunk size in `src/config.py`
- Use GPU acceleration if available (configure in Ollama)
- Adjust `TOP_K_RESULTS` for more/fewer context chunks
- Lower temperature for more factual responses

## Next Steps

Potential enhancements:
- Add support for multiple file formats (DOCX, TXT, MD)
- Implement conversation memory
- Add OCR for scanned PDFs
- Create export functionality for Q&A sessions
- Add different embedding models
- Implement re-ranking for better search results

## Notes

- **Memory usage**: Large PDFs require significant RAM
- **Model size**: Llama3 is ~4.7GB
- **Processing time**: Initial embedding generation can take time
- **Persistence**: Vector embeddings are saved in `data/vectordb/`

## License

MIT

## Author

Created for SHB
Location: /Users/SHB/Documents/CHG/repo/astrobase_marimo_rag/
