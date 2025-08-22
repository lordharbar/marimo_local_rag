import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo as mo
    import sys
    from pathlib import Path
    
    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    return mo, sys, Path


@app.cell
def __(mo):
    mo.md(
        r"""
        # 📚 Astrobase RAG System
        
        Upload PDFs and ask questions about their content using local LLMs.
        """
    )
    return


@app.cell
def __():
    # Import all modules
    from src.pdf_processor import PDFProcessor
    from src.embeddings import EmbeddingGenerator
    from src.vector_store import VectorStore, parse_search_results
    from src.llm_interface import LLMInterface, ConversationManager
    from src.config import PDF_DIR
    
    # Initialize components
    pdf_processor = PDFProcessor()
    embedding_generator = EmbeddingGenerator()
    vector_store = VectorStore()
    llm_interface = LLMInterface()
    conversation_manager = ConversationManager()
    
    return (
        ConversationManager,
        EmbeddingGenerator,
        LLMInterface,
        PDFProcessor,
        VectorStore,
        parse_search_results,
        PDF_DIR,
        pdf_processor,
        embedding_generator,
        vector_store,
        llm_interface,
        conversation_manager,
    )


@app.cell
def __(mo):
    mo.md("## 📄 Upload Document")
    return


@app.cell
def __(mo):
    # File upload widget
    file_upload = mo.ui.file(
        label="Upload PDF",
        filetypes=[".pdf"],
        multiple=False
    )
    file_upload
    return file_upload,


@app.cell
def __(file_upload, mo):
    # Display upload status
    if file_upload.value:
        uploaded_file = file_upload.value[0]
        mo.md(f"✅ Uploaded: **{uploaded_file.name}**")
    else:
        uploaded_file = None
        mo.md("*No file uploaded yet*")
    return uploaded_file,


@app.cell
def __(mo, uploaded_file):
    # Create process button
    process_button = mo.ui.button(
        label="Process PDF",
        disabled=not uploaded_file
    )
    mo.md(f"### Process Document\n\n{process_button}")
    return process_button,


@app.cell
def __(
    PDF_DIR,
    embedding_generator,
    mo,
    pdf_processor,
    process_button,
    uploaded_file,
    vector_store,
):
    # Process PDF when button is clicked
    if process_button.value and uploaded_file:
        mo.output.clear()
        mo.output.append(mo.md("🔄 Processing PDF..."))
        
        try:
            # Save uploaded file
            pdf_path = PDF_DIR / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.contents)
            
            # Extract and chunk text
            chunks = pdf_processor.process_pdf(pdf_path)
            
            # Generate embeddings
            chunks_with_embeddings = embedding_generator.embed_chunks(chunks)
            
            # Store in vector database
            vector_store.add_chunks(chunks_with_embeddings)
            
            mo.output.clear()
            mo.output.append(mo.md(f"✅ Successfully processed {len(chunks)} chunks from {uploaded_file.name}"))
        except Exception as e:
            import traceback
            traceback.print_exc()
            mo.output.clear()
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    return


@app.cell
def __(mo, vector_store):
    mo.md("## 📊 Document Statistics")
    
    try:
        sources = vector_store.get_all_sources()
        total_chunks = vector_store.count
        
        if sources:
            doc_list = "\n".join(f"- {source}" for source in sorted(sources))
            mo.md(f"""
- **Documents indexed:** {len(sources)}
- **Total chunks:** {total_chunks}

**Available documents:**
{doc_list}
""")
        else:
            mo.md("*No documents indexed yet*")
    except:
        mo.md("*No documents indexed yet*")
    return


@app.cell
def __(mo):
    mo.md("## 💬 Ask Questions")
    return


@app.cell
def __(mo):
    # Query input
    query_input = mo.ui.text_area(
        label="Ask a question about your documents:",
        placeholder="What is the main topic of the document?",
        rows=3
    )
    query_input
    return query_input,


@app.cell
def __(mo, query_input):
    # Search button
    search_button = mo.ui.button(
        label="Search & Answer",
        disabled=not query_input.value
    )
    search_button
    return search_button,


@app.cell
def __(
    conversation_manager,
    embedding_generator,
    llm_interface,
    mo,
    parse_search_results,
    query_input,
    search_button,
    vector_store,
):
    # Perform search when button is clicked
    if search_button.value and query_input.value:
        mo.output.clear()
        mo.output.append(mo.md("🔍 Searching and generating answer..."))
        
        try:
            # Generate query embedding
            query_embedding = embedding_generator.embed_query(query_input.value)
            
            # Search vector store
            search_results_raw = vector_store.search(query_embedding)
            search_results = parse_search_results(search_results_raw)
            
            if not search_results:
                mo.output.clear()
                mo.output.append(mo.md("❌ No relevant documents found. Please process a PDF first."))
            else:
                # Generate response
                answer = llm_interface.generate_response(
                    query_input.value,
                    search_results
                )
                
                # Add to conversation history
                conversation_manager.add_exchange(query_input.value, answer)
                
                # Format results
                sources_text = "\n\n".join([
                    f"**Source {i+1}:** {result.source_file} (Pages: {', '.join(map(str, result.page_numbers))})"
                    for i, result in enumerate(search_results)
                ])
                
                mo.output.clear()
                mo.output.append(mo.md(f"""## 🎯 Answer

{answer}

---

### 📍 Sources

{sources_text}"""))
        except Exception as e:
            import traceback
            traceback.print_exc()
            mo.output.clear()
            mo.output.append(mo.md(f"❌ Error: {str(e)}"))
    return


@app.cell
def __(conversation_manager, mo):
    # Display conversation history
    history = conversation_manager.get_formatted_history()
    if history:
        mo.md(f"""## 📜 Conversation History

{history}""")
    return


@app.cell
def __(mo):
    mo.md("""
---

### ⚙️ System Requirements

- Ollama must be running (`ollama serve`)
- Required model must be pulled (`ollama pull llama3`)
- Python packages must be installed (`pip install -r requirements.txt`)
""")
    return


if __name__ == "__main__":
    app.run()
