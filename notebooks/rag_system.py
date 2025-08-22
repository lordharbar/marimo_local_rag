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
    mo.md("# 📚 Astrobase RAG System")
    return


@app.cell
def __():
    from src.pdf_processor import PDFProcessor
    from src.embeddings import EmbeddingGenerator
    from src.vector_store import VectorStore, parse_search_results
    from src.llm_interface import LLMInterface, ConversationManager
    from src.config import PDF_DIR
    
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
    mo.md("## Upload PDF")
    return


@app.cell
def __(mo):
    file_upload = mo.ui.file(
        label="Upload PDF",
        filetypes=[".pdf"],
        multiple=False
    )
    return file_upload,


@app.cell
def __(file_upload):
    file_upload
    return


@app.cell
def __(file_upload, mo):
    if file_upload.value:
        uploaded_file = file_upload.value[0]
        upload_status = mo.md(f"Uploaded: {uploaded_file.name}")
    else:
        uploaded_file = None
        upload_status = mo.md("No file uploaded")
    return uploaded_file, upload_status


@app.cell
def __(upload_status):
    upload_status
    return


@app.cell
def __(mo, uploaded_file):
    process_button = mo.ui.button(
        label="Process PDF",
        disabled=not uploaded_file
    )
    return process_button,


@app.cell
def __(process_button):
    process_button
    return


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
    process_status = mo.md("")
    
    # Check if button was clicked (value will be truthy when clicked)
    if process_button.value and uploaded_file:
        try:
            pdf_path = PDF_DIR / uploaded_file.name
            pdf_path.write_bytes(uploaded_file.contents)
            chunks = pdf_processor.process_pdf(pdf_path)
            chunks_with_embeddings = embedding_generator.embed_chunks(chunks)
            vector_store.add_chunks(chunks_with_embeddings)
            process_status = mo.md(f"✅ Processed {len(chunks)} chunks")
        except Exception as e:
            import traceback
            traceback.print_exc()
            process_status = mo.md(f"❌ Error: {str(e)}")
    
    return process_status,


@app.cell
def __(process_status):
    process_status
    return


@app.cell
def __(mo):
    mo.md("## Ask Questions")
    return


@app.cell
def __(mo):
    query_input = mo.ui.text_area(
        label="Your question:",
        placeholder="What is this document about?",
        rows=3
    )
    return query_input,


@app.cell
def __(query_input):
    query_input
    return


@app.cell
def __(mo, query_input):
    search_button = mo.ui.button(
        label="Get Answer",
        disabled=not query_input.value
    )
    return search_button,


@app.cell
def __(search_button):
    search_button
    return


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
    answer_display = mo.md("")
    
    # Check if button was clicked (value will be truthy when clicked)
    if search_button.value and query_input.value:
        try:
            query_embedding = embedding_generator.embed_query(query_input.value)
            search_results_raw = vector_store.search(query_embedding)
            search_results = parse_search_results(search_results_raw)
            
            if search_results:
                answer = llm_interface.generate_response(
                    query_input.value,
                    search_results
                )
                conversation_manager.add_exchange(query_input.value, answer)
                
                # Format sources
                sources_text = "\n\n".join([
                    f"**Source {i+1}:** {result.source_file} (Pages: {', '.join(map(str, result.page_numbers))})"
                    for i, result in enumerate(search_results)
                ])
                
                answer_display = mo.md(f"""
**Answer:** {answer}

**Sources:**
{sources_text}
                """)
            else:
                answer_display = mo.md("No relevant documents found. Please process a PDF first.")
        except Exception as e:
            import traceback
            traceback.print_exc()
            answer_display = mo.md(f"❌ Error: {str(e)}")
    
    return answer_display,


@app.cell
def __(answer_display):
    answer_display
    return


@app.cell
def __(mo, vector_store):
    # Show document statistics
    try:
        sources = vector_store.get_all_sources()
        total_chunks = vector_store.count
        
        if sources:
            doc_list = "\n".join(f"- {source}" for source in sorted(sources))
            stats = mo.md(f"""
## 📊 Document Statistics

**Documents indexed:** {len(sources)}  
**Total chunks:** {total_chunks}

**Available documents:**
{doc_list}
            """)
        else:
            stats = mo.md("## 📊 Document Statistics\n\n*No documents indexed yet*")
    except:
        stats = mo.md("## 📊 Document Statistics\n\n*No documents indexed yet*")
    
    return stats,


@app.cell
def __(stats):
    stats
    return


if __name__ == "__main__":
    app.run()
