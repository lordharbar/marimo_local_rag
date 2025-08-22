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
    
    # Import the RAG system
    from src.rag_helpers import rag_system
    
    return mo, sys, Path, rag_system


@app.cell
def __(mo):
    mo.md("# 📚 Astrobase RAG System\n\nUpload PDFs and ask questions about their content using local LLMs.")
    return


@app.cell
def __(mo):
    mo.md("## 📄 Upload Document")
    return


@app.cell
def __(mo):
    file_upload = mo.ui.file(
        label="Select a PDF file",
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
        upload_display = mo.md(f"✅ **File ready:** {uploaded_file.name}")
    else:
        uploaded_file = None
        upload_display = mo.md("*No file selected*")
    
    upload_display
    return uploaded_file, upload_display


@app.cell
def __(mo, uploaded_file):
    # Process button
    process_button = mo.ui.button(
        label="Process PDF",
        disabled=not uploaded_file,
        kind="success"
    )
    process_button
    return process_button,


@app.cell
def __(mo, process_button, rag_system, uploaded_file):
    # Process the PDF when button is clicked
    process_result = mo.md("")
    
    if process_button.value and uploaded_file:
        # Show processing status
        mo.output.replace(mo.md("🔄 Processing PDF... Please wait..."))
        
        # Process the file
        success, message, chunk_count = rag_system.process_pdf(
            uploaded_file.name,
            uploaded_file.contents
        )
        
        # Show result
        if success:
            process_result = mo.md(f"✅ **Success!** {message}")
        else:
            process_result = mo.md(f"❌ **Error:** {message}")
        
        mo.output.replace(process_result)
    
    process_result
    return process_result, success, message, chunk_count


@app.cell
def __(mo):
    mo.md("## 💬 Ask Questions")
    return


@app.cell
def __(mo):
    # Question input
    query_input = mo.ui.text_area(
        label="Enter your question:",
        placeholder="What is the main topic of the document?",
        rows=3
    )
    query_input
    return query_input,


@app.cell
def __(mo, query_input):
    # Answer button
    answer_button = mo.ui.button(
        label="Get Answer",
        disabled=not query_input.value,
        kind="primary"
    )
    answer_button
    return answer_button,


@app.cell
def __(answer_button, mo, query_input, rag_system):
    # Generate answer when button is clicked
    answer_result = mo.md("")
    
    if answer_button.value and query_input.value:
        # Show processing status
        mo.output.replace(mo.md("🤔 Thinking..."))
        
        # Get answer
        success, answer, sources = rag_system.answer_question(query_input.value)
        
        if success:
            # Format sources
            if sources:
                sources_text = "\n\n**📍 Sources:**\n"
                for i, source in enumerate(sources, 1):
                    pages = ", ".join(map(str, source["pages"]))
                    sources_text += f"- {source['file']} (Pages: {pages})\n"
            else:
                sources_text = ""
            
            answer_result = mo.md(f"## 🎯 Answer\n\n{answer}{sources_text}")
        else:
            answer_result = mo.md(f"❌ {answer}")
        
        mo.output.replace(answer_result)
    
    answer_result
    return answer_result, success, answer, sources


@app.cell
def __(mo):
    mo.md("## 📊 System Status")
    return


@app.cell
def __(mo, rag_system):
    # Show statistics using the Statistics dataclass
    stats = rag_system.get_statistics()
    
    if stats.success and stats.num_documents > 0:
        doc_list = "\n".join([f"- {doc}" for doc in stats.documents])
        stats_display = mo.md(f"""
**Documents indexed:** {stats.num_documents}  
**Total chunks:** {stats.total_chunks}

**Files:**
{doc_list}
        """)
    else:
        stats_display = mo.md("*No documents indexed yet*")
    
    stats_display
    return stats, stats_display


@app.cell
def __(mo, rag_system):
    # Clear database button
    clear_button = mo.ui.button(
        label="Clear Database",
        kind="danger"
    )
    
    if clear_button.value:
        success, message = rag_system.clear_database()
        mo.md(f"{'✅' if success else '❌'} {message}")
    
    clear_button
    return clear_button, success, message


if __name__ == "__main__":
    app.run()
