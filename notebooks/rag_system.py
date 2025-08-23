import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell
def setup_imports_and_helpers():
    import marimo as mo
    import sys
    from pathlib import Path

    # Add src to path for local development
    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Import the RAG system orchestrator
    from src.rag_helpers import rag_system

    return mo, sys, Path, rag_system


@app.cell
def title(mo):
    mo.md(
        "# 📚 Astrobase RAG System\n\nUpload PDFs and ask questions about their content using local LLMs."
    )
    return


@app.cell
def upload_section_header(mo):
    mo.md("---")
    mo.md("## 📄 1. Upload and Process Document")
    return


@app.cell
def initialize_upload_state(mo):
    # Use mo.state to manage UI outputs reactively
    process_status = mo.state("")
    upload_status = mo.state("*No file selected*")
    uploaded_file_state = mo.state(None)
    return process_status, upload_status, uploaded_file_state


@app.cell
def file_processing_interface(
    mo,
    process_status,
    rag_system,
    upload_status,
    uploaded_file_state,
):
    # This handler is called when a file is selected.
    def on_upload(files):
        if files:
            uploaded_file = files[0]
            uploaded_file_state.set_value(uploaded_file)
            upload_status.set_value(f"✅ **File ready:** {uploaded_file['name']}")
            process_status.set_value("")
        else:
            uploaded_file_state.set_value(None)
            upload_status.set_value("*No file selected*")

    # This handler is called when the process button is clicked.
    def on_process(value):
        uploaded_file = uploaded_file_state.value
        if uploaded_file:
            process_status.set_value("🔄 Processing PDF... Please wait...")
            success, message, chunk_count = rag_system.process_pdf(
                uploaded_file["name"], uploaded_file["contents"]
            )
            if success:
                process_status.set_value(f"✅ **Success!** {message}")
            else:
                process_status.set_value(f"❌ **Error:** {message}")

    file_upload = mo.ui.file(
        label="Select a PDF file", filetypes=[".pdf"], on_change=on_upload
    )
    process_button = mo.ui.button(
        label="Process PDF",
        disabled=not uploaded_file_state.value,
        kind="success",
        on_click=on_process,
    )
    mo.vstack([file_upload, upload_status, process_button, process_status])
    return file_upload, on_process, on_upload, process_button


@app.cell
def qa_section_header(mo):
    mo.md("---")
    mo.md("## 💬 2. Ask Questions")
    return


@app.cell
def initialize_qa_state(mo):
    # State for managing the question/answer flow
    answer_output = mo.state("")
    query_text = mo.state("")
    return answer_output, query_text


@app.cell
def question_answering_interface(answer_output, mo, query_text, rag_system):
    def on_query_change(value):
        query_text.set_value(value)
        if answer_output.value:
            answer_output.set_value("")

    def on_answer(value):
        if query_text.value:
            answer_output.set_value("🤔 Thinking...")
            success, answer, sources = rag_system.answer_question(query_text.value)
            if success:
                sources_text = ""
                if sources:
                    sources_text = "\n\n**📍 Sources:**\n"
                    for source in sources:
                        pages = ", ".join(map(str, source["pages"]))
                        sources_text += f"- {source['file']} (Pages: {pages})\n"
                final_answer = f"## 🎯 Answer\n\n{answer}{sources_text}"
                answer_output.set_value(final_answer)
            else:
                answer_output.set_value(f"❌ {answer}")

    query_input = mo.ui.text_area(
        label="Enter your question:",
        placeholder="What is the main topic of the document?",
        rows=3,
        on_change=on_query_change,
    )
    answer_button = mo.ui.button(
        label="Get Answer",
        disabled=not query_text.value,
        kind="primary",
        on_click=on_answer,
    )
    mo.vstack([query_input, answer_button, answer_output])
    return answer_button, on_answer, on_query_change, query_input


@app.cell
def status_section_header(mo):
    mo.md("---")
    mo.md("## 📊 System Status")
    return


@app.cell
def system_status_display(mo, process_status, rag_system):
    # Marimo knows to re-run this cell whenever process_status changes
    # because it's listed as a parameter in the function definition above.
    stats = rag_system.get_statistics()

    if stats.success and stats.num_documents > 0:
        doc_list = "\n".join([f"- {doc}" for doc in stats.documents])
        stats_display = mo.md(f"""**Documents indexed:** {stats.num_documents}**Total chunks:** {stats.total_chunks}**Files:**{doc_list}""")
    else:
        stats_display = mo.md("*No documents indexed yet*")

    stats_display
    return doc_list, stats, stats_display


@app.cell
def management_section_header(mo):
    mo.md("---")
    mo.md("## ⚙️ System Management")
    return


@app.cell
def system_management_interface(mo, rag_system):
    clear_status = mo.state("")

    def on_clear(value):
        clear_status.set_value("🗑️ Clearing database...")
        success, message = rag_system.clear_database()
        clear_status.set_value(f"{'✅' if success else '❌'} {message}")

    clear_button = mo.ui.button(
        label="Clear Database & Stored PDFs", kind="danger", on_click=on_clear
    )

    mo.vstack(
        [
            clear_button,
            mo.md(f"_{clear_status.value}_") if clear_status.value else mo.md(""),
        ]
    )
    return clear_button, clear_status, on_clear


if __name__ == "__main__":
    app.run()
