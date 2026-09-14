import tempfile
from pathlib import Path

import streamlit as st

from ai_assistant.document_loader import load_document
from ai_assistant.rag_pipeline import RAGAssistant
from ai_assistant.text_splitter import split_text

st.set_page_config(page_title="AI Assistant", page_icon="🤖")


@st.cache_resource
def get_assistant() -> RAGAssistant:
    return RAGAssistant()


def ingest_uploaded_file(assistant: RAGAssistant, uploaded_file) -> int:
    """Write the upload to a temp file so we can reuse load_document(), then chunk + index it."""
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = Path(tmp.name)

    try:
        document = load_document(tmp_path)
        chunks = split_text(
            document.text, assistant.config.chunk_size, assistant.config.chunk_overlap
        )
        return assistant.vector_store.add_chunks(chunks, source=uploaded_file.name)
    finally:
        tmp_path.unlink(missing_ok=True)


def render_sources(chunks: list[dict]) -> None:
    with st.expander("Sources"):
        for chunk in chunks:
            st.markdown(f"**{chunk['source']}**")
            st.text(chunk["text"])


assistant = get_assistant()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload documents", type=["txt", "md", "pdf"], accept_multiple_files=True
    )
    if uploaded_files:
        for uploaded_file in uploaded_files:
            if uploaded_file.name in st.session_state.ingested_files:
                continue
            with st.spinner(f"Ingesting {uploaded_file.name}..."):
                added = ingest_uploaded_file(assistant, uploaded_file)
            st.session_state.ingested_files.add(uploaded_file.name)
            st.success(f"Added {added} chunks from {uploaded_file.name}")

    st.metric("Indexed chunks", assistant.vector_store.count())

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("AI Assistant")
st.caption("Ask questions grounded in the documents you've uploaded.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

has_documents = assistant.vector_store.count() > 0

if not has_documents:
    st.info("Upload a document in the sidebar before asking a question.")
else:
    question = st.chat_input("Ask a question about your documents")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                answer, chunks = assistant.ask(question)
            st.markdown(answer)
            if chunks:
                render_sources(chunks)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": chunks}
        )
