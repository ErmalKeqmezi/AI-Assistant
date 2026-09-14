import tempfile
from pathlib import Path

import streamlit as st

from ai_assistant.config import Config
from ai_assistant.document_loader import load_document
from ai_assistant.history_store import HistoryStore
from ai_assistant.rag_pipeline import RAGAssistant
from ai_assistant.text_splitter import split_text

st.set_page_config(page_title="AI Assistant", page_icon="🤖")

EMPTY_STATE_MESSAGE = "Upload a document in the sidebar before asking a question."


@st.cache_resource(show_spinner="Starting your assistant...")
def get_assistant() -> RAGAssistant:
    return RAGAssistant()


@st.cache_resource(show_spinner="Starting your assistant...")
def get_history_store() -> HistoryStore:
    return HistoryStore(db_path=Config().history_db_path)


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


def render_uploaded_documents(assistant: RAGAssistant) -> None:
    with st.sidebar.expander("Uploaded Documents"):
        sources = assistant.vector_store.list_sources()
        if not sources:
            st.info(EMPTY_STATE_MESSAGE)
            return

        for doc in sources:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{doc['source']}**")
                st.caption(f"{doc['chunk_count']} chunk(s)")
            with col2:
                if st.button("🗑️", key=f"remove_{doc['source']}", help="Remove this document"):
                    assistant.vector_store.delete_source(doc["source"])
                    st.session_state.ingested_files.discard(doc["source"])
                    st.rerun()


def switch_to_conversation(history_store: HistoryStore, conversation_id: int) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.messages = history_store.get_messages(conversation_id)
    st.session_state.renaming_conversation_id = None
    st.rerun()


def start_new_chat() -> None:
    st.session_state.conversation_id = None
    st.session_state.messages = []
    st.session_state.renaming_conversation_id = None
    st.rerun()


def render_conversation_sidebar(history_store: HistoryStore, conversation_id: int | None) -> None:
    if st.button("+ New chat", key="new_chat", use_container_width=True):
        start_new_chat()

    conversations = history_store.list_conversations()

    if conversation_id is None:
        st.caption("New conversation")

    for conv in conversations:
        is_active = conv["id"] == conversation_id
        title = conv["title"] or "New conversation"

        label_col, rename_col, delete_col = st.columns([4, 1, 1])
        with label_col:
            if is_active:
                st.markdown(f"**{title}**")
            elif st.button(
                title, key=f"switch_conv_{conv['id']}", type="tertiary", use_container_width=True
            ):
                switch_to_conversation(history_store, conv["id"])
        with rename_col:
            if st.button(
                "✏️", key=f"rename_conv_{conv['id']}", help="Rename this chat", type="tertiary"
            ):
                st.session_state.renaming_conversation_id = conv["id"]
        with delete_col:
            if st.button(
                "🗑️", key=f"delete_conv_{conv['id']}", help="Delete this chat", type="tertiary"
            ):
                history_store.delete_conversation(conv["id"])
                if is_active:
                    start_new_chat()
                else:
                    st.rerun()

        if st.session_state.get("renaming_conversation_id") == conv["id"]:
            with st.form(f"rename_conversation_form_{conv['id']}"):
                new_title = st.text_input("Conversation name", value=title)
                save_col, cancel_col = st.columns(2)
                with save_col:
                    save_clicked = st.form_submit_button("Save")
                with cancel_col:
                    cancel_clicked = st.form_submit_button("Cancel")

            if save_clicked:
                cleaned = new_title.strip()
                if cleaned:
                    history_store.rename_conversation(conv["id"], cleaned)
                st.session_state.renaming_conversation_id = None
                st.rerun()
            elif cancel_clicked:
                st.session_state.renaming_conversation_id = None
                st.rerun()


assistant = get_assistant()
history_store = get_history_store()

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = history_store.get_latest_conversation_id()

if "messages" not in st.session_state:
    st.session_state.messages = (
        history_store.get_messages(st.session_state.conversation_id)
        if st.session_state.conversation_id is not None
        else []
    )

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

with st.sidebar:
    render_conversation_sidebar(history_store, st.session_state.conversation_id)
    st.divider()
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

    render_uploaded_documents(assistant)

st.title("AI Assistant")
st.caption("Ask questions grounded in the documents you've uploaded.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

has_documents = assistant.vector_store.count() > 0

if not has_documents:
    st.info(EMPTY_STATE_MESSAGE)
else:
    question = st.chat_input("Ask a question about your documents")
    if question:
        if st.session_state.conversation_id is None:
            st.session_state.conversation_id = history_store.create_conversation()
        st.session_state.messages.append({"role": "user", "content": question})
        history_store.add_message(st.session_state.conversation_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating an answer..."):
                answer, chunks = assistant.ask(question)
            st.markdown(answer)

        history_store.add_message(
            st.session_state.conversation_id, "assistant", answer, sources=chunks
        )
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": chunks}
        )
        # rerun so the sidebar picks up the just-auto-generated conversation title
        st.rerun()
