# AI Assistant

This project is a Retrieval-Augmented Generation (RAG) assistant built in Python.

## How it works

1. Documents are loaded from a directory (`.txt`, `.md`, `.pdf`).
2. Each document is split into overlapping text chunks.
3. Chunks are embedded with a local sentence-transformers model and stored in a
   persistent ChromaDB vector database.
4. When a user asks a question, the question is embedded and the most similar
   chunks are retrieved from ChromaDB.
5. The retrieved chunks are passed to Claude as context, and Claude answers the
   question grounded in that context.

## Components

- `ai_assistant/document_loader.py` - loads text, markdown, and PDF files.
- `ai_assistant/text_splitter.py` - chunks long text with configurable overlap.
- `ai_assistant/vector_store.py` - wraps ChromaDB for storing and querying embeddings.
- `ai_assistant/rag_pipeline.py` - retrieves context and calls the Claude API.
- `ai_assistant/cli.py` - command line interface (`ingest`, `chat`, `ask`).
