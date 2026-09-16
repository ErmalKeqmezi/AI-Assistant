# AI Assistant

A Retrieval-Augmented Generation (RAG) assistant built in Python, using
[ChromaDB](https://www.trychroma.com/) as the vector database, local
[sentence-transformers](https://www.sbert.net/) embeddings, and the
[Claude API](https://docs.anthropic.com/) for generation.

## How it works

1. Documents (`.txt`, `.md`, `.pdf`) are loaded and split into overlapping chunks.
2. Chunks are embedded locally and stored in a persistent ChromaDB collection.
3. A question is embedded and the most similar chunks are retrieved from ChromaDB.
4. The retrieved chunks are sent to Claude as context, which answers the question
   grounded in that context and cites its sources.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Usage

Ingest documents into the vector store:

```bash
python main.py ingest data/sample_docs
```

Ask a single question:

```bash
python main.py ask "How does this project work?"
```

Start an interactive chat session:

```bash
python main.py chat
```

## Streamlit UI

A browser-based alternative to the CLI, backed by the same `ai_assistant` package
(document loading, chunking, vector store, and RAG pipeline — no logic is duplicated).

```bash
streamlit run app.py
```

This opens a chat interface where you can:

- Upload `.txt` / `.md` / `.pdf` files from the sidebar, which are chunked and
  indexed into the same persistent ChromaDB collection used by the CLI.
- See how many chunks are currently indexed.
- Ask questions in a chat box; each answer shows a spinner while retrieval and
  generation run, and an expandable **Sources** section listing the chunks used.
- Clear the on-screen conversation with the sidebar button (this only resets the
  chat history — indexed documents stay in the vector store).

If no documents have been indexed yet, the app prompts you to upload one instead
of letting you ask a question.

## Configuration

All settings are read from environment variables (see `.env.example`):

| Variable              | Default              | Description                                  |
|------------------------|-----------------------|-----------------------------------------------|
| `ANTHROPIC_API_KEY`   | *(required)*          | Your Anthropic API key                        |
| `CLAUDE_MODEL`        | `claude-opus-5`       | Claude model used for generation              |
| `EMBEDDING_MODEL`     | `all-MiniLM-L6-v2`    | sentence-transformers model for embeddings    |
| `CHROMA_PERSIST_DIR`  | `./chroma_db`         | Where ChromaDB persists its data              |
| `CHROMA_COLLECTION`   | `documents`           | ChromaDB collection name                      |
| `CHUNK_SIZE`          | `800`                 | Max characters per chunk                      |
| `CHUNK_OVERLAP`       | `150`                 | Overlap (characters) between chunks           |
| `TOP_K`               | `4`                   | Number of chunks retrieved per question       |

## Project layout

```
app.py                  # Streamlit UI
ai_assistant/
├── config.py          # env-based configuration
├── document_loader.py # loads .txt/.md/.pdf files
├── text_splitter.py   # chunking with overlap
├── vector_store.py    # ChromaDB wrapper
├── rag_pipeline.py     # retrieval + Claude generation
└── cli.py              # ingest / ask / chat commands
```

## Tests

```bash
pytest
```

## Author

**Ermal Keqmezi**
Software Developer & AI Engineer

## License

This project is available for educational and portfolio purposes.
