import argparse
import sys

from .config import Config
from .document_loader import load_documents
from .rag_pipeline import RAGAssistant
from .text_splitter import split_text
from .vector_store import VectorStore


def cmd_ingest(args: argparse.Namespace) -> None:
    config = Config()
    store = VectorStore(
        persist_dir=config.chroma_persist_dir,
        collection_name=config.chroma_collection,
        embedding_model=config.embedding_model,
    )

    documents = load_documents(args.path)
    if not documents:
        print(f"No supported documents found in {args.path}")
        return

    total_chunks = 0
    for doc in documents:
        chunks = split_text(doc.text, config.chunk_size, config.chunk_overlap)
        added = store.add_chunks(chunks, source=doc.source)
        total_chunks += added
        print(f"Ingested {doc.source}: {added} chunks")

    print(f"\nDone. {total_chunks} chunks from {len(documents)} document(s) "
          f"now in collection '{config.chroma_collection}' ({store.count()} total).")


def cmd_chat(args: argparse.Namespace) -> None:
    assistant = RAGAssistant()

    if assistant.vector_store.count() == 0:
        print("Warning: the vector store is empty. Run `ingest` first.\n")

    print("AI Assistant (RAG). Type 'exit' or 'quit' to leave.\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        text_iterator, chunks = assistant.ask_stream(question)
        print("Assistant: ", end="", flush=True)
        for piece in text_iterator:
            print(piece, end="", flush=True)
        print()

        if chunks:
            sources = sorted({c["source"] for c in chunks})
            print(f"  (sources: {', '.join(sources)})")
        print()


def cmd_ask(args: argparse.Namespace) -> None:
    assistant = RAGAssistant()
    answer, chunks = assistant.ask(args.question)
    print(answer)
    if chunks:
        sources = sorted({c["source"] for c in chunks})
        print(f"\n(sources: {', '.join(sources)})")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ai-assistant", description="RAG-powered AI assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load documents into the vector store")
    ingest_parser.add_argument("path", help="Directory containing .txt/.md/.pdf files")
    ingest_parser.set_defaults(func=cmd_ingest)

    chat_parser = subparsers.add_parser("chat", help="Start an interactive RAG chat session")
    chat_parser.set_defaults(func=cmd_chat)

    ask_parser = subparsers.add_parser("ask", help="Ask a single question and exit")
    ask_parser.add_argument("question", help="The question to ask")
    ask_parser.set_defaults(func=cmd_ask)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
