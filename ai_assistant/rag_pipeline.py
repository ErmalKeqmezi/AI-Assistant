from collections.abc import Iterator

import anthropic

from .config import Config
from .vector_store import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant that answers questions using the provided \
context excerpts retrieved from the user's documents.

Rules:
- Base your answer only on the given context. If the context doesn't contain the answer, \
say you don't know rather than guessing.
- Cite which source(s) you used by filename.
- Be concise and direct.
"""


class RAGAssistant:
    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.vector_store = VectorStore(
            persist_dir=self.config.chroma_persist_dir,
            collection_name=self.config.chroma_collection,
            embedding_model=self.config.embedding_model,
        )
        self.client = anthropic.Anthropic(api_key=self.config.anthropic_api_key or None)

    def retrieve(self, question: str) -> list[dict]:
        return self.vector_store.query(question, top_k=self.config.top_k)

    def _build_prompt(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            context_block = "(no relevant documents found)"
        else:
            context_block = "\n\n".join(
                f"[Source: {c['source']}]\n{c['text']}" for c in chunks
            )
        return f"Context:\n{context_block}\n\nQuestion: {question}"

    def ask(self, question: str) -> tuple[str, list[dict]]:
        """Non-streaming: returns (answer, retrieved_chunks)."""
        chunks = self.retrieve(question)
        prompt = self._build_prompt(question, chunks)

        response = self.client.messages.create(
            model=self.config.claude_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        answer = next((b.text for b in response.content if b.type == "text"), "")
        return answer, chunks

    def ask_stream(self, question: str) -> tuple[Iterator[str], list[dict]]:
        """Streaming: returns (text_iterator, retrieved_chunks)."""
        chunks = self.retrieve(question)
        prompt = self._build_prompt(question, chunks)

        stream = self.client.messages.stream(
            model=self.config.claude_model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )

        def text_iterator() -> Iterator[str]:
            with stream as s:
                for text in s.text_stream:
                    yield text

        return text_iterator(), chunks
