import hashlib

import chromadb
from chromadb.utils import embedding_functions


class VectorStore:
    """Thin wrapper around a persistent ChromaDB collection."""

    def __init__(self, persist_dir: str, collection_name: str, embedding_model: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

    def add_chunks(self, chunks: list[str], source: str) -> int:
        if not chunks:
            return 0

        ids = [f"{source}:{i}:{hashlib.sha1(chunk.encode()).hexdigest()[:8]}" for i, chunk in enumerate(chunks)]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self._collection.upsert(ids=ids, documents=chunks, metadatas=metadatas)
        return len(chunks)

    def query(self, query_text: str, top_k: int = 4) -> list[dict]:
        results = self._collection.query(query_texts=[query_text], n_results=top_k)

        documents = results.get("documents") or [[]]
        metadatas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        return [
            {"text": doc, "source": meta.get("source"), "distance": dist}
            for doc, meta, dist in zip(documents[0], metadatas[0], distances[0])
        ]

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> list[dict]:
        """Return chunk counts per distinct source, derived from stored chunk metadata."""
        result = self._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        counts: dict[str, int] = {}
        for meta in metadatas:
            source = meta.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1

        return [
            {"source": source, "chunk_count": count}
            for source, count in sorted(counts.items())
        ]

    def delete_source(self, source: str) -> int:
        """Delete all chunks belonging to a source. Returns the number of chunks removed."""
        existing = self._collection.get(where={"source": source}, include=[])
        ids = existing.get("ids") or []
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)
