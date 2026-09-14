import pytest


@pytest.fixture(autouse=True)
def stub_sentence_transformer_embedding(monkeypatch):
    """Avoid downloading a real embedding model during tests.

    Patches chromadb's SentenceTransformerEmbeddingFunction with a deterministic
    hash-based embedder so vector_store tests run offline and fast.
    """
    import hashlib

    from chromadb.utils import embedding_functions

    class FakeEmbeddingFunction(embedding_functions.EmbeddingFunction):
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, input):  # noqa: A002 - chromadb's expected signature
            return [self._embed(text) for text in input]

        def _embed(self, text: str) -> list[float]:
            digest = hashlib.sha256(text.encode()).digest()
            return [b / 255.0 for b in digest[:16]]

        @staticmethod
        def name():
            return "fake"

    monkeypatch.setattr(
        embedding_functions, "SentenceTransformerEmbeddingFunction", FakeEmbeddingFunction
    )
    yield
