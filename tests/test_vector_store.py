from ai_assistant.vector_store import VectorStore


def make_store(tmp_path):
    return VectorStore(
        persist_dir=str(tmp_path / "chroma"),
        collection_name="test_collection",
        embedding_model="fake-model",
    )


def test_add_chunks_and_count(tmp_path):
    store = make_store(tmp_path)
    added = store.add_chunks(["chunk one", "chunk two"], source="doc1.txt")

    assert added == 2
    assert store.count() == 2


def test_add_chunks_with_empty_list_is_noop(tmp_path):
    store = make_store(tmp_path)
    assert store.add_chunks([], source="doc1.txt") == 0
    assert store.count() == 0


def test_query_returns_source_metadata(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(["Paris is the capital of France."], source="geo.txt")
    store.add_chunks(["Bananas are yellow fruit."], source="food.txt")

    results = store.query("What is the capital of France?", top_k=2)

    assert len(results) == 2
    sources = {r["source"] for r in results}
    assert sources == {"geo.txt", "food.txt"}
    for r in results:
        assert "text" in r and "distance" in r


def test_upsert_deduplicates_identical_chunks(tmp_path):
    store = make_store(tmp_path)
    store.add_chunks(["same chunk"], source="doc.txt")
    store.add_chunks(["same chunk"], source="doc.txt")

    assert store.count() == 1
