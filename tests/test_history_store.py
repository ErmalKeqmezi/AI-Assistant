from ai_assistant.history_store import HistoryStore


def make_store(tmp_path):
    return HistoryStore(db_path=str(tmp_path / "history.db"))


def test_get_or_create_active_conversation_creates_one_when_none_exists(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    assert isinstance(conversation_id, int)
    assert store.get_messages(conversation_id) == []


def test_get_or_create_active_conversation_returns_same_id_on_repeat_calls(tmp_path):
    store = make_store(tmp_path)
    first = store.get_or_create_active_conversation()
    second = store.get_or_create_active_conversation()

    assert first == second


def test_add_message_and_get_messages_round_trip(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    store.add_message(conversation_id, "user", "hello")
    store.add_message(conversation_id, "assistant", "hi there", sources=[{"source": "doc.txt", "text": "..."}])

    messages = store.get_messages(conversation_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "hello"
    assert messages[0]["sources"] is None
    assert messages[1]["role"] == "assistant"
    assert messages[1]["sources"] == [{"source": "doc.txt", "text": "..."}]


def test_get_messages_preserves_insertion_order(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    for i in range(5):
        store.add_message(conversation_id, "user", f"message {i}")

    messages = store.get_messages(conversation_id)
    assert [m["content"] for m in messages] == [f"message {i}" for i in range(5)]


def test_get_messages_respects_limit_and_stays_chronological(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    for i in range(5):
        store.add_message(conversation_id, "user", f"message {i}")

    messages = store.get_messages(conversation_id, limit=2)
    assert [m["content"] for m in messages] == ["message 3", "message 4"]


def test_clear_conversation_deletes_messages_and_conversation(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()
    store.add_message(conversation_id, "user", "hello")

    store.clear_conversation(conversation_id)

    assert store.get_messages(conversation_id) == []

    # a fresh call should mint a brand-new conversation, not reuse the cleared one
    new_id = store.get_or_create_active_conversation()
    assert new_id != conversation_id


def test_messages_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "history.db")
    store1 = HistoryStore(db_path=db_path)
    conversation_id = store1.get_or_create_active_conversation()
    store1.add_message(conversation_id, "user", "hello")

    store2 = HistoryStore(db_path=db_path)
    messages = store2.get_messages(conversation_id)

    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
