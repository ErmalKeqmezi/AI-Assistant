from ai_assistant.history_store import HistoryStore


def make_store(tmp_path):
    return HistoryStore(db_path=str(tmp_path / "history.db"))


def test_get_latest_conversation_id_returns_none_when_empty(tmp_path):
    store = make_store(tmp_path)
    assert store.get_latest_conversation_id() is None


def test_get_latest_conversation_id_returns_most_recent(tmp_path):
    store = make_store(tmp_path)
    store.create_conversation()
    second = store.create_conversation()

    assert store.get_latest_conversation_id() == second


def test_create_conversation_creates_a_new_row_each_time(tmp_path):
    store = make_store(tmp_path)
    first = store.create_conversation()
    second = store.create_conversation()

    assert first != second
    assert store.get_conversation(first) is not None
    assert store.get_conversation(second) is not None


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


def test_delete_conversation_deletes_messages_and_conversation(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()
    store.add_message(conversation_id, "user", "hello")

    store.delete_conversation(conversation_id)

    assert store.get_messages(conversation_id) == []
    assert store.get_conversation(conversation_id) is None


def test_list_conversations_empty(tmp_path):
    store = make_store(tmp_path)
    assert store.list_conversations() == []


def test_list_conversations_returns_most_recent_first(tmp_path):
    store = make_store(tmp_path)
    first = store.create_conversation()
    store.add_message(first, "user", "first chat")
    second = store.create_conversation()
    store.add_message(second, "user", "second chat")

    conversations = store.list_conversations()

    assert [c["id"] for c in conversations] == [second, first]
    assert conversations[0]["title"] == "second chat"
    assert conversations[1]["title"] == "first chat"


def test_list_conversations_does_not_include_deleted_conversation(tmp_path):
    store = make_store(tmp_path)
    keep = store.create_conversation()
    gone = store.create_conversation()

    store.delete_conversation(gone)

    assert [c["id"] for c in store.list_conversations()] == [keep]


def test_first_user_message_sets_conversation_title(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    store.add_message(conversation_id, "user", "What does this project do?")

    conversation = store.get_conversation(conversation_id)
    assert conversation["title"] == "What does this project do?"


def test_later_user_messages_do_not_overwrite_title(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    store.add_message(conversation_id, "user", "first question")
    store.add_message(conversation_id, "assistant", "an answer")
    store.add_message(conversation_id, "user", "second question")

    conversation = store.get_conversation(conversation_id)
    assert conversation["title"] == "first question"


def test_long_first_message_title_is_truncated_at_word_boundary(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    long_message = "explain how the retrieval augmented generation pipeline chunks and embeds documents before indexing them"
    store.add_message(conversation_id, "user", long_message)

    title = store.get_conversation(conversation_id)["title"]
    assert len(title) <= 61  # max_length + ellipsis
    assert title.endswith("…")
    assert not title[:-1].endswith(" ")  # no dangling space before the ellipsis


def test_assistant_only_conversation_has_no_title(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()

    store.add_message(conversation_id, "assistant", "unsolicited answer")

    assert store.get_conversation(conversation_id)["title"] is None


def test_get_conversation_returns_none_for_unknown_id(tmp_path):
    store = make_store(tmp_path)
    assert store.get_conversation(999) is None


def test_rename_conversation_updates_title(tmp_path):
    store = make_store(tmp_path)
    conversation_id = store.get_or_create_active_conversation()
    store.add_message(conversation_id, "user", "auto-generated title")

    store.rename_conversation(conversation_id, "My custom title")

    assert store.get_conversation(conversation_id)["title"] == "My custom title"


def test_rename_conversation_on_unknown_id_is_noop(tmp_path):
    store = make_store(tmp_path)
    store.rename_conversation(999, "does not exist")  # should not raise
    assert store.get_conversation(999) is None


def test_messages_persist_across_store_instances(tmp_path):
    db_path = str(tmp_path / "history.db")
    store1 = HistoryStore(db_path=db_path)
    conversation_id = store1.get_or_create_active_conversation()
    store1.add_message(conversation_id, "user", "hello")

    store2 = HistoryStore(db_path=db_path)
    messages = store2.get_messages(conversation_id)

    assert len(messages) == 1
    assert messages[0]["content"] == "hello"
