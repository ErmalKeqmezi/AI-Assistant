from ai_assistant.text_splitter import split_text


def test_empty_text_returns_no_chunks():
    assert split_text("") == []
    assert split_text("   ") == []


def test_short_text_returns_single_chunk():
    chunks = split_text("Hello world.", chunk_size=800, chunk_overlap=150)
    assert chunks == ["Hello world."]


def test_long_text_is_split_into_multiple_chunks():
    paragraph = "This is a sentence about the project. " * 50
    chunks = split_text(paragraph, chunk_size=200, chunk_overlap=50)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 200 + 50  # allow for overlap growth


def test_overlap_produces_longer_tail_chunks():
    text = "Paragraph one has some words.\n\nParagraph two has other words.\n\n" * 10

    no_overlap = split_text(text, chunk_size=100, chunk_overlap=1)
    with_overlap = split_text(text, chunk_size=100, chunk_overlap=30)

    assert len(with_overlap) > 1
    # overlapping chunks (after the first) should be longer than their no-overlap counterparts
    for a, b in zip(no_overlap[1:], with_overlap[1:]):
        assert len(b) >= len(a)


def test_rejects_overlap_larger_than_chunk_size():
    import pytest

    with pytest.raises(ValueError):
        split_text("some text", chunk_size=100, chunk_overlap=100)
