def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph/sentence breaks."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    separators = ["\n\n", "\n", ". ", " "]
    raw_chunks = _split_raw(text, chunk_size, separators)
    return _apply_overlap(raw_chunks, chunk_overlap)


def _split_raw(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    """Split text into non-overlapping chunks, each at most chunk_size long."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    separator = separators[0] if separators else ""
    parts = text.split(separator) if separator else list(text)

    chunks: list[str] = []
    current = ""
    for part in parts:
        candidate = current + separator + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
        if len(part) > chunk_size:
            chunks.extend(_split_raw(part, chunk_size, separators[1:]))
            current = ""
        else:
            current = part

    if current:
        chunks.append(current)

    return chunks


def _apply_overlap(chunks: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0 or len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for prev, curr in zip(chunks, chunks[1:]):
        tail = prev[-chunk_overlap:]
        result.append(f"{tail}{curr}")
    return result
