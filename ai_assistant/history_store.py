import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    title TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    sources TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
"""


class HistoryStore:
    """SQLite-backed storage for conversation history.

    The schema supports multiple conversations, but for now the app only ever
    uses a single ongoing one: get_or_create_active_conversation() always
    returns the most recently created conversation, creating one if none
    exists yet.
    """

    def __init__(self, db_path: str):
        self._db_path = db_path
        with closing(self._connect()) as conn:
            conn.executescript(SCHEMA)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        return conn

    def get_latest_conversation_id(self) -> int | None:
        """Return the most recent conversation's id, or None if none exist yet (does not create one)."""
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT id FROM conversations ORDER BY id DESC LIMIT 1").fetchone()
        return row["id"] if row else None

    def create_conversation(self) -> int:
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO conversations (created_at, title) VALUES (?, ?)",
                (_now(), None),
            )
            conn.commit()
            return cursor.lastrowid

    def get_or_create_active_conversation(self) -> int:
        conversation_id = self.get_latest_conversation_id()
        if conversation_id is not None:
            return conversation_id
        return self.create_conversation()

    def add_message(
        self,
        conversation_id: int,
        role: str,
        content: str,
        sources: list[dict] | None = None,
    ) -> int:
        sources_json = json.dumps(sources) if sources else None
        with closing(self._connect()) as conn:
            cursor = conn.execute(
                "INSERT INTO messages (conversation_id, role, content, sources, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (conversation_id, role, content, sources_json, _now()),
            )
            if role == "user":
                # only fills the title in the first time (title IS NULL); later
                # user messages don't overwrite it
                conn.execute(
                    "UPDATE conversations SET title = ? WHERE id = ? AND title IS NULL",
                    (_generate_title(content), conversation_id),
                )
            conn.commit()
            return cursor.lastrowid

    def get_conversation(self, conversation_id: int) -> dict | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT id, created_at, title FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()

        if row is None:
            return None
        return {"id": row["id"], "created_at": row["created_at"], "title": row["title"]}

    def get_messages(self, conversation_id: int, limit: int | None = None) -> list[dict]:
        params: tuple = (conversation_id,)
        if limit is None:
            query = (
                "SELECT role, content, sources, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id ASC"
            )
        else:
            # take the most recent `limit` rows, then re-sort them chronologically
            query = (
                "SELECT role, content, sources, created_at FROM ("
                "SELECT id, role, content, sources, created_at FROM messages "
                "WHERE conversation_id = ? ORDER BY id DESC LIMIT ?"
                ") ORDER BY id ASC"
            )
            params = (conversation_id, limit)

        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
                "sources": json.loads(row["sources"]) if row["sources"] else None,
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def rename_conversation(self, conversation_id: int, title: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conversation_id),
            )
            conn.commit()

    def clear_conversation(self, conversation_id: int) -> None:
        """Delete a conversation and all its messages (cascades via foreign key)."""
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            conn.commit()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generate_title(content: str, max_length: int = 60) -> str:
    text = " ".join(content.split())
    if len(text) <= max_length:
        return text

    truncated = text[:max_length].rsplit(" ", 1)[0] or text[:max_length]
    return f"{truncated}…"
