from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class DocumentRegistry:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    normalized_path TEXT NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    chunk_count INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )

    def add(self, record: dict[str, Any]) -> dict[str, Any]:
        columns = (
            "document_id", "title", "category", "source_name", "source_url",
            "updated_at", "original_filename", "normalized_path", "content_hash",
            "chunk_count", "status", "created_at",
        )
        values = [record[column] for column in columns]
        with self._connect() as connection:
            try:
                connection.execute(
                    f"INSERT INTO documents ({', '.join(columns)}) "
                    f"VALUES ({', '.join('?' for _ in columns)})",
                    values,
                )
            except sqlite3.IntegrityError as exc:
                if "content_hash" in str(exc):
                    raise ValueError("相同内容的文档已经上传") from exc
                raise
        return record

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get(self, document_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if row is None:
            raise KeyError(document_id)
        return dict(row)

    def update_index(self, document_id: str, chunk_count: int, status: str = "indexed") -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE documents SET chunk_count = ?, status = ? WHERE document_id = ?",
                (chunk_count, status, document_id),
            )
        if cursor.rowcount != 1:
            raise KeyError(document_id)

    def delete(self, document_id: str) -> dict[str, Any]:
        record = self.get(document_id)
        with self._connect() as connection:
            connection.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
        return record

