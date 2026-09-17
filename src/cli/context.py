"""Per-invocation CLI context: settings plus a lazily opened database."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.config import Settings
from src.database.connection import get_user_id, open_existing


@dataclass
class AppContext:
    settings: Settings
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    _user_id: int | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = open_existing(self.settings.db_path)
        return self._conn

    @property
    def user_id(self) -> int:
        if self._user_id is None:
            self._user_id = get_user_id(self.conn, self.settings.user_name)
        return self._user_id

    def db_available(self) -> bool:
        return self._conn is not None or self.settings.db_path.exists()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
