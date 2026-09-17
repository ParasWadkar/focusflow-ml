"""SQLite connection management and schema initialisation."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from src.utils.errors import DatabaseError

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection with row access by name and foreign keys enforced."""
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as exc:
        raise DatabaseError(f"Could not open database at {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def initialize(db_path: Path, user_name: str) -> sqlite3.Connection:
    """Create the schema if needed, ensure the default user exists, and return a connection."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise DatabaseError(
                f"Database schema version {version} is newer than this application "
                f"supports ({SCHEMA_VERSION})."
            )
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (user_name,))
        conn.commit()
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise DatabaseError(f"Failed to initialise database at {db_path}: {exc}") from exc
    except DatabaseError:
        conn.close()
        raise
    return conn


def open_existing(db_path: Path) -> sqlite3.Connection:
    """Open an initialised database, failing clearly if ``init`` has not been run."""
    if not db_path.exists():
        raise DatabaseError(
            f"Database not found at {db_path}. Run `python -m src.main init` first."
        )
    conn = connect(db_path)
    try:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        conn.close()
        raise DatabaseError(f"{db_path} is not a valid FocusFlow database: {exc}") from exc
    if version != SCHEMA_VERSION:
        conn.close()
        raise DatabaseError(
            f"Database at {db_path} has schema version {version}; expected {SCHEMA_VERSION}. "
            "Run `python -m src.main init`."
        )
    return conn


def get_user_id(conn: sqlite3.Connection, user_name: str) -> int:
    row = conn.execute("SELECT id FROM users WHERE name = ?", (user_name,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO users (name) VALUES (?)", (user_name,))
        conn.commit()
        row = conn.execute("SELECT id FROM users WHERE name = ?", (user_name,)).fetchone()
    return int(row["id"])


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Commit on success, roll back on any exception (which is re-raised)."""
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
