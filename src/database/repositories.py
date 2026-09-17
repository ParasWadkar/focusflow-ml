"""Data-access layer.

Repositories only translate between Python values and SQL. They perform no
business validation beyond what the schema enforces, and they convert SQLite
integrity errors into application exceptions. Rows are returned as plain
dictionaries so the domain layer stays independent of ``sqlite3``.
"""

from __future__ import annotations

import sqlite3
from typing import Any, Iterable

from src.utils.errors import DatabaseError, DuplicateError

Row = dict[str, Any]


def _rows(cursor: sqlite3.Cursor) -> list[Row]:
    return [dict(r) for r in cursor.fetchall()]


def _one(cursor: sqlite3.Cursor) -> Row | None:
    row = cursor.fetchone()
    return dict(row) if row is not None else None


def _set_clause(fields: dict[str, Any], allowed: Iterable[str]) -> tuple[str, list[Any]]:
    allowed_set = set(allowed)
    unknown = set(fields) - allowed_set
    if unknown:
        raise ValueError(f"Unsupported update fields: {sorted(unknown)}")
    columns = sorted(fields)
    return ", ".join(f"{c} = ?" for c in columns), [fields[c] for c in columns]


def _execute(conn: sqlite3.Connection, sql: str, params: Iterable[Any] = (),
             duplicate_message: str | None = None) -> sqlite3.Cursor:
    try:
        return conn.execute(sql, tuple(params))
    except sqlite3.IntegrityError as exc:
        if duplicate_message and "UNIQUE" in str(exc).upper():
            raise DuplicateError(duplicate_message) from exc
        raise DatabaseError(f"Database constraint violated: {exc}") from exc
    except sqlite3.OperationalError as exc:
        raise DatabaseError(f"Database operation failed: {exc}") from exc


class HabitRepository:
    UPDATABLE = ("name", "description", "schedule", "start_date", "archived")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, user_id: int, name: str, description: str | None,
               schedule: str, start_date: str) -> int:
        cur = _execute(
            self.conn,
            "INSERT INTO habits (user_id, name, description, schedule, start_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, name, description, schedule, start_date),
            duplicate_message=f"A habit named '{name}' already exists.",
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, user_id: int, habit_id: int) -> Row | None:
        return _one(_execute(self.conn, "SELECT * FROM habits WHERE id = ? AND user_id = ?",
                             (habit_id, user_id)))

    def get_by_name(self, user_id: int, name: str) -> Row | None:
        return _one(_execute(self.conn,
                             "SELECT * FROM habits WHERE user_id = ? AND name = ? COLLATE NOCASE",
                             (user_id, name)))

    def list(self, user_id: int, include_archived: bool = False) -> list[Row]:
        sql = "SELECT * FROM habits WHERE user_id = ?"
        if not include_archived:
            sql += " AND archived = 0"
        return _rows(_execute(self.conn, sql + " ORDER BY id", (user_id,)))

    def update(self, user_id: int, habit_id: int, fields: dict[str, Any]) -> None:
        if not fields:
            return
        clause, values = _set_clause(fields, self.UPDATABLE)
        _execute(self.conn, f"UPDATE habits SET {clause} WHERE id = ? AND user_id = ?",
                 [*values, habit_id, user_id],
                 duplicate_message=f"A habit named '{fields.get('name')}' already exists.")
        self.conn.commit()

    def delete(self, user_id: int, habit_id: int) -> bool:
        cur = _execute(self.conn, "DELETE FROM habits WHERE id = ? AND user_id = ?",
                       (habit_id, user_id))
        self.conn.commit()
        return cur.rowcount > 0


class HabitRecordRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def upsert(self, habit_id: int, day: str, completed: bool, note: str | None) -> None:
        _execute(
            self.conn,
            "INSERT INTO habit_records (habit_id, date, completed, note) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(habit_id, date) DO UPDATE SET completed = excluded.completed, "
            "note = COALESCE(excluded.note, habit_records.note), updated_at = datetime('now')",
            (habit_id, day, int(completed), note),
        )
        self.conn.commit()

    def delete(self, habit_id: int, day: str) -> bool:
        cur = _execute(self.conn, "DELETE FROM habit_records WHERE habit_id = ? AND date = ?",
                       (habit_id, day))
        self.conn.commit()
        return cur.rowcount > 0

    def for_habit(self, habit_id: int, start: str | None = None, end: str | None = None) -> list[Row]:
        sql = "SELECT * FROM habit_records WHERE habit_id = ?"
        params: list[Any] = [habit_id]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        return _rows(_execute(self.conn, sql + " ORDER BY date", params))

    def for_user(self, user_id: int, start: str | None = None, end: str | None = None) -> list[Row]:
        sql = ("SELECT r.* FROM habit_records r JOIN habits h ON h.id = r.habit_id "
               "WHERE h.user_id = ?")
        params: list[Any] = [user_id]
        if start:
            sql += " AND r.date >= ?"
            params.append(start)
        if end:
            sql += " AND r.date <= ?"
            params.append(end)
        return _rows(_execute(self.conn, sql + " ORDER BY r.date", params))


class TimeBlockRepository:
    UPDATABLE = ("date", "start_time", "end_time", "task_name", "category", "priority", "completed")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def create(self, user_id: int, values: dict[str, Any]) -> int:
        cur = _execute(
            self.conn,
            "INSERT INTO time_blocks (user_id, date, start_time, end_time, task_name, category, "
            "priority, completed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, values["date"], values["start_time"], values["end_time"],
             values["task_name"], values["category"], values["priority"],
             int(values.get("completed", False))),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def get(self, user_id: int, block_id: int) -> Row | None:
        return _one(_execute(self.conn, "SELECT * FROM time_blocks WHERE id = ? AND user_id = ?",
                             (block_id, user_id)))

    def list(self, user_id: int, start: str | None = None, end: str | None = None) -> list[Row]:
        sql = "SELECT * FROM time_blocks WHERE user_id = ?"
        params: list[Any] = [user_id]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        return _rows(_execute(self.conn, sql + " ORDER BY date, start_time", params))

    def update(self, user_id: int, block_id: int, fields: dict[str, Any]) -> None:
        if not fields:
            return
        clause, values = _set_clause(fields, self.UPDATABLE)
        _execute(self.conn, f"UPDATE time_blocks SET {clause} WHERE id = ? AND user_id = ?",
                 [*values, block_id, user_id])
        self.conn.commit()

    def delete(self, user_id: int, block_id: int) -> bool:
        cur = _execute(self.conn, "DELETE FROM time_blocks WHERE id = ? AND user_id = ?",
                       (block_id, user_id))
        self.conn.commit()
        return cur.rowcount > 0


class DailyRecordRepository:
    FIELDS = ("sleep_hours", "energy_level", "mood_score", "exercise_minutes",
              "interruptions", "journal", "productivity_score")

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def get(self, user_id: int, day: str) -> Row | None:
        return _one(_execute(self.conn,
                             "SELECT * FROM daily_records WHERE user_id = ? AND date = ?",
                             (user_id, day)))

    def create(self, user_id: int, day: str, fields: dict[str, Any]) -> int:
        _set_clause(fields, self.FIELDS)  # rejects unknown fields
        columns = ["user_id", "date", *sorted(fields)]
        placeholders = ", ".join("?" for _ in columns)
        cur = _execute(
            self.conn,
            f"INSERT INTO daily_records ({', '.join(columns)}) VALUES ({placeholders})",
            [user_id, day, *[fields[c] for c in sorted(fields)]],
            duplicate_message=f"A daily record for {day} already exists.",
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def update(self, user_id: int, day: str, fields: dict[str, Any]) -> None:
        if not fields:
            return
        clause, values = _set_clause(fields, self.FIELDS)
        _execute(self.conn,
                 f"UPDATE daily_records SET {clause}, updated_at = datetime('now') "
                 "WHERE user_id = ? AND date = ?",
                 [*values, user_id, day])
        self.conn.commit()

    def list(self, user_id: int, start: str | None = None, end: str | None = None) -> list[Row]:
        sql = "SELECT * FROM daily_records WHERE user_id = ?"
        params: list[Any] = [user_id]
        if start:
            sql += " AND date >= ?"
            params.append(start)
        if end:
            sql += " AND date <= ?"
            params.append(end)
        return _rows(_execute(self.conn, sql + " ORDER BY date", params))

    def delete(self, user_id: int, day: str) -> bool:
        cur = _execute(self.conn, "DELETE FROM daily_records WHERE user_id = ? AND date = ?",
                       (user_id, day))
        self.conn.commit()
        return cur.rowcount > 0


class PhotoRepository:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def add(self, daily_record_id: int, file_path: str, caption: str | None) -> int:
        cur = _execute(self.conn,
                       "INSERT INTO day_photos (daily_record_id, file_path, caption) VALUES (?, ?, ?)",
                       (daily_record_id, file_path, caption))
        self.conn.commit()
        return int(cur.lastrowid)

    def for_record(self, daily_record_id: int) -> list[Row]:
        return _rows(_execute(self.conn,
                              "SELECT * FROM day_photos WHERE daily_record_id = ? ORDER BY id",
                              (daily_record_id,)))
