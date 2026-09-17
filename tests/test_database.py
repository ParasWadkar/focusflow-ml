import sqlite3

import pytest

from src.config import load_settings
from src.database.connection import SCHEMA_VERSION, initialize, open_existing
from src.database.repositories import DailyRecordRepository, HabitRecordRepository, HabitRepository
from src.utils.errors import DatabaseError, DuplicateError


def test_initialize_creates_all_tables(conn):
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"users", "habits", "habit_records", "time_blocks", "daily_records", "day_photos"} <= tables
    assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_initialize_is_idempotent(settings, conn, user_id):
    HabitRepository(conn).create(user_id, "Read", None, "daily", "2026-01-01")
    again = initialize(settings.db_path, settings.user_name)
    assert len(HabitRepository(again).list(user_id)) == 1
    again.close()


def test_open_existing_requires_init(tmp_path):
    with pytest.raises(DatabaseError, match="init"):
        open_existing(tmp_path / "missing.db")


def test_open_existing_rejects_non_database_file(tmp_path):
    bogus = tmp_path / "bogus.db"
    bogus.write_text("this is not sqlite", encoding="utf-8")
    with pytest.raises(DatabaseError):
        open_existing(bogus)


def test_duplicate_habit_name_raises(conn, user_id):
    repo = HabitRepository(conn)
    repo.create(user_id, "Read", None, "daily", "2026-01-01")
    with pytest.raises(DuplicateError):
        repo.create(user_id, "Read", None, "daily", "2026-01-01")


def test_deleting_habit_cascades_to_records(conn, user_id):
    habits = HabitRepository(conn)
    records = HabitRecordRepository(conn)
    habit_id = habits.create(user_id, "Read", None, "daily", "2026-01-01")
    records.upsert(habit_id, "2026-01-02", True, None)
    habits.delete(user_id, habit_id)
    assert conn.execute("SELECT COUNT(*) FROM habit_records").fetchone()[0] == 0


def test_habit_record_upsert_overwrites(conn, user_id):
    habit_id = HabitRepository(conn).create(user_id, "Read", None, "daily", "2026-01-01")
    records = HabitRecordRepository(conn)
    records.upsert(habit_id, "2026-01-02", True, "first")
    records.upsert(habit_id, "2026-01-02", False, None)
    rows = records.for_habit(habit_id)
    assert len(rows) == 1
    assert rows[0]["completed"] == 0
    assert rows[0]["note"] == "first"


def test_schema_check_constraints_enforced(conn, user_id):
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO daily_records (user_id, date, mood_score) VALUES (?, '2026-01-01', 11)",
                     (user_id,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO time_blocks (user_id, date, start_time, end_time, task_name, category) "
            "VALUES (?, '2026-01-01', '10:00', '09:00', 'x', 'study')", (user_id,))


def test_daily_record_duplicate_date(conn, user_id):
    repo = DailyRecordRepository(conn)
    repo.create(user_id, "2026-01-01", {"sleep_hours": 7.0})
    with pytest.raises(DuplicateError):
        repo.create(user_id, "2026-01-01", {"sleep_hours": 6.0})


def test_settings_env_override(tmp_path):
    s = load_settings(root=tmp_path, env={"FOCUSFLOW_DATA_DIR": "custom", "FOCUSFLOW_USER_NAME": "me"})
    assert s.data_dir == tmp_path / "custom"
    assert s.db_path == tmp_path / "custom" / "focusflow.db"
    assert s.user_name == "me"


def test_settings_config_file(tmp_path):
    (tmp_path / "focusflow.toml").write_text('[paths]\nmodels_dir = "m"\n', encoding="utf-8")
    s = load_settings(root=tmp_path, env={})
    assert s.models_dir == tmp_path / "m"
