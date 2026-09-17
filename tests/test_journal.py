from datetime import date, timedelta

import pytest

from src.journal.service import JournalService
from src.utils.errors import NotFoundError, ValidationError


@pytest.fixture
def journal(conn, user_id, settings) -> JournalService:
    return JournalService(conn, user_id, settings.photos_dir)


def test_create_and_incrementally_update(journal):
    journal.record("2026-09-01", sleep_hours=7.5, energy_level=7)
    rec = journal.record("2026-09-01", mood_score=6, productivity_score=72, journal="Good focus.")
    assert rec.sleep_hours == 7.5 and rec.energy_level == 7
    assert rec.mood_score == 6 and rec.productivity_score == 72 and rec.journal == "Good focus."


def test_clear_field(journal):
    journal.record("2026-09-01", sleep_hours=7, mood_score=5)
    rec = journal.record("2026-09-01", clear=["mood_score"])
    assert rec.mood_score is None and rec.sleep_hours == 7


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"mood_score": 11}, "at most 10"),
        ({"energy_level": 0}, "at least 1"),
        ({"sleep_hours": -1}, "negative"),
        ({"exercise_minutes": -5}, "negative"),
        ({"interruptions": 2.5}, "whole number"),
        ({"productivity_score": 101}, "at most 100"),
        ({"sleep_hours": "lots"}, "number"),
    ],
)
def test_invalid_values_rejected(journal, fields, match):
    with pytest.raises(ValidationError, match=match):
        journal.record("2026-09-01", **fields)


def test_future_date_rejected(journal):
    with pytest.raises(ValidationError, match="future"):
        journal.record((date.today() + timedelta(days=1)).isoformat(), sleep_hours=7)


def test_empty_record_rejected(journal):
    with pytest.raises(ValidationError, match="at least one"):
        journal.record("2026-09-01")


def test_list_and_delete(journal):
    journal.record("2026-09-01", sleep_hours=7)
    journal.record("2026-09-03", sleep_hours=8)
    assert [r.date.day for r in journal.list("2026-09-01", "2026-09-02")] == [1]
    journal.delete("2026-09-01")
    with pytest.raises(NotFoundError):
        journal.get("2026-09-01")
    with pytest.raises(NotFoundError):
        journal.delete("2026-09-01")


def test_add_photo_copies_file(journal, tmp_path, settings):
    src = tmp_path / "walk.png"
    src.write_bytes(b"\x89PNG fake image bytes")
    journal.record("2026-09-01", sleep_hours=7)
    photo = journal.add_photo("2026-09-01", src, "Evening walk")
    assert (settings.photos_dir / photo.file_path).read_bytes() == src.read_bytes()
    second = journal.add_photo("2026-09-01", src)
    assert second.file_path != photo.file_path
    assert len(journal.get("2026-09-01").photos) == 2


def test_add_photo_validation(journal, tmp_path):
    journal.record("2026-09-01", sleep_hours=7)
    with pytest.raises(ValidationError, match="not found"):
        journal.add_photo("2026-09-01", tmp_path / "missing.jpg")
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    with pytest.raises(ValidationError, match="Unsupported"):
        journal.add_photo("2026-09-01", txt)
    with pytest.raises(NotFoundError):
        journal.add_photo("2026-09-02", txt)
