"""Daily record and journal business logic."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from src.database.repositories import DailyRecordRepository, PhotoRepository
from src.journal.models import ALL_FIELDS, NUMERIC_FIELDS, DailyRecord, Photo
from src.journal.photos import store_photo
from src.utils import validators as v
from src.utils.errors import NotFoundError, ValidationError


def validate_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """Validate the provided (non-``None``) daily-record fields."""
    clean: dict[str, Any] = {}
    for name, value in fields.items():
        if name not in ALL_FIELDS:
            raise ValidationError(f"Unknown daily record field '{name}'.")
        if value is None:
            continue
        if name in NUMERIC_FIELDS:
            kind, lo, hi = NUMERIC_FIELDS[name]
            label = name.replace("_", " ")
            clean[name] = (v.int_in_range(value, label, int(lo), int(hi)) if kind is int
                           else v.number_in_range(value, label, lo, hi))
        else:
            clean[name] = v.optional_text(value, name, max_length=20_000)
    return clean


class JournalService:
    def __init__(self, conn: sqlite3.Connection, user_id: int, photos_dir: Path | None = None) -> None:
        self.user_id = user_id
        self.repo = DailyRecordRepository(conn)
        self.photos = PhotoRepository(conn)
        self.photos_dir = photos_dir

    def record(self, day: str | date | None = None, clear: Iterable[str] = (), **fields: Any) -> DailyRecord:
        """Create or update the record for ``day``.

        Only the supplied fields change, so a day can be filled in gradually
        (e.g. sleep in the morning, score in the evening). Names in ``clear``
        are reset to empty. Future dates are rejected.
        """
        target = v.parse_date(day, "date", allow_future=False) if day else date.today()
        values = validate_fields(fields)
        for name in clear:
            if name not in ALL_FIELDS:
                raise ValidationError(f"Cannot clear unknown field '{name}'. Fields: {', '.join(ALL_FIELDS)}.")
            if name in values:
                raise ValidationError(f"Field '{name}' cannot be both set and cleared.")
            values[name] = None

        existing = self.repo.get(self.user_id, target.isoformat())
        if existing is None:
            if not any(val is not None for val in values.values()):
                raise ValidationError("Provide at least one value to create a daily record.")
            self.repo.create(self.user_id, target.isoformat(), values)
        else:
            if not values:
                raise ValidationError("Nothing to update. Provide at least one field to change.")
            self.repo.update(self.user_id, target.isoformat(), values)
        return self.get(target)

    def get(self, day: str | date) -> DailyRecord:
        target = v.parse_date(day, "date")
        row = self.repo.get(self.user_id, target.isoformat())
        if row is None:
            raise NotFoundError(f"No daily record for {target.isoformat()}.")
        photos = [Photo(int(p["id"]), p["file_path"], p["caption"]) for p in self.photos.for_record(row["id"])]
        return DailyRecord.from_row(row, photos)

    def find(self, day: date) -> DailyRecord | None:
        try:
            return self.get(day)
        except NotFoundError:
            return None

    def list(self, start: str | date | None = None, end: str | date | None = None) -> list[DailyRecord]:
        start_s = v.parse_date(start, "start date").isoformat() if start else None
        end_s = v.parse_date(end, "end date").isoformat() if end else None
        if start_s and end_s and start_s > end_s:
            raise ValidationError(f"Start date {start_s} is after end date {end_s}.")
        return [DailyRecord.from_row(r) for r in self.repo.list(self.user_id, start_s, end_s)]

    def delete(self, day: str | date) -> None:
        target = v.parse_date(day, "date")
        if not self.repo.delete(self.user_id, target.isoformat()):
            raise NotFoundError(f"No daily record for {target.isoformat()}.")

    def add_photo(self, day: str | date, source: str | Path, caption: str | None = None) -> Photo:
        if self.photos_dir is None:
            raise ValidationError("Photo storage directory is not configured.")
        record = self.get(day)
        caption = v.optional_text(caption, "caption", max_length=300)
        rel_path = store_photo(source, self.photos_dir, record.date)
        photo_id = self.photos.add(record.id, rel_path, caption)
        return Photo(photo_id, rel_path, caption)
