"""Daily record domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

# field -> (validator kind, minimum, maximum)
NUMERIC_FIELDS: dict[str, tuple[type, float, float]] = {
    "sleep_hours": (float, 0, 24),
    "energy_level": (int, 1, 10),
    "mood_score": (int, 1, 10),
    "exercise_minutes": (float, 0, 1440),
    "interruptions": (int, 0, 500),
    "productivity_score": (float, 0, 100),
}
TEXT_FIELDS = ("journal",)
ALL_FIELDS = (*NUMERIC_FIELDS, *TEXT_FIELDS)


@dataclass(frozen=True)
class Photo:
    id: int
    file_path: str
    caption: str | None


@dataclass(frozen=True)
class DailyRecord:
    id: int
    date: date
    sleep_hours: float | None
    energy_level: int | None
    mood_score: int | None
    exercise_minutes: float | None
    interruptions: int | None
    journal: str | None
    productivity_score: float | None
    photos: list[Photo] = field(default_factory=list)

    @classmethod
    def from_row(cls, row: dict[str, Any], photos: list[Photo] | None = None) -> "DailyRecord":
        return cls(
            id=int(row["id"]),
            date=date.fromisoformat(row["date"]),
            sleep_hours=row["sleep_hours"],
            energy_level=row["energy_level"],
            mood_score=row["mood_score"],
            exercise_minutes=row["exercise_minutes"],
            interruptions=row["interruptions"],
            journal=row["journal"],
            productivity_score=row["productivity_score"],
            photos=photos or [],
        )
