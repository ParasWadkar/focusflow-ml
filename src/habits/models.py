"""Habit domain models."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

SCHEDULES = ("daily", "weekdays", "weekends")


@dataclass(frozen=True)
class Habit:
    id: int
    name: str
    description: str | None
    schedule: str
    start_date: date
    archived: bool

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Habit":
        return cls(
            id=int(row["id"]),
            name=row["name"],
            description=row["description"],
            schedule=row["schedule"],
            start_date=date.fromisoformat(row["start_date"]),
            archived=bool(row["archived"]),
        )

    def is_scheduled(self, day: date) -> bool:
        """Whether the habit is planned on ``day``."""
        if day < self.start_date:
            return False
        if self.schedule == "weekdays":
            return day.weekday() < 5
        if self.schedule == "weekends":
            return day.weekday() >= 5
        return True


@dataclass(frozen=True)
class HabitDay:
    """Status of one habit on one date."""

    date: date
    scheduled: bool
    completed: bool
    recorded: bool
    note: str | None = None


@dataclass(frozen=True)
class HabitStats:
    habit: Habit
    start: date
    end: date
    scheduled_days: int
    completed_days: int
    completion_rate: float | None
    current_streak: int
    longest_streak: int
