"""Time-block domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from typing import Any

from src.utils.dates import minutes_between

CATEGORIES = ("study", "deep_work", "work", "exercise", "admin", "personal", "other")
PRIORITIES = {1: "high", 2: "medium", 3: "low"}


@dataclass(frozen=True)
class TimeBlock:
    id: int
    date: date
    start_time: time
    end_time: time
    task_name: str
    category: str
    priority: int
    completed: bool

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TimeBlock":
        return cls(
            id=int(row["id"]),
            date=date.fromisoformat(row["date"]),
            start_time=time.fromisoformat(row["start_time"]),
            end_time=time.fromisoformat(row["end_time"]),
            task_name=row["task_name"],
            category=row["category"],
            priority=int(row["priority"]),
            completed=bool(row["completed"]),
        )

    @property
    def duration_hours(self) -> float:
        return minutes_between(self.start_time, self.end_time) / 60.0

    def overlaps(self, start: time, end: time) -> bool:
        return self.start_time < end and start < self.end_time


@dataclass(frozen=True)
class BlockStats:
    block_count: int
    completed_count: int
    planned_hours: float
    completed_hours: float
    completion_ratio: float | None
    planned_study_hours: float
    study_hours: float
    planned_deep_work_hours: float
    deep_work_hours: float
    hours_by_category: dict[str, float] = field(default_factory=dict)
    completed_hours_by_category: dict[str, float] = field(default_factory=dict)
