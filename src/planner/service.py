"""Time-blocking business logic."""

from __future__ import annotations

import sqlite3
from datetime import date, time

from src.database.repositories import TimeBlockRepository
from src.planner.models import CATEGORIES, BlockStats, TimeBlock
from src.planner.stats import compute_stats
from src.utils import validators as v
from src.utils.errors import NotFoundError, ValidationError


class PlannerService:
    def __init__(self, conn: sqlite3.Connection, user_id: int) -> None:
        self.user_id = user_id
        self.repo = TimeBlockRepository(conn)

    def add(self, day: str | date, start: str | time, end: str | time, task_name: str,
            category: str = "other", priority: int | str = 2, completed: bool = False) -> TimeBlock:
        values = self._validate(day, start, end, task_name, category, priority)
        self._check_overlap(values["date"], values["start_time"], values["end_time"])
        values["completed"] = completed
        block_id = self.repo.create(self.user_id, values)
        return self.get(block_id)

    def get(self, block_id: int | str) -> TimeBlock:
        block_id = v.int_in_range(block_id, "block id", minimum=1)
        row = self.repo.get(self.user_id, block_id)
        if row is None:
            raise NotFoundError(f"No time block with id {block_id}.")
        return TimeBlock.from_row(row)

    def list(self, start: str | date | None = None, end: str | date | None = None) -> list[TimeBlock]:
        start_s = v.parse_date(start, "start date").isoformat() if start else None
        end_s = v.parse_date(end, "end date").isoformat() if end else None
        if start_s and end_s and start_s > end_s:
            raise ValidationError(f"Start date {start_s} is after end date {end_s}.")
        return [TimeBlock.from_row(r) for r in self.repo.list(self.user_id, start_s, end_s)]

    def for_day(self, day: str | date) -> list[TimeBlock]:
        d = v.parse_date(day, "date")
        return self.list(d, d)

    def edit(self, block_id: int | str, *, day: str | None = None, start: str | None = None,
             end: str | None = None, task_name: str | None = None, category: str | None = None,
             priority: int | str | None = None) -> TimeBlock:
        current = self.get(block_id)
        if all(x is None for x in (day, start, end, task_name, category, priority)):
            raise ValidationError("Nothing to update. Provide at least one field to change.")
        values = self._validate(
            day if day is not None else current.date,
            start if start is not None else current.start_time,
            end if end is not None else current.end_time,
            task_name if task_name is not None else current.task_name,
            category if category is not None else current.category,
            priority if priority is not None else current.priority,
        )
        self._check_overlap(values["date"], values["start_time"], values["end_time"], ignore_id=current.id)
        self.repo.update(self.user_id, current.id, values)
        return self.get(current.id)

    def set_completed(self, block_id: int | str, completed: bool = True) -> TimeBlock:
        block = self.get(block_id)
        self.repo.update(self.user_id, block.id, {"completed": int(completed)})
        return self.get(block.id)

    def delete(self, block_id: int | str) -> TimeBlock:
        block = self.get(block_id)
        self.repo.delete(self.user_id, block.id)
        return block

    def stats(self, start: str | date | None = None, end: str | date | None = None) -> BlockStats:
        return compute_stats(self.list(start, end))

    # ----- helpers --------------------------------------------------------
    @staticmethod
    def _validate(day, start, end, task_name, category, priority) -> dict:
        d = v.parse_date(day, "date")
        s = v.parse_time(start, "start time")
        e = v.parse_time(end, "end time")
        if e <= s:
            raise ValidationError(
                f"End time {e.strftime('%H:%M')} must be after start time {s.strftime('%H:%M')}. "
                "Blocks cannot span midnight; split them into two blocks."
            )
        return {
            "date": d.isoformat(),
            "start_time": s.strftime("%H:%M"),
            "end_time": e.strftime("%H:%M"),
            "task_name": v.require_text(task_name, "Task name", max_length=120),
            "category": v.choice(category, "category", CATEGORIES),
            "priority": v.int_in_range(priority, "priority", 1, 3),
        }

    def _check_overlap(self, day: str, start: str, end: str, ignore_id: int | None = None) -> None:
        s, e = time.fromisoformat(start), time.fromisoformat(end)
        for block in self.list(day, day):
            if block.id != ignore_id and block.overlaps(s, e):
                raise ValidationError(
                    f"Block overlaps existing block #{block.id} '{block.task_name}' "
                    f"({block.start_time:%H:%M}-{block.end_time:%H:%M}) on {day}."
                )
