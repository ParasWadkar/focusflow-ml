"""Habit business logic: validation, completion tracking and statistics."""

from __future__ import annotations

import sqlite3
from datetime import date

from src.database.repositories import HabitRecordRepository, HabitRepository
from src.habits import streaks
from src.habits.models import SCHEDULES, Habit, HabitDay, HabitStats
from src.utils import validators as v
from src.utils.dates import date_range
from src.utils.errors import NotFoundError, ValidationError


class HabitService:
    def __init__(self, conn: sqlite3.Connection, user_id: int) -> None:
        self.user_id = user_id
        self.habits = HabitRepository(conn)
        self.records = HabitRecordRepository(conn)

    # ----- CRUD -----------------------------------------------------------
    def add(self, name: str, description: str | None = None, schedule: str = "daily",
            start_date: str | date | None = None) -> Habit:
        name = v.require_text(name, "Habit name", max_length=80)
        description = v.optional_text(description, "Description", max_length=500)
        schedule = v.choice(schedule, "schedule", SCHEDULES)
        start = v.parse_date(start_date, "start date", allow_future=False) if start_date else date.today()
        habit_id = self.habits.create(self.user_id, name, description, schedule, start.isoformat())
        return self.get(habit_id)

    def get(self, ref: int | str) -> Habit:
        """Look up a habit by numeric id or by (case-insensitive) name."""
        row = None
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
            row = self.habits.get(self.user_id, int(ref))
        if row is None and isinstance(ref, str):
            row = self.habits.get_by_name(self.user_id, ref.strip())
        if row is None:
            raise NotFoundError(f"No habit found with id or name '{ref}'.")
        return Habit.from_row(row)

    def list(self, include_archived: bool = False) -> list[Habit]:
        return [Habit.from_row(r) for r in self.habits.list(self.user_id, include_archived)]

    def edit(self, ref: int | str, *, name: str | None = None, description: str | None = None,
             schedule: str | None = None, start_date: str | date | None = None,
             archived: bool | None = None) -> Habit:
        habit = self.get(ref)
        fields: dict[str, object] = {}
        if name is not None:
            fields["name"] = v.require_text(name, "Habit name", max_length=80)
        if description is not None:
            fields["description"] = v.optional_text(description, "Description", max_length=500)
        if schedule is not None:
            fields["schedule"] = v.choice(schedule, "schedule", SCHEDULES)
        if start_date is not None:
            fields["start_date"] = v.parse_date(start_date, "start date", allow_future=False).isoformat()
        if archived is not None:
            fields["archived"] = int(archived)
        if not fields:
            raise ValidationError("Nothing to update. Provide at least one field to change.")
        self.habits.update(self.user_id, habit.id, fields)
        return self.get(habit.id)

    def delete(self, ref: int | str) -> Habit:
        habit = self.get(ref)
        self.habits.delete(self.user_id, habit.id)
        return habit

    # ----- completion -----------------------------------------------------
    def mark(self, ref: int | str, day: str | date | None = None, completed: bool = True,
             note: str | None = None) -> HabitDay:
        """Record completion (or explicit non-completion) for a date. Past dates are allowed."""
        habit = self.get(ref)
        target = v.parse_date(day, "date", allow_future=False) if day else date.today()
        if target < habit.start_date:
            raise ValidationError(
                f"{target.isoformat()} is before '{habit.name}' started ({habit.start_date.isoformat()}). "
                "Use `habit edit --start-date` to backfill earlier history."
            )
        note = v.optional_text(note, "Note", max_length=500)
        self.records.upsert(habit.id, target.isoformat(), completed, note)
        return HabitDay(target, habit.is_scheduled(target), completed, True, note)

    def clear(self, ref: int | str, day: str | date | None = None) -> bool:
        """Remove the record for a date, returning it to 'unrecorded'."""
        habit = self.get(ref)
        target = v.parse_date(day, "date") if day else date.today()
        return self.records.delete(habit.id, target.isoformat())

    def history(self, ref: int | str, start: str | date | None = None,
                end: str | date | None = None) -> list[HabitDay]:
        habit = self.get(ref)
        start_d, end_d = self._window(habit, start, end)
        recs = {r["date"]: r for r in self.records.for_habit(habit.id, start_d.isoformat(), end_d.isoformat())}
        days = []
        for day in date_range(start_d, end_d):
            rec = recs.get(day.isoformat())
            days.append(HabitDay(
                date=day,
                scheduled=habit.is_scheduled(day),
                completed=bool(rec and rec["completed"]),
                recorded=rec is not None,
                note=rec["note"] if rec else None,
            ))
        return days

    def stats(self, ref: int | str, start: str | date | None = None,
              end: str | date | None = None) -> HabitStats:
        habit = self.get(ref)
        start_d, end_d = self._window(habit, start, end)
        return self._stats_for(habit, start_d, end_d)

    def all_stats(self, start: str | date | None = None, end: str | date | None = None) -> list[HabitStats]:
        results = []
        for habit in self.list():
            start_d, end_d = self._window(habit, start, end)
            results.append(self._stats_for(habit, start_d, end_d))
        return results

    def day_summary(self, day: date) -> tuple[int, int | None]:
        """(habits scheduled, scheduled habits completed) for one date.

        The completed count is ``None`` when nothing at all was recorded for
        that date, so an unlogged day is "unknown" rather than "0 completed".
        """
        records = self.records.for_user(self.user_id, day.isoformat(), day.isoformat())
        done = {r["habit_id"] for r in records if r["completed"]}
        planned = completed = 0
        for habit in self.list(include_archived=True):
            if habit.is_scheduled(day) and (not habit.archived or habit.id in done):
                planned += 1
                completed += habit.id in done
        return planned, (completed if records else None)

    # ----- helpers --------------------------------------------------------
    def _stats_for(self, habit: Habit, start_d: date, end_d: date) -> HabitStats:
        history = self.history(habit.id, start_d, end_d) if start_d <= end_d else []
        scheduled = [d.date for d in history if d.scheduled]
        completed = {d.date for d in history if d.completed}
        done_scheduled = sum(1 for d in scheduled if d in completed)
        # Streaks consider the full history so a window does not truncate them.
        full = self.history(habit.id, habit.start_date, end_d) if habit.start_date <= end_d else []
        full_sched = [d.date for d in full if d.scheduled]
        full_done = {d.date for d in full if d.completed}
        return HabitStats(
            habit=habit,
            start=start_d,
            end=end_d,
            scheduled_days=len(scheduled),
            completed_days=done_scheduled,
            completion_rate=(done_scheduled / len(scheduled)) if scheduled else None,
            current_streak=streaks.current_streak(full_sched, full_done, end_d),
            longest_streak=streaks.longest_streak(full_sched, full_done),
        )

    @staticmethod
    def _window(habit: Habit, start: str | date | None, end: str | date | None) -> tuple[date, date]:
        end_d = v.parse_date(end, "end date") if end else date.today()
        start_d = v.parse_date(start, "start date") if start else habit.start_date
        if start_d > end_d:
            raise ValidationError(f"Start date {start_d} is after end date {end_d}.")
        return max(start_d, habit.start_date), end_d
