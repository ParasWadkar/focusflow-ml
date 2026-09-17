"""Build the raw daily table (same schema as the seed CSV) from application data."""

from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

from src.habits.service import HabitService
from src.journal.service import JournalService
from src.ml.features import DATE_COLUMN, RAW_COLUMNS
from src.planner.service import PlannerService
from src.planner.stats import compute_stats
from src.utils.dates import date_range


def daily_row(day: date, habits: HabitService, planner: PlannerService,
              journal: JournalService) -> dict[str, object]:
    """Raw values for one date. Missing information is ``None``."""
    record = journal.find(day)
    block_stats = compute_stats(planner.for_day(day))
    planned, completed = habits.day_summary(day)
    return {
        DATE_COLUMN: day.isoformat(),
        "sleep_hours": record.sleep_hours if record else None,
        "energy_level": record.energy_level if record else None,
        "mood_score": record.mood_score if record else None,
        "exercise_minutes": record.exercise_minutes if record else None,
        "interruptions": record.interruptions if record else None,
        "habits_planned": planned,
        "habits_completed": completed,
        "planned_task_hours": block_stats.planned_hours,
        "completed_task_hours": block_stats.completed_hours,
        "planned_study_hours": block_stats.planned_study_hours,
        "study_hours": block_stats.study_hours,
        "planned_deep_work_hours": block_stats.planned_deep_work_hours,
        "deep_work_hours": block_stats.deep_work_hours,
        "productivity_score": record.productivity_score if record else None,
    }


def extract_daily_frame(conn: sqlite3.Connection, user_id: int, start: date | None = None,
                        end: date | None = None) -> pd.DataFrame:
    """One row per date that has a daily record or at least one time block."""
    habits = HabitService(conn, user_id)
    planner = PlannerService(conn, user_id)
    journal = JournalService(conn, user_id)

    active_dates: set[date] = {r.date for r in journal.list(start, end)}
    active_dates |= {b.date for b in planner.list(start, end)}
    if not active_dates:
        return pd.DataFrame(columns=RAW_COLUMNS)
    lo, hi = min(active_dates), max(active_dates)
    rows = [daily_row(d, habits, planner, journal) for d in date_range(lo, hi) if d in active_dates]
    return pd.DataFrame(rows, columns=RAW_COLUMNS)
