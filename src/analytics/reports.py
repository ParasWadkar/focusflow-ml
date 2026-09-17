"""Analytics over application data and trained models."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src.habits.models import HabitStats
from src.habits.service import HabitService
from src.ml.db_extract import extract_daily_frame
from src.ml.features import MODES
from src.ml.persistence import metadata_path
from src.planner.models import BlockStats
from src.planner.service import PlannerService
from src.utils import validators as v
from src.utils.dates import week_start
from src.utils.errors import ModelError, ValidationError

DEFAULT_WINDOW_DAYS = 30


@dataclass(frozen=True)
class Period:
    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1


def resolve_period(start: str | None, end: str | None, default_days: int = DEFAULT_WINDOW_DAYS) -> Period:
    end_d = v.parse_date(end, "end date") if end else date.today()
    start_d = v.parse_date(start, "start date") if start else end_d - timedelta(days=default_days - 1)
    if start_d > end_d:
        raise ValidationError(f"Start date {start_d} is after end date {end_d}.")
    return Period(start_d, end_d)


def daily_table(conn: sqlite3.Connection, user_id: int, period: Period) -> pd.DataFrame:
    """One row per logged day with derived ratios."""
    df = extract_daily_frame(conn, user_id, period.start, period.end)
    if df.empty:
        return df
    for col in df.columns.drop("date"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["habit_rate"] = df["habits_completed"] / df["habits_planned"].where(df["habits_planned"] > 0)
    df["task_completion"] = df["completed_task_hours"] / df["planned_task_hours"].where(df["planned_task_hours"] > 0)
    return df


def weekly_table(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["week_start"] = pd.to_datetime(df["date"]).dt.date.map(week_start)
    grouped = df.groupby("week_start")
    weekly = pd.DataFrame({
        "days_logged": grouped.size(),
        "avg_score": grouped["productivity_score"].mean(),
        "planned_hours": grouped["planned_task_hours"].sum(),
        "completed_hours": grouped["completed_task_hours"].sum(),
        "study_hours": grouped["study_hours"].sum(),
        "deep_work_hours": grouped["deep_work_hours"].sum(),
        "habits_completed": grouped["habits_completed"].sum(),
        "habits_planned": grouped["habits_planned"].sum(),
        "avg_sleep": grouped["sleep_hours"].mean(),
    }).reset_index()
    weekly["completion_ratio"] = weekly["completed_hours"] / weekly["planned_hours"].where(weekly["planned_hours"] > 0)
    weekly["habit_rate"] = weekly["habits_completed"] / weekly["habits_planned"].where(weekly["habits_planned"] > 0)
    return weekly


def overview(daily: pd.DataFrame) -> dict[str, float | int | None]:
    """Headline numbers for a period."""
    if daily.empty:
        return {"days_logged": 0}

    def mean(col: str) -> float | None:
        val = daily[col].mean()
        return None if pd.isna(val) else float(val)

    planned = float(daily["planned_task_hours"].sum())
    completed = float(daily["completed_task_hours"].sum())
    h_planned = float(daily["habits_planned"].sum())
    scored = daily.dropna(subset=["productivity_score"])
    best = scored.loc[scored["productivity_score"].idxmax()] if not scored.empty else None
    return {
        "days_logged": int(len(daily)),
        "days_scored": int(len(scored)),
        "avg_productivity_score": mean("productivity_score"),
        "best_day": None if best is None else f"{best['date']} ({best['productivity_score']:.0f})",
        "avg_sleep_hours": mean("sleep_hours"),
        "avg_energy_level": mean("energy_level"),
        "avg_mood_score": mean("mood_score"),
        "planned_hours": planned,
        "completed_hours": completed,
        "task_completion_ratio": (completed / planned) if planned else None,
        "study_hours": float(daily["study_hours"].sum()),
        "deep_work_hours": float(daily["deep_work_hours"].sum()),
        "avg_daily_deep_work_hours": mean("deep_work_hours"),
        "habit_completion_rate": (float(daily["habits_completed"].sum()) / h_planned) if h_planned else None,
        "total_interruptions": None if daily["interruptions"].isna().all() else int(daily["interruptions"].sum()),
    }


def habit_stats(conn: sqlite3.Connection, user_id: int, period: Period) -> list[HabitStats]:
    return HabitService(conn, user_id).all_stats(period.start, period.end)


def task_stats(conn: sqlite3.Connection, user_id: int, period: Period) -> BlockStats:
    return PlannerService(conn, user_id).stats(period.start, period.end)


def model_summary(models_dir: Path) -> dict[str, dict]:
    """Read saved metadata for each trained mode (no model loading required)."""
    summary: dict[str, dict] = {}
    for mode in MODES:
        path = metadata_path(models_dir, mode)
        if not path.exists():
            continue
        try:
            summary[mode] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelError(f"Metadata file {path} is not valid JSON; retrain the model.") from exc
    return summary
