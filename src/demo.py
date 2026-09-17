"""Opt-in demo data for an empty database (``init --demo-data``).

The demo data is SYNTHETIC. It uses the same generator as the seed dataset,
written through the normal services, so analytics and database-driven
predictions can be tried immediately. Each demo journal entry is marked
``[demo]``.

Past days get habits, completed and missed time blocks, and a full daily
record with a score. Today gets only what would be known in the morning:
sleep, energy, planned blocks and scheduled habits.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.habits.service import HabitService
from src.journal.service import JournalService
from src.ml.dataset import generate_dataset, productivity_score
from src.ml.db_extract import extract_daily_frame
from src.planner.service import PlannerService
from src.utils.errors import ValidationError

DEMO_HABITS = [
    ("Morning walk", "daily"),
    ("Read 20 pages", "daily"),
    ("Meditate 10 minutes", "daily"),
    ("Plan tomorrow", "daily"),
    ("Review flashcards", "weekdays"),
    ("No phone before 9am", "weekdays"),
]
DEMO_TAG = "[demo]"
MAX_BLOCK_HOURS = 2.0
DAY_START = datetime(2000, 1, 1, 8, 0)


@dataclass(frozen=True)
class DemoSummary:
    days: int
    habits: int
    blocks: int
    start: date
    end: date


def _has_data(conn: sqlite3.Connection, user_id: int) -> bool:
    for table in ("habits", "time_blocks", "daily_records"):
        if conn.execute(f"SELECT 1 FROM {table} WHERE user_id = ? LIMIT 1", (user_id,)).fetchone():
            return True
    return False


def _chunks(hours: float) -> list[float]:
    pieces = []
    while hours > 1e-9:
        piece = min(MAX_BLOCK_HOURS, hours)
        pieces.append(piece)
        hours -= piece
    return pieces


def _add_blocks(planner: PlannerService, day: date, row: pd.Series, execute: bool) -> int:
    """Lay out the day's planned hours as consecutive blocks from 08:00 and mark completions."""
    plan = [
        ("study", "Study session", row["planned_study_hours"], row["study_hours"]),
        ("deep_work", "Deep work", row["planned_deep_work_hours"], row["deep_work_hours"]),
        ("work", "Project work",
         row["planned_task_hours"] - row["planned_study_hours"] - row["planned_deep_work_hours"],
         row["completed_task_hours"] - row["study_hours"] - row["deep_work_hours"]),
    ]
    cursor = DAY_START
    count = 0
    for category, label, planned, done in plan:
        remaining_done = max(0.0, float(done))
        for i, hours in enumerate(_chunks(max(0.0, float(planned)))):
            end = cursor + timedelta(hours=hours)
            if end.day != cursor.day or end.time() > datetime.strptime("23:45", "%H:%M").time():
                return count
            block = planner.add(day, cursor.strftime("%H:%M"), end.strftime("%H:%M"),
                                f"{label} {i + 1}", category, priority=1 if category == "deep_work" else 2)
            count += 1
            # A block counts as done when at least half of it fits in the completed time.
            if execute and remaining_done >= hours / 2 - 1e-9:
                planner.set_completed(block.id)
                remaining_done = max(0.0, remaining_done - hours)
            cursor = end + timedelta(minutes=15)
    return count


def seed_demo(conn: sqlite3.Connection, user_id: int, days: int = 60, seed: int = 7) -> DemoSummary:
    if _has_data(conn, user_id):
        raise ValidationError("Demo data can only be added to an empty database.")
    if not 14 <= days <= 365:
        raise ValidationError("Demo days must be between 14 and 365.")

    today = date.today()
    start = today - timedelta(days=days)
    raw, form = generate_dataset(rows=days + 1, seed=seed, start=start, missing=False, return_form=True)
    rng = np.random.default_rng(seed + 1)

    habits = HabitService(conn, user_id)
    planner = PlannerService(conn, user_id)
    journal = JournalService(conn, user_id)
    created = [habits.add(name, schedule=schedule, start_date=start) for name, schedule in DEMO_HABITS]

    blocks = 0
    for idx, row in raw.iterrows():
        day = date.fromisoformat(row["date"])
        is_today = day == today
        blocks += _add_blocks(planner, day, row, execute=not is_today)
        if is_today:
            journal.record(day, sleep_hours=row["sleep_hours"], energy_level=int(row["energy_level"]))
            continue
        scheduled = [h for h in created if h.is_scheduled(day)]
        rate = row["habits_completed"] / row["habits_planned"] if row["habits_planned"] else 0.0
        n_done = int(round(rate * len(scheduled)))
        for pos in rng.permutation(len(scheduled)):
            habits.mark(scheduled[pos].id, day, completed=n_done > 0)
            n_done -= 1
        journal.record(
            day,
            sleep_hours=row["sleep_hours"],
            energy_level=int(row["energy_level"]),
            mood_score=int(row["mood_score"]),
            exercise_minutes=row["exercise_minutes"],
            interruptions=int(row["interruptions"]),
            journal=f"{DEMO_TAG} Synthetic demo entry.",
        )

    # Score each past day from what was actually stored, using the generator's scoring function.
    stored = extract_daily_frame(conn, user_id, start, today - timedelta(days=1))
    for col in stored.columns.drop("date"):
        stored[col] = pd.to_numeric(stored[col], errors="coerce").astype(float)
    form_by_date = dict(zip(raw["date"], form))
    forms = np.array([form_by_date[d] for d in stored["date"]])
    scores = productivity_score(stored, forms, rng.normal(0, 4.0, len(stored)))
    for day_str, score in zip(stored["date"], scores):
        journal.record(day_str, productivity_score=float(score))

    return DemoSummary(days=days, habits=len(created), blocks=blocks, start=start, end=today)
