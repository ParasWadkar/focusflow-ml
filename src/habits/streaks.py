"""Pure streak calculations.

A streak is a run of consecutive *scheduled* days on which the habit was
completed. Unscheduled days (e.g. weekends for a weekday habit) neither
extend nor break a streak.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class StreakRun:
    start: date
    end: date
    length: int


def streak_runs(scheduled_days: Sequence[date], completed: set[date]) -> list[StreakRun]:
    """All maximal completed runs over the (sorted) scheduled days."""
    runs: list[StreakRun] = []
    run_start: date | None = None
    run_end: date | None = None
    length = 0
    for day in sorted(scheduled_days):
        if day in completed:
            if run_start is None:
                run_start = day
            run_end = day
            length += 1
        elif run_start is not None:
            runs.append(StreakRun(run_start, run_end, length))  # type: ignore[arg-type]
            run_start, run_end, length = None, None, 0
    if run_start is not None:
        runs.append(StreakRun(run_start, run_end, length))  # type: ignore[arg-type]
    return runs


def longest_streak(scheduled_days: Sequence[date], completed: set[date]) -> int:
    return max((r.length for r in streak_runs(scheduled_days, completed)), default=0)


def current_streak(scheduled_days: Sequence[date], completed: set[date], as_of: date) -> int:
    """Length of the streak ending at ``as_of``.

    If ``as_of`` itself is scheduled but not yet completed, it is treated as
    still in progress, so it does not break a streak that ended the previous
    scheduled day.
    """
    days = sorted(d for d in scheduled_days if d <= as_of)
    if days and days[-1] == as_of and as_of not in completed:
        days.pop()
    count = 0
    for day in reversed(days):
        if day not in completed:
            break
        count += 1
    return count
