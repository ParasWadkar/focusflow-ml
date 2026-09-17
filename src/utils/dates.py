"""Date helpers."""

from __future__ import annotations

from datetime import date, time, timedelta
from typing import Iterator


def date_range(start: date, end: date) -> Iterator[date]:
    """Yield every date from ``start`` to ``end`` inclusive."""
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def week_start(day: date) -> date:
    """Monday of the ISO week containing ``day``."""
    return day - timedelta(days=day.weekday())


def is_weekend(day: date) -> bool:
    return day.weekday() >= 5


def minutes_between(start: time, end: time) -> int:
    """Minutes from ``start`` to ``end`` on the same day."""
    return (end.hour * 60 + end.minute) - (start.hour * 60 + start.minute)
