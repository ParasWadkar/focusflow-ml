"""Small, reusable input validators.

Each validator returns the normalised value or raises
:class:`~src.utils.errors.ValidationError` with a message naming the field.
"""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Iterable

from src.utils.errors import ValidationError

DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%H:%M"


def parse_date(value: str | date | None, field: str = "date", *, allow_future: bool = True) -> date:
    """Parse a ``YYYY-MM-DD`` string (or pass through a ``date``)."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{field} is required (format YYYY-MM-DD).")
    if isinstance(value, datetime):
        parsed = value.date()
    elif isinstance(value, date):
        parsed = value
    else:
        try:
            parsed = datetime.strptime(value.strip(), DATE_FORMAT).date()
        except ValueError:
            raise ValidationError(
                f"Invalid {field} '{value}'. Expected a real calendar date in YYYY-MM-DD format."
            ) from None
    if not allow_future and parsed > date.today():
        raise ValidationError(f"{field} {parsed.isoformat()} is in the future.")
    return parsed


def parse_time(value: str | time | None, field: str = "time") -> time:
    """Parse an ``HH:MM`` 24-hour time string."""
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ValidationError(f"{field} is required (format HH:MM).")
    if isinstance(value, time):
        return value.replace(second=0, microsecond=0)
    try:
        return datetime.strptime(value.strip(), TIME_FORMAT).time()
    except ValueError:
        raise ValidationError(f"Invalid {field} '{value}'. Expected 24-hour HH:MM, e.g. 09:30.") from None


def require_text(value: str | None, field: str, max_length: int = 200) -> str:
    """Require a non-empty string no longer than ``max_length``."""
    if value is None or not str(value).strip():
        raise ValidationError(f"{field} is required and cannot be empty.")
    text = str(value).strip()
    if len(text) > max_length:
        raise ValidationError(f"{field} must be at most {max_length} characters (got {len(text)}).")
    return text


def optional_text(value: str | None, field: str, max_length: int = 10_000) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if len(text) > max_length:
        raise ValidationError(f"{field} must be at most {max_length} characters (got {len(text)}).")
    return text or None


def _to_number(value: object, field: str, kind: type) -> float | int:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be a number, not a boolean.")
    try:
        number = kind(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a{'n integer' if kind is int else ' number'} (got {value!r}).") from None
    if kind is int and isinstance(value, float) and not value.is_integer():
        raise ValidationError(f"{field} must be a whole number (got {value!r}).")
    if kind is float and number != number:  # NaN
        raise ValidationError(f"{field} must be a number (got NaN).")
    return number


def number_in_range(value: object, field: str, minimum: float | None = None,
                    maximum: float | None = None) -> float:
    """Validate a float within an inclusive range."""
    number = float(_to_number(value, field, float))
    _check_range(number, field, minimum, maximum)
    return number


def int_in_range(value: object, field: str, minimum: int | None = None,
                 maximum: int | None = None) -> int:
    """Validate an integer within an inclusive range."""
    number = int(_to_number(value, field, int))
    _check_range(number, field, minimum, maximum)
    return number


def _check_range(number: float, field: str, minimum: float | None, maximum: float | None) -> None:
    if minimum is not None and number < minimum:
        if minimum == 0:
            raise ValidationError(f"{field} cannot be negative (got {number:g}).")
        raise ValidationError(f"{field} must be at least {minimum:g} (got {number:g}).")
    if maximum is not None and number > maximum:
        raise ValidationError(f"{field} must be at most {maximum:g} (got {number:g}).")


def choice(value: str | None, field: str, options: Iterable[str]) -> str:
    """Validate that ``value`` is one of ``options`` (case-insensitive)."""
    allowed = list(options)
    if value is None:
        raise ValidationError(f"{field} is required. Choose one of: {', '.join(allowed)}.")
    normalised = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalised not in allowed:
        raise ValidationError(f"Invalid {field} '{value}'. Choose one of: {', '.join(allowed)}.")
    return normalised
