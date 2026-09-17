"""Interactive fallbacks for missing arguments.

Prompts are used only when stdin is an interactive terminal. In scripts and
pipes, a missing value is a clear error.
"""

from __future__ import annotations

import sys

from src.utils.errors import ValidationError


def interactive() -> bool:
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def require(value: str | None, flag: str, prompt: str) -> str:
    if value not in (None, ""):
        return value  # type: ignore[return-value]
    if interactive():
        try:
            answer = input(f"{prompt}: ").strip()
        except EOFError:
            answer = ""
        if answer:
            return answer
    raise ValidationError(f"{flag} is required.")


def confirm(question: str, assume_yes: bool) -> None:
    """Ask for confirmation of a destructive action."""
    if assume_yes:
        return
    if not interactive():
        raise ValidationError("This action needs confirmation; re-run with --yes.")
    try:
        answer = input(f"{question} [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        raise ValidationError("Cancelled.")
