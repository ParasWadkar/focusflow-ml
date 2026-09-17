"""Plain-text output helpers for the CLI."""

from __future__ import annotations

import math
from typing import Any, Sequence


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    """Render rows as a fixed-width text table."""
    cells = [[_cell(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in cells:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(v.ljust(widths[i]) for i, v in enumerate(row)) for row in cells]
    return "\n".join([line, sep, *body])


def format_kv(pairs: Sequence[tuple[str, Any]]) -> str:
    """Render key/value pairs aligned on the key column."""
    if not pairs:
        return ""
    width = max(len(k) for k, _ in pairs)
    return "\n".join(f"{k.ljust(width)} : {_cell(v)}" for k, v in pairs)


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return "-" if math.isnan(value) else f"{value:.2f}"
    return str(value)
