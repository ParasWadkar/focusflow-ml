"""Dataset validation.

Structural problems that make a dataset unusable (missing columns, unparseable
or duplicate dates, non-numeric values, too few rows) raise
:class:`DataError`. Recoverable issues (out-of-range values, missing values,
completed > planned) are collected as warnings and handled in cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.ml.features import DATE_COLUMN, RAW_COLUMNS, RAW_NUMERIC_SPEC, TARGET
from src.utils.errors import DataError

CONSISTENCY_PAIRS = [
    ("habits_completed", "habits_planned"),
    ("completed_task_hours", "planned_task_hours"),
    ("study_hours", "planned_study_hours"),
    ("deep_work_hours", "planned_deep_work_hours"),
]


@dataclass
class ValidationReport:
    rows: int
    missing_values: dict[str, int] = field(default_factory=dict)
    out_of_range: dict[str, int] = field(default_factory=dict)
    inconsistent: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings


def _preview(values: pd.Index | list, limit: int = 5) -> str:
    items = [str(v) for v in list(values)[:limit]]
    more = len(values) - limit
    return ", ".join(items) + (f" (+{more} more)" if more > 0 else "")


def validate_raw(df: pd.DataFrame, min_rows: int = 1) -> ValidationReport:
    """Validate a raw daily table. Raises on fatal problems, returns a report otherwise."""
    if df is None or df.empty:
        raise DataError("Dataset is empty.")

    missing_cols = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataError(f"Dataset is missing required column(s): {', '.join(missing_cols)}.")

    dates = pd.to_datetime(df[DATE_COLUMN], format="%Y-%m-%d", errors="coerce")
    bad_dates = df.index[dates.isna()]
    if len(bad_dates):
        raise DataError(
            f"{len(bad_dates)} row(s) have missing or invalid dates (expected YYYY-MM-DD), "
            f"e.g. rows {_preview(bad_dates)}."
        )
    dupes = dates[dates.duplicated()].dt.strftime("%Y-%m-%d").unique()
    if len(dupes):
        raise DataError(f"Dataset has duplicate dates: {_preview(list(dupes))}.")

    report = ValidationReport(rows=len(df))
    for col, (lo, hi) in RAW_NUMERIC_SPEC.items():
        original_na = df[col].isna()
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_numeric = numeric.isna() & ~original_na
        if non_numeric.any():
            raise DataError(
                f"Column '{col}' has {int(non_numeric.sum())} non-numeric value(s), "
                f"e.g. {_preview(df.loc[non_numeric, col].tolist())}."
            )
        n_missing = int(original_na.sum())
        if n_missing:
            report.missing_values[col] = n_missing
        n_bad = int(((numeric < lo) | (numeric > hi)).sum())
        if n_bad:
            report.out_of_range[col] = n_bad
            report.warnings.append(f"{n_bad} value(s) in '{col}' outside [{lo:g}, {hi:g}] will be treated as missing.")

    for done, planned in CONSISTENCY_PAIRS:
        n = int((pd.to_numeric(df[done], errors="coerce") > pd.to_numeric(df[planned], errors="coerce")).sum())
        if n:
            report.inconsistent[done] = n
            report.warnings.append(f"{n} row(s) where '{done}' exceeds '{planned}' will be capped.")

    usable = int(pd.to_numeric(df[TARGET], errors="coerce").between(0, 100).sum())
    if usable < min_rows:
        raise DataError(
            f"Only {usable} row(s) have a valid {TARGET}; at least {min_rows} are required."
        )
    if report.missing_values:
        total = sum(report.missing_values.values())
        report.warnings.append(f"{total} missing value(s) across {len(report.missing_values)} column(s).")
    return report
