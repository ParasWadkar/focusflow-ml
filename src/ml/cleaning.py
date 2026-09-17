"""Deterministic cleaning of a validated raw daily table.

Cleaning never looks at the data distribution, so it cannot leak test-set
information. Median imputation happens later, inside the model pipeline, and
is fitted on training rows only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.features import DATE_COLUMN, RAW_COLUMNS, RAW_NUMERIC_SPEC, TARGET
from src.ml.validation import CONSISTENCY_PAIRS


@dataclass(frozen=True)
class CleaningSummary:
    rows_in: int
    rows_out: int
    dropped_missing_target: int
    values_nulled_out_of_range: int
    values_capped: int


def clean(df: pd.DataFrame, require_target: bool = True) -> tuple[pd.DataFrame, CleaningSummary]:
    """Coerce types, null out-of-range values, cap inconsistencies, drop unlabeled rows."""
    out = df[RAW_COLUMNS].copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], format="%Y-%m-%d")

    nulled = 0
    for col, (lo, hi) in RAW_NUMERIC_SPEC.items():
        values = pd.to_numeric(out[col], errors="coerce").astype(float)
        bad = (values < lo) | (values > hi)
        nulled += int(bad.sum())
        out[col] = values.mask(bad, np.nan)

    capped = 0
    for done, planned in CONSISTENCY_PAIRS:
        over = out[done] > out[planned]
        capped += int(over.sum())
        out.loc[over, done] = out.loc[over, planned]

    dropped = 0
    if require_target:
        has_target = out[TARGET].notna()
        dropped = int((~has_target).sum())
        out = out[has_target]

    out = out.sort_values(DATE_COLUMN).reset_index(drop=True)
    return out, CleaningSummary(
        rows_in=len(df),
        rows_out=len(out),
        dropped_missing_target=dropped,
        values_nulled_out_of_range=nulled,
        values_capped=capped,
    )
