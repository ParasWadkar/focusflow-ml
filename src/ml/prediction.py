"""Single-day prediction.

Prediction reuses :func:`engineer_features` on the target day plus its
recent history, so training and inference compute features identically.
The target day's own productivity score is always hidden from the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.ml.features import (DATE_COLUMN, POST_DAY_FEATURES, RAW_COLUMNS, RAW_NUMERIC_SPEC,
                             REQUIRED_PREDICTION_INPUTS, TARGET, build_matrix, engineer_features)
from src.ml.persistence import ModelBundle
from src.utils.errors import DataError, ValidationError

HISTORY_DAYS = 7

# Raw input -> CLI flag that supplies it (used in error messages).
INPUT_FLAGS: dict[str, str] = {
    "sleep_hours": "--sleep",
    "energy_level": "--energy",
    "habits_planned": "--habits-planned",
    "planned_task_hours": "--planned-hours",
    "planned_study_hours": "--planned-study",
    "planned_deep_work_hours": "--planned-deep",
    "habits_completed": "--habits-completed",
    "completed_task_hours": "--completed-hours",
    "study_hours": "--study-hours",
    "deep_work_hours": "--deep-hours",
    "mood_score": "--mood",
    "interruptions": "--interruptions",
    "exercise_minutes": "--exercise",
    "prev_productivity_score": "--prev-score",
}
LAG_FEATURES = ("prev_productivity_score", "prev_habit_completion_rate", "rolling_7d_productivity")


@dataclass
class PredictionResult:
    date: date
    mode: str
    score: float
    model: str
    test_mae: float | None
    inputs: dict[str, Any]
    imputed: list[str] = field(default_factory=list)


def _validate_overrides(overrides: dict[str, Any]) -> dict[str, float]:
    clean: dict[str, float] = {}
    for name, value in overrides.items():
        if value is None:
            continue
        if name not in INPUT_FLAGS:
            raise ValidationError(f"Unknown prediction input '{name}'.")
        lo, hi = RAW_NUMERIC_SPEC.get(name, (0, 100))
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValidationError(f"{INPUT_FLAGS[name]} must be a number (got {value!r}).") from None
        if not lo <= number <= hi:
            raise ValidationError(f"{INPUT_FLAGS[name]} must be between {lo:g} and {hi:g} (got {number:g}).")
        clean[name] = number
    return clean


def build_feature_row(day: date, mode: str, today_raw: dict[str, Any] | None,
                      history: pd.DataFrame | None, overrides: dict[str, Any] | None = None
                      ) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    """Return (model input row, raw inputs used, names of lag features left for imputation)."""
    overrides = _validate_overrides(overrides or {})
    prev_override = overrides.pop("prev_productivity_score", None)

    raw = {c: np.nan for c in RAW_COLUMNS}
    raw.update({k: v for k, v in (today_raw or {}).items() if v is not None and k in raw})
    raw.update(overrides)
    raw[DATE_COLUMN] = day.isoformat()
    raw[TARGET] = np.nan  # never reveal the day's own outcome
    if mode == "planning":
        for col in POST_DAY_FEATURES:
            if col in raw:
                raw[col] = np.nan

    missing = [c for c in REQUIRED_PREDICTION_INPUTS[mode] if pd.isna(raw.get(c))]
    if missing:
        hints = ", ".join(f"{c} ({INPUT_FLAGS.get(c, '?')})" for c in missing)
        raise DataError(
            f"Missing required input(s) for a {mode} prediction on {day.isoformat()}: {hints}. "
            "Record them with `day record` / `block add` / `habit add`, or pass the flags."
        )
    # Optional planned breakdowns default to zero when blocks were not categorised.
    for col in ("planned_study_hours", "planned_deep_work_hours"):
        if pd.isna(raw[col]):
            raw[col] = 0.0

    frames = []
    if history is not None and not history.empty:
        hist = history.copy()
        hist_dates = pd.to_datetime(hist[DATE_COLUMN])
        window_start = pd.Timestamp(day - timedelta(days=HISTORY_DAYS))
        hist = hist[(hist_dates < pd.Timestamp(day)) & (hist_dates >= window_start)]
        frames.append(hist[RAW_COLUMNS])
    frames.append(pd.DataFrame([raw], columns=RAW_COLUMNS))
    combined = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    for col in RAW_NUMERIC_SPEC:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").astype(float)

    featured = engineer_features(combined)
    row = featured[featured[DATE_COLUMN] == pd.Timestamp(day)].tail(1).copy()
    if prev_override is not None:
        row["prev_productivity_score"] = prev_override
    X = build_matrix(row, mode)
    imputed = [c for c in LAG_FEATURES if c in X.columns and X[c].isna().any()]
    used = {c: (None if pd.isna(X[c].iloc[0]) else X[c].iloc[0].item()) for c in X.columns}
    return X, used, imputed


def predict_day(bundle: ModelBundle, day: date, today_raw: dict[str, Any] | None,
                history: pd.DataFrame | None, overrides: dict[str, Any] | None = None) -> PredictionResult:
    X, used, imputed = build_feature_row(day, bundle.mode, today_raw, history, overrides)
    missing_cols = [c for c in bundle.features if c not in X.columns]
    if missing_cols:
        raise DataError(f"Model expects feature(s) not produced by this code: {', '.join(missing_cols)}.")
    score = float(np.clip(bundle.best.predict(X[bundle.features])[0], 0.0, 100.0))
    test_mae = (bundle.metadata.get("test_metrics", {}).get(bundle.selected, {}) or {}).get("mae")
    return PredictionResult(day, bundle.mode, round(score, 1), bundle.selected, test_mae, used, imputed)
