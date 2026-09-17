"""Feature definitions and feature engineering.

Leakage policy
--------------
The model can run in two modes, each with an explicit feature list:

* ``planning``: predicts a day's score at the *start* of the day. It uses
  only information known by then: last night's sleep, morning energy, what is
  planned for the day (habits, task hours, study and deep-work hours), the
  calendar, and lagged outcomes from *previous* days.
* ``retrospective``: explains a day that has already finished. It may also
  use outcomes that are only known once the day is over (completed hours,
  habits completed, mood, interruptions, exercise). It must never be used to
  forecast a day that has not happened yet.

Lag features are computed on a calendar-day index, so "previous day" really
means yesterday, and the rolling mean excludes the current day.
"""

from __future__ import annotations

import pandas as pd

from src.utils.errors import DataError

DATE_COLUMN = "date"
TARGET = "productivity_score"

# Raw daily table: the same schema for the synthetic CSV and for DB extracts.
# column -> (minimum, maximum), inclusive
RAW_NUMERIC_SPEC: dict[str, tuple[float, float]] = {
    "sleep_hours": (0, 24),
    "energy_level": (1, 10),
    "mood_score": (1, 10),
    "exercise_minutes": (0, 1440),
    "interruptions": (0, 500),
    "habits_planned": (0, 100),
    "habits_completed": (0, 100),
    "planned_task_hours": (0, 24),
    "completed_task_hours": (0, 24),
    "planned_study_hours": (0, 24),
    "study_hours": (0, 24),
    "planned_deep_work_hours": (0, 24),
    "deep_work_hours": (0, 24),
    TARGET: (0, 100),
}
RAW_COLUMNS: list[str] = [DATE_COLUMN, *RAW_NUMERIC_SPEC]

CATEGORICAL_FEATURES = ["day_of_week"]

PLANNING_FEATURES: list[str] = [
    "sleep_hours",
    "energy_level",
    "habits_planned",
    "planned_task_hours",
    "planned_study_hours",
    "planned_deep_work_hours",
    "day_of_week",
    "is_weekend",
    "prev_productivity_score",
    "prev_habit_completion_rate",
    "rolling_7d_productivity",
]

# Known only after the day has ended.
POST_DAY_FEATURES: list[str] = [
    "habits_completed",
    "habit_completion_rate",
    "completed_task_hours",
    "task_completion_ratio",
    "study_hours",
    "deep_work_hours",
    "exercise_minutes",
    "mood_score",
    "interruptions",
]

RETROSPECTIVE_FEATURES: list[str] = PLANNING_FEATURES + POST_DAY_FEATURES

FEATURE_SETS: dict[str, list[str]] = {
    "planning": PLANNING_FEATURES,
    "retrospective": RETROSPECTIVE_FEATURES,
}
MODES = tuple(FEATURE_SETS)

# Raw inputs a prediction cannot sensibly be made without (lags may be imputed).
REQUIRED_PREDICTION_INPUTS: dict[str, list[str]] = {
    "planning": ["sleep_hours", "energy_level", "habits_planned", "planned_task_hours"],
    "retrospective": ["sleep_hours", "energy_level", "habits_planned", "planned_task_hours",
                      "habits_completed", "completed_task_hours", "mood_score", "interruptions"],
}


def feature_list(mode: str) -> list[str]:
    try:
        return list(FEATURE_SETS[mode])
    except KeyError:
        raise ValueError(f"Unknown mode '{mode}'. Choose one of: {', '.join(MODES)}.") from None


def numeric_features(features: list[str]) -> list[str]:
    return [f for f in features if f not in CATEGORICAL_FEATURES]


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    ratio = numerator / denominator.where(denominator > 0)
    return ratio.clip(0, 1)


def engineer_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to a cleaned raw daily table.

    The input must have unique, parseable dates. Rows are returned sorted by
    date, with the original rows only (calendar gap-filling is internal).
    """
    df = raw.copy()
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

    df["day_of_week"] = df[DATE_COLUMN].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["habit_completion_rate"] = _safe_ratio(df["habits_completed"], df["habits_planned"])
    df["task_completion_ratio"] = _safe_ratio(df["completed_task_hours"], df["planned_task_hours"])

    # Lags on a continuous calendar so that gaps in logging are not bridged.
    calendar = df.set_index(DATE_COLUMN)[[TARGET, "habit_completion_rate"]]
    full_index = pd.date_range(calendar.index.min(), calendar.index.max(), freq="D")
    calendar = calendar.reindex(full_index)
    lagged = pd.DataFrame(index=full_index)
    lagged["prev_productivity_score"] = calendar[TARGET].shift(1)
    lagged["prev_habit_completion_rate"] = calendar["habit_completion_rate"].shift(1)
    # Mean of the previous 7 calendar days (today excluded); needs >= 3 observations.
    lagged["rolling_7d_productivity"] = (
        calendar[TARGET].shift(1).rolling(window=7, min_periods=3).mean()
    )
    df = df.join(lagged, on=DATE_COLUMN)
    return df


def build_matrix(features_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Select the model inputs for ``mode``, failing clearly if any are absent."""
    cols = feature_list(mode)
    missing = [c for c in cols if c not in features_df.columns]
    if missing:
        raise DataError(f"Missing ML feature column(s) for {mode} mode: {', '.join(missing)}.")
    X = features_df[cols].copy()
    X["day_of_week"] = X["day_of_week"].astype(int)
    return X.astype({c: float for c in numeric_features(cols)})


def leakage_check(features: list[str]) -> list[str]:
    """Return any post-day features present in a feature list (should be empty for planning)."""
    return [f for f in features if f in POST_DAY_FEATURES or f == TARGET]

