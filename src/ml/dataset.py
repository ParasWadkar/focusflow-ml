"""Synthetic seed dataset: generation and loading.

The seed dataset is SYNTHETIC. It was not collected from real people. It
exists so the ML pipeline can be developed and demonstrated before a user
has logged enough real days. Real records from the SQLite database can be
used instead of it, or combined with it (see ``src.ml.pipeline``).

Generative story, per day:

* A hidden "form" variable follows an AR(1) process, so good and bad
  stretches persist across days. It is never written to the dataset, so
  models can only pick it up indirectly through lagged outcomes.
* Sleep depends on weekends and form. Morning energy follows an inverted U
  over sleep (best around 7.5 h).
* Planned workload depends on the weekday. Interruptions grow with workload.
* The share of planned work that actually gets done is a logistic function of
  energy, form, interruptions and over-planning.
* The productivity score combines completed work with diminishing returns,
  deep work (hurt more by interruptions), habit completion, a sleep
  inverted-U, energy, mood, exercise, a weekday effect and Gaussian noise.

A small share of values is removed completely at random (MCAR), to imitate
days where a field was not logged.
"""

from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features import DATE_COLUMN, RAW_COLUMNS, TARGET
from src.utils.errors import DataError

DEFAULT_ROWS = 1000
DEFAULT_SEED = 42
DEFAULT_START = date(2023, 1, 2)
MISSING_RATE = 0.02
MISSING_TARGET_RATE = 0.005
MISSABLE_COLUMNS = ["sleep_hours", "energy_level", "mood_score", "exercise_minutes", "interruptions"]

_DOW_EFFECT = np.array([-1.5, 0.5, 1.0, 0.5, -1.0, -2.0, -2.5])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _round_to(values: np.ndarray, step: float) -> np.ndarray:
    # ``+ 0.0`` normalises negative zero produced by rounding tiny negatives.
    return np.maximum(np.round(values / step) * step, 0.0) + 0.0


def productivity_score(df: pd.DataFrame, form: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """Ground-truth scoring function used by the generator (and by the demo data)."""
    dow = pd.to_datetime(df[DATE_COLUMN]).dt.dayofweek.to_numpy()
    habit_rate = np.where(df["habits_planned"] > 0,
                          df["habits_completed"] / df["habits_planned"].replace(0, np.nan), 0.0)
    habit_rate = np.nan_to_num(habit_rate, nan=0.0)
    sleep = df["sleep_hours"].to_numpy(dtype=float)
    score = (
        28.0
        + 9.0 * np.sqrt(df["completed_task_hours"].to_numpy(dtype=float))
        + 7.0 * np.log1p(df["deep_work_hours"].to_numpy(dtype=float))
        + 1.5 * np.minimum(df["study_hours"].to_numpy(dtype=float), 4.0)
        + 10.0 * habit_rate
        - np.minimum(1.6 * (sleep - 7.5) ** 2, 15.0)
        + 1.5 * (df["energy_level"].to_numpy(dtype=float) - 5.5)
        + 1.0 * (df["mood_score"].to_numpy(dtype=float) - 6.0)
        - 0.7 * df["interruptions"].to_numpy(dtype=float)
        * (1.0 + 0.2 * df["planned_deep_work_hours"].to_numpy(dtype=float))
        + 4.0 * np.minimum(df["exercise_minutes"].to_numpy(dtype=float), 60.0) / 60.0
        + 3.0 * form
        + _DOW_EFFECT[dow]
        + noise
    )
    return np.clip(np.round(score, 1), 0.0, 100.0)


def generate_dataset(rows: int = DEFAULT_ROWS, seed: int = DEFAULT_SEED,
                     start: date = DEFAULT_START, missing: bool = True,
                     return_form: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    """Generate ``rows`` consecutive synthetic days, reproducibly for a given ``seed``."""
    if rows < 1:
        raise DataError("rows must be a positive integer.")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=rows, freq="D")
    dow = dates.dayofweek.to_numpy()
    weekend = (dow >= 5).astype(float)

    form = np.zeros(rows)
    for t in range(1, rows):
        form[t] = 0.8 * form[t - 1] + rng.normal(0, 0.6)

    sleep = np.clip(7.0 + 0.6 * weekend + 0.3 * form + rng.normal(0, 0.9, rows), 3.5, 10.5)
    sleep = np.round(sleep, 1)
    energy = 5.5 - 0.35 * (sleep - 7.6) ** 2 + 0.8 * form + 0.6 * (sleep - 7.0) + rng.normal(0, 1.1, rows)
    energy = np.clip(np.round(energy), 1, 10)

    habits_planned = np.where(weekend == 1, 3, 5) + rng.integers(0, 2, rows)
    planned = np.where(weekend == 1, rng.normal(3.0, 1.5, rows), rng.normal(7.0, 1.3, rows))
    planned = np.clip(_round_to(planned, 0.25), 0.0, 12.0)
    study_share = rng.beta(2, 4, rows)
    deep_share = rng.beta(2, 3, rows) * (1 - study_share)
    planned_study = _round_to(planned * study_share, 0.25)
    planned_deep = np.minimum(_round_to(planned * deep_share, 0.25), planned - planned_study)

    interruptions = rng.poisson(np.clip(1.5 + 0.4 * planned - 0.3 * form, 0.2, None))

    exercised = rng.random(rows) < np.clip(0.5 + 0.1 * form + 0.1 * weekend, 0.05, 0.95)
    exercise = np.where(exercised, _round_to(rng.gamma(3.0, 12.0, rows), 5.0), 0.0)

    mood = (6.0 + 0.5 * form + 0.25 * (energy - 5.5) + 0.01 * exercise
            - 0.1 * interruptions + rng.normal(0, 1.2, rows))
    mood = np.clip(np.round(mood), 1, 10)

    z = (1.4 + 0.35 * (energy - 5.5) + 0.45 * form - 0.12 * interruptions
         - 0.25 * np.maximum(0.0, planned - 8.0) + rng.normal(0, 0.5, rows))
    execution = _sigmoid(z)
    completed = np.minimum(_round_to(planned * np.clip(execution + rng.normal(0, 0.05, rows), 0, 1), 0.25), planned)
    study = np.minimum(_round_to(planned_study * np.clip(execution + rng.normal(0, 0.08, rows), 0, 1), 0.25),
                       planned_study)
    deep_exec = _sigmoid(z - 0.1 * interruptions)
    deep = np.minimum(_round_to(planned_deep * np.clip(deep_exec + rng.normal(0, 0.08, rows), 0, 1), 0.25),
                      planned_deep)

    habit_p = _sigmoid(0.6 + 0.3 * (energy - 5.5) + 0.4 * form + 0.3 * exercised)
    habits_completed = rng.binomial(habits_planned, habit_p)

    df = pd.DataFrame({
        DATE_COLUMN: dates.strftime("%Y-%m-%d"),
        "sleep_hours": sleep,
        "energy_level": energy.astype(int),
        "mood_score": mood.astype(int),
        "exercise_minutes": exercise,
        "interruptions": interruptions.astype(int),
        "habits_planned": habits_planned.astype(int),
        "habits_completed": habits_completed.astype(int),
        "planned_task_hours": planned,
        "completed_task_hours": completed,
        "planned_study_hours": planned_study,
        "study_hours": study,
        "planned_deep_work_hours": planned_deep,
        "deep_work_hours": deep,
    })
    df[TARGET] = productivity_score(df, form, rng.normal(0, 4.0, rows))

    if missing:
        df = inject_missing(df, rng)
    df = df[RAW_COLUMNS]
    return (df, form) if return_form else df


def inject_missing(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Blank a small random share of optional fields and targets (MCAR)."""
    out = df.copy()
    for col in MISSABLE_COLUMNS:
        mask = rng.random(len(out)) < MISSING_RATE
        out[col] = out[col].astype(float)
        out.loc[mask, col] = np.nan
    target_mask = rng.random(len(out)) < MISSING_TARGET_RATE
    out.loc[target_mask, TARGET] = np.nan
    return out


def save_dataset(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4g", lineterminator="\n")
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    """Read a raw daily CSV. Structural problems raise :class:`DataError`."""
    if not path.exists():
        raise DataError(
            f"Dataset not found at {path}. Run `python -m src.main dataset generate` "
            "or `python -m src.main init`."
        )
    try:
        df = pd.read_csv(path)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise DataError(f"Could not read dataset {path}: {exc}") from exc
    if df.empty:
        raise DataError(f"Dataset {path} contains no rows.")
    return df


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Stable SHA-256 of a data frame's content, for reproducibility tracking."""
    hashed = pd.util.hash_pandas_object(df.reset_index(drop=True), index=True).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()
