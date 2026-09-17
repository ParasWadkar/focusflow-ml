"""End-to-end ML pipeline.

    raw data -> validation -> cleaning -> feature engineering -> chronological split
    -> time-series CV on train -> selection -> refit on train -> held-out test
    -> persistence

``evaluate_saved`` reloads the persisted bundle and rebuilds exactly the same
split from the stored split date, so evaluation can be run independently of
training.
"""

from __future__ import annotations

import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sklearn

from src.config import Settings
from src.ml import evaluation as ev
from src.ml.cleaning import CleaningSummary, clean
from src.ml.dataset import frame_fingerprint, load_dataset
from src.ml.features import DATE_COLUMN, TARGET, build_matrix, engineer_features, feature_list, leakage_check
from src.ml.persistence import ModelBundle, load_bundle, save_bundle
from src.ml.selection import select_best
from src.ml.training import (CV_SPLITS, RANDOM_STATE, TEST_FRACTION, build_pipelines,
                             chronological_split, cross_validate, fit_all)
from src.ml.validation import ValidationReport, validate_raw
from src.utils.errors import DataError, ModelError

SOURCES = ("seed", "db", "combined")
MIN_DB_ROWS = 60
MIN_ROWS = 30

DbLoader = Callable[[], pd.DataFrame]


@dataclass
class PreparedData:
    source: str
    features: pd.DataFrame          # labelled rows with engineered features
    report: ValidationReport
    cleaning: CleaningSummary
    fingerprint: str


@dataclass
class TrainingResult:
    bundle: ModelBundle
    cv: dict
    test_metrics: dict[str, dict[str, float]]
    baseline: dict[str, float]
    split_date: pd.Timestamp
    n_train: int
    n_test: int


@dataclass
class EvaluationResult:
    mode: str
    selected: str
    features: list[str]
    test_metrics: dict[str, dict[str, float]]
    baseline: dict[str, float]
    cv: dict
    dates: pd.Series
    y_test: pd.Series
    predictions: dict[str, np.ndarray]
    importances: dict[str, pd.Series]
    errors: dict[str, object]
    n_train: int
    n_test: int
    split_date: str
    warnings: list[str] = field(default_factory=list)


# ----- data -------------------------------------------------------------------

def load_raw(source: str, settings: Settings, db_loader: DbLoader | None = None) -> pd.DataFrame:
    if source not in SOURCES:
        raise DataError(f"Unknown data source '{source}'. Choose one of: {', '.join(SOURCES)}.")
    frames = []
    if source in ("seed", "combined"):
        frames.append(load_dataset(settings.dataset_path))
    if source in ("db", "combined"):
        if db_loader is None:
            raise DataError("A database connection is required for the 'db' and 'combined' sources.")
        db_frame = db_loader()
        labelled = int(pd.to_numeric(db_frame.get(TARGET), errors="coerce").notna().sum()) if len(db_frame) else 0
        if source == "db" and labelled < MIN_DB_ROWS:
            raise DataError(
                f"The database has only {labelled} day(s) with a productivity score; at least "
                f"{MIN_DB_ROWS} are needed to train on real data alone. Keep logging with "
                "`day record --score`, or use `--source combined`."
            )
        frames.append(db_frame)
    if len(frames) == 1:
        return frames[0]
    seed, db = frames
    combined = pd.concat([seed, db], ignore_index=True)
    # Real records win over synthetic rows that share a date.
    return combined.drop_duplicates(subset=DATE_COLUMN, keep="last").reset_index(drop=True)


def prepare(raw: pd.DataFrame, source: str, min_rows: int = MIN_ROWS) -> PreparedData:
    report = validate_raw(raw, min_rows=min_rows)
    cleaned, summary = clean(raw, require_target=False)
    featured = engineer_features(cleaned)
    labelled = featured[featured[TARGET].notna()].reset_index(drop=True)
    dropped = len(featured) - len(labelled)
    summary = CleaningSummary(summary.rows_in, len(labelled), dropped,
                              summary.values_nulled_out_of_range, summary.values_capped)
    return PreparedData(source, labelled, report, summary, frame_fingerprint(cleaned))


# ----- training -----------------------------------------------------------------

def train_mode(data: PreparedData, mode: str, models_dir: Path,
               test_fraction: float = TEST_FRACTION, random_state: int = RANDOM_STATE) -> TrainingResult:
    features = feature_list(mode)
    if mode == "planning" and leakage_check(features):
        raise DataError(f"Planning feature set contains post-day features: {leakage_check(features)}")

    train_df, test_df, split_ts = chronological_split(data.features, test_fraction)
    X_train, y_train = build_matrix(train_df, mode), train_df[TARGET]
    X_test, y_test = build_matrix(test_df, mode), test_df[TARGET]

    pipelines = build_pipelines(features, random_state)
    cv = cross_validate(pipelines, X_train, y_train, CV_SPLITS)
    selected = select_best(cv)
    fitted = fit_all(pipelines, X_train, y_train)
    test_metrics, _ = ev.evaluate_models(fitted, X_test, y_test)
    baseline = ev.baseline_metrics(y_train, y_test)

    metadata = {
        "mode": mode,
        "selected_model": selected,
        "selection_rule": f"lowest mean RMSE over {CV_SPLITS}-fold TimeSeriesSplit on the training period",
        "features": features,
        "source": data.source,
        "dataset_fingerprint": data.fingerprint,
        "split_date": split_ts.strftime("%Y-%m-%d"),
        "test_fraction": test_fraction,
        "train_period": [train_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                         train_df[DATE_COLUMN].max().strftime("%Y-%m-%d")],
        "test_period": [test_df[DATE_COLUMN].min().strftime("%Y-%m-%d"),
                        test_df[DATE_COLUMN].max().strftime("%Y-%m-%d")],
        "n_train": len(train_df),
        "n_test": len(test_df),
        "cv": {k: r.as_dict() for k, r in cv.items()},
        "test_metrics": test_metrics,
        "baseline_mean_predictor": baseline,
        "cleaning": asdict(data.cleaning),
        "validation_warnings": data.report.warnings,
        "random_state": random_state,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    bundle = ModelBundle(mode, features, selected, fitted, metadata)
    save_bundle(bundle, models_dir)
    return TrainingResult(bundle, metadata["cv"], test_metrics, baseline, split_ts,
                          len(train_df), len(test_df))


# ----- evaluation --------------------------------------------------------------

def evaluate_saved(settings: Settings, mode: str, db_loader: DbLoader | None = None) -> EvaluationResult:
    bundle = load_bundle(settings.models_dir, mode)
    meta = bundle.metadata
    if "split_date" not in meta or "source" not in meta:
        raise ModelError(f"Metadata for the {mode} model is missing; run `python -m src.main train`.")
    notes: list[str] = []
    data = prepare(load_raw(meta["source"], settings, db_loader), meta["source"])
    if data.fingerprint != meta.get("dataset_fingerprint"):
        notes.append("The dataset has changed since this model was trained; metrics reflect the "
                     "current data. Retrain for a like-for-like evaluation.")
    if bundle.features != feature_list(mode):
        raise ModelError(f"The saved {mode} model uses a different feature list than this code. "
                         "Run `python -m src.main train`.")

    train_df, test_df, _ = chronological_split(data.features, split_date=meta["split_date"])
    X_test, y_test = build_matrix(test_df, mode), test_df[TARGET]
    metrics, predictions = ev.evaluate_models(bundle.models, X_test, y_test)
    importances = {name: imp for name, m in bundle.models.items()
                   if (imp := ev.feature_importance(m)) is not None}
    return EvaluationResult(
        mode=mode,
        selected=bundle.selected,
        features=bundle.features,
        test_metrics=metrics,
        baseline=ev.baseline_metrics(train_df[TARGET], y_test),
        cv=meta.get("cv", {}),
        dates=test_df[DATE_COLUMN],
        y_test=y_test,
        predictions=predictions,
        importances=importances,
        errors=ev.error_analysis(y_test, predictions[bundle.selected], test_df[DATE_COLUMN]),
        n_train=len(train_df),
        n_test=len(test_df),
        split_date=meta["split_date"],
        warnings=notes,
    )

