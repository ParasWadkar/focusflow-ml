"""Model definitions, chronological splitting and cross-validated training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.ml.evaluation import regression_metrics
from src.ml.features import CATEGORICAL_FEATURES, DATE_COLUMN, numeric_features
from src.utils.errors import DataError

RANDOM_STATE = 42
TEST_FRACTION = 0.2
CV_SPLITS = 5
MODEL_NAMES = ("linear_regression", "decision_tree", "random_forest")


def _preprocessor(features: list[str], scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = [c for c in features if c in CATEGORICAL_FEATURES]
    transformers = [("num", Pipeline(numeric_steps), numeric_features(features))]
    if categorical:
        transformers.append((
            "cat",
            OneHotEncoder(categories=[list(range(7))] * len(categorical),
                          handle_unknown="ignore", sparse_output=False),
            categorical,
        ))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def build_pipelines(features: list[str], random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """The three candidate regressors, each with its own preprocessing."""
    return {
        "linear_regression": Pipeline([
            ("prep", _preprocessor(features, scale=True)),
            ("model", LinearRegression()),
        ]),
        "decision_tree": Pipeline([
            ("prep", _preprocessor(features, scale=False)),
            ("model", DecisionTreeRegressor(max_depth=6, min_samples_leaf=10,
                                            random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("prep", _preprocessor(features, scale=False)),
            ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                            random_state=random_state, n_jobs=1)),
        ]),
    }


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_FRACTION,
                        split_date: pd.Timestamp | str | None = None
                        ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split by time: all training dates come strictly before all test dates.

    Shuffling is deliberately avoided. Lag features make neighbouring days
    correlated, so a random split would leak information into the test set.
    """
    if not 0 < test_fraction < 1:
        raise DataError("test_fraction must be between 0 and 1.")
    ordered = df.sort_values(DATE_COLUMN).reset_index(drop=True)
    if split_date is None:
        n_test = max(1, int(round(len(ordered) * test_fraction)))
        if len(ordered) - n_test < CV_SPLITS + 1:
            raise DataError(
                f"Not enough labelled rows to train ({len(ordered)}). Need at least "
                f"{CV_SPLITS + 1 + n_test} rows."
            )
        split_ts = pd.Timestamp(ordered[DATE_COLUMN].iloc[len(ordered) - n_test])
    else:
        split_ts = pd.Timestamp(split_date)
    train = ordered[ordered[DATE_COLUMN] < split_ts]
    test = ordered[ordered[DATE_COLUMN] >= split_ts]
    if train.empty or test.empty:
        raise DataError(f"Split at {split_ts.date()} leaves an empty train or test set.")
    return train.reset_index(drop=True), test.reset_index(drop=True), split_ts


@dataclass(frozen=True)
class CVResult:
    model: str
    mae_mean: float
    rmse_mean: float
    rmse_std: float
    r2_mean: float
    folds: int

    def as_dict(self) -> dict[str, float | int | str]:
        return {"model": self.model, "mae_mean": self.mae_mean, "rmse_mean": self.rmse_mean,
                "rmse_std": self.rmse_std, "r2_mean": self.r2_mean, "folds": self.folds}


def cross_validate(pipelines: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
                   n_splits: int = CV_SPLITS) -> dict[str, CVResult]:
    """Expanding-window time-series cross-validation on the training data only."""
    n_splits = min(n_splits, len(X) - 1)
    if n_splits < 2:
        raise DataError("Not enough training rows for cross-validation.")
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: dict[str, CVResult] = {}
    for name, pipeline in pipelines.items():
        fold_metrics = []
        for train_idx, val_idx in splitter.split(X):
            model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
            fold_metrics.append(regression_metrics(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
        rmse = np.array([m["rmse"] for m in fold_metrics])
        results[name] = CVResult(
            model=name,
            mae_mean=float(np.mean([m["mae"] for m in fold_metrics])),
            rmse_mean=float(rmse.mean()),
            rmse_std=float(rmse.std()),
            r2_mean=float(np.mean([m["r2"] for m in fold_metrics])),
            folds=n_splits,
        )
    return results


def fit_all(pipelines: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series) -> dict[str, Pipeline]:
    return {name: clone(p).fit(X, y) for name, p in pipelines.items()}
