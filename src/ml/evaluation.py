"""Evaluation metrics and model introspection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R² (R² is NaN when the target is constant)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 and np.ptp(y_true) > 0 else float("nan")
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2,
    }


def baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Metrics of always predicting the training mean, as a reference point."""
    return regression_metrics(y_test, np.full(len(y_test), float(y_train.mean())))


def predict_clipped(model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(X), 0.0, 100.0)


def evaluate_models(models: dict[str, Pipeline], X_test: pd.DataFrame, y_test: pd.Series
                    ) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics, predictions = {}, {}
    for name, model in models.items():
        preds = predict_clipped(model, X_test)
        predictions[name] = preds
        metrics[name] = regression_metrics(y_test, preds)
    return metrics, predictions


def feature_importance(model: Pipeline) -> pd.Series | None:
    """Importances for tree models, or |standardised coefficient| for linear models.

    Returns ``None`` if the estimator exposes neither.
    """
    names = model.named_steps["prep"].get_feature_names_out()
    estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.ravel(estimator.coef_))
    else:
        return None
    return pd.Series(values, index=names).sort_values(ascending=False)


def error_analysis(y_true: pd.Series, y_pred: np.ndarray, dates: pd.Series) -> dict[str, object]:
    """Residual summary and the days with the largest errors."""
    residuals = np.asarray(y_true, dtype=float) - y_pred
    abs_err = np.abs(residuals)
    worst = np.argsort(abs_err)[::-1][:5]
    return {
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "within_5_points": float((abs_err <= 5).mean()),
        "within_10_points": float((abs_err <= 10).mean()),
        "largest_errors": [
            {"date": pd.Timestamp(dates.iloc[i]).strftime("%Y-%m-%d"),
             "actual": float(np.asarray(y_true)[i]), "predicted": float(y_pred[i])}
            for i in worst
        ],
    }
