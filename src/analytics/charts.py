"""Matplotlib charts for application analytics and model evaluation.

All charts are written as PNG files using the non-interactive ``Agg`` backend,
so they work in terminals, CI and headless machines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Validated categorical order (first three slots are colour-vision-deficiency safe).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"

MODEL_LABELS = {
    "linear_regression": "Linear Regression",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
}


def _style(ax: plt.Axes, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.tick_params(colors=INK_MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def _label(name: str) -> str:
    return MODEL_LABELS.get(name, name.replace("_", " ").title())


# ----- application analytics ------------------------------------------------------

def plot_daily_scores(daily: pd.DataFrame, path: Path) -> Path:
    """Daily productivity score with its 7-day rolling mean."""
    df = daily.dropna(subset=["productivity_score"]).sort_values("date")
    fig, ax = plt.subplots(figsize=(9, 4))
    dates = pd.to_datetime(df["date"])
    ax.plot(dates, df["productivity_score"], color=SERIES[0], linewidth=1.2, alpha=0.55,
            marker="o", markersize=3, label="Daily score")
    rolling = df.set_index(dates)["productivity_score"].rolling("7D", min_periods=1).mean()
    ax.plot(rolling.index, rolling.values, color=SERIES[1], linewidth=2, label="7-day mean")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Daily productivity score", ylabel="Score (0-100)")
    fig.autofmt_xdate()
    return _save(fig, path)


def plot_weekly_hours(weekly: pd.DataFrame, path: Path) -> Path:
    """Planned vs completed hours per week (grouped bars)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(weekly))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, weekly["planned_hours"], width, color=SERIES[0], label="Planned")
    ax.bar(x + width / 2 + 0.01, weekly["completed_hours"], width, color=SERIES[1], label="Completed")
    ax.set_xticks(x, [pd.Timestamp(w).strftime("%d %b") for w in weekly["week_start"]], rotation=45)
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Planned vs completed hours per week", xlabel="Week starting", ylabel="Hours")
    return _save(fig, path)


def plot_habit_rates(names: Sequence[str], rates: Sequence[float], path: Path) -> Path:
    """Completion rate per habit (horizontal bars, labelled)."""
    order = np.argsort(rates)
    fig, ax = plt.subplots(figsize=(8, max(2.5, 0.5 * len(names) + 1)))
    vals = [rates[i] * 100 for i in order]
    ax.barh([names[i] for i in order], vals, color=SERIES[0], height=0.6)
    for y, val in enumerate(vals):
        ax.text(val + 1, y, f"{val:.0f}%", va="center", fontsize=9, color=INK_MUTED)
    ax.set_xlim(0, 110)
    _style(ax, "Habit completion rate", xlabel="% of scheduled days completed")
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_category_hours(planned: Mapping[str, float], completed: Mapping[str, float], path: Path) -> Path:
    cats = sorted(set(planned) | set(completed))
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(cats))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, [planned.get(c, 0) for c in cats], width, color=SERIES[0], label="Planned")
    ax.bar(x + width / 2 + 0.01, [completed.get(c, 0) for c in cats], width, color=SERIES[1], label="Completed")
    ax.set_xticks(x, [c.replace("_", " ") for c in cats])
    ax.legend(frameon=False, fontsize=9)
    _style(ax, "Hours by category", ylabel="Hours")
    return _save(fig, path)


# ----- model evaluation -------------------------------------------------------------

def plot_actual_vs_predicted(y_true: Sequence[float], y_pred: Sequence[float], model: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=18, color=SERIES[0], alpha=0.7, edgecolors=SURFACE, linewidths=0.5)
    ax.plot([0, 100], [0, 100], color=NEUTRAL, linewidth=1, linestyle="--", label="Perfect prediction")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, f"Actual vs predicted - {_label(model)}", "Actual score", "Predicted score")
    return _save(fig, path)


def plot_model_comparison(metrics: Mapping[str, Mapping[str, float]], path: Path,
                          baseline: Mapping[str, float] | None = None) -> Path:
    """MAE and RMSE per model (same unit, one axis); R² in its own panel."""
    names = list(metrics)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))
    width = 0.38
    mae = [metrics[n]["mae"] for n in names]
    rmse = [metrics[n]["rmse"] for n in names]
    ax1.bar(x - width / 2 - 0.01, mae, width, color=SERIES[0], label="MAE")
    ax1.bar(x + width / 2 + 0.01, rmse, width, color=SERIES[1], label="RMSE")
    for i, (a, b) in enumerate(zip(mae, rmse)):
        ax1.text(i - width / 2, a + 0.2, f"{a:.1f}", ha="center", fontsize=8, color=INK_MUTED)
        ax1.text(i + width / 2, b + 0.2, f"{b:.1f}", ha="center", fontsize=8, color=INK_MUTED)
    if baseline:
        ax1.axhline(baseline["mae"], color=NEUTRAL, linestyle="--", linewidth=1, label="Baseline MAE (mean)")
    top = max(rmse + ([baseline["mae"]] if baseline else []))
    ax1.set_ylim(0, top * 1.35)
    ax1.set_xticks(x, [_label(n) for n in names])
    ax1.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    _style(ax1, "Test error (lower is better)", ylabel="Score points")

    r2 = [metrics[n]["r2"] for n in names]
    ax2.bar(x, r2, 0.5, color=SERIES[2])
    for i, val in enumerate(r2):
        ax2.text(i, val + 0.01, f"{val:.3f}", ha="center", fontsize=8, color=INK_MUTED)
    ax2.set_xticks(x, [_label(n) for n in names])
    ax2.set_ylim(min(0.0, min(r2)), 1)
    _style(ax2, "Test R² (higher is better)")
    return _save(fig, path)


def plot_feature_importance(importance: pd.Series, model: str, path: Path, top: int = 15) -> Path:
    data = importance.head(top)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(data) + 1)))
    ax.barh(data.index, data.values, color=SERIES[0], height=0.6)
    is_linear = model == "linear_regression"
    xlabel = "|standardised coefficient|" if is_linear else "Impurity-based importance"
    _style(ax, f"Feature importance - {_label(model)}", xlabel=xlabel)
    ax.grid(axis="y", visible=False)
    return _save(fig, path)


def plot_residuals(y_true: Sequence[float], y_pred: Sequence[float], model: str, path: Path) -> Path:
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.hist(residuals, bins=25, color=SERIES[0], edgecolor=SURFACE)
    ax1.axvline(0, color=NEUTRAL, linewidth=1)
    _style(ax1, "Residual distribution", "Actual - predicted", "Days")
    ax2.scatter(y_pred, residuals, s=16, color=SERIES[0], alpha=0.7, edgecolors=SURFACE, linewidths=0.5)
    ax2.axhline(0, color=NEUTRAL, linewidth=1)
    _style(ax2, "Residuals vs predicted", "Predicted score", "Actual - predicted")
    fig.suptitle(f"Prediction errors - {_label(model)}", x=0.01, ha="left", color=INK)
    return _save(fig, path)


def plot_predictions_over_time(dates: Sequence, y_true: Sequence[float], y_pred: Sequence[float],
                               model: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4))
    d = pd.to_datetime(pd.Series(dates))
    ax.plot(d, y_true, color=SERIES[0], linewidth=1.5, label="Actual")
    ax.plot(d, y_pred, color=SERIES[1], linewidth=1.5, label=f"Predicted ({_label(model)})")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax, "Held-out test period", ylabel="Score (0-100)")
    fig.autofmt_xdate()
    return _save(fig, path)
