"""Write evaluation results to disk (JSON, Markdown and PNG charts)."""

from __future__ import annotations

import json
from pathlib import Path

from src.analytics import charts
from src.ml.pipeline import EvaluationResult


def write_evaluation_report(result: EvaluationResult, reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    mode, best = result.mode, result.selected
    y = result.y_test.to_numpy()
    files: dict[str, Path] = {}

    files["actual_vs_predicted"] = charts.plot_actual_vs_predicted(
        y, result.predictions[best], best, reports_dir / f"{mode}_actual_vs_predicted.png")
    files["model_comparison"] = charts.plot_model_comparison(
        result.test_metrics, reports_dir / f"{mode}_model_comparison.png", result.baseline)
    files["residuals"] = charts.plot_residuals(
        y, result.predictions[best], best, reports_dir / f"{mode}_residuals.png")
    files["test_timeline"] = charts.plot_predictions_over_time(
        result.dates, y, result.predictions[best], best, reports_dir / f"{mode}_test_timeline.png")
    for name, importance in result.importances.items():
        files[f"importance_{name}"] = charts.plot_feature_importance(
            importance, name, reports_dir / f"{mode}_feature_importance_{name}.png")

    payload = {
        "mode": mode,
        "selected_model": best,
        "split_date": result.split_date,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "features": result.features,
        "cv": result.cv,
        "test_metrics": result.test_metrics,
        "baseline_mean_predictor": result.baseline,
        "error_analysis": result.errors,
        "feature_importance": {k: {f: float(val) for f, val in s.items()} for k, s in result.importances.items()},
        "warnings": result.warnings,
    }
    files["json"] = reports_dir / f"metrics_{mode}.json"
    files["json"].write_text(json.dumps(payload, indent=2), encoding="utf-8")
    files["markdown"] = reports_dir / f"evaluation_{mode}.md"
    files["markdown"].write_text(_markdown(result, files, reports_dir), encoding="utf-8")
    return files


def _markdown(result: EvaluationResult, files: dict[str, Path], reports_dir: Path) -> str:
    lines = [
        f"# Evaluation: {result.mode} model",
        "",
        f"- Selected model: **{result.selected}** (lowest mean time-series CV RMSE on the training period)",
        f"- Train rows: {result.n_train}; test rows: {result.n_test}; test period starts {result.split_date}",
        "",
        "## Cross-validation (training period only)",
        "",
        "| Model | CV MAE | CV RMSE (± sd) | CV R² |",
        "|---|---:|---:|---:|",
    ]
    for name, cv in result.cv.items():
        lines.append(f"| {name} | {cv['mae_mean']:.2f} | {cv['rmse_mean']:.2f} ± {cv['rmse_std']:.2f} "
                     f"| {cv['r2_mean']:.3f} |")
    lines += ["", "## Held-out test set", "", "| Model | MAE | RMSE | R² |", "|---|---:|---:|---:|"]
    for name, m in result.test_metrics.items():
        marker = " (selected)" if name == result.selected else ""
        lines.append(f"| {name}{marker} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} |")
    b = result.baseline
    lines.append(f"| mean-of-train baseline | {b['mae']:.2f} | {b['rmse']:.2f} | {b['r2']:.3f} |")
    err = result.errors
    lines += [
        "",
        "## Error analysis (selected model)",
        "",
        f"- Mean residual: {err['residual_mean']:.2f}; residual sd: {err['residual_std']:.2f}",
        f"- Within ±5 points: {err['within_5_points']:.0%}; within ±10 points: {err['within_10_points']:.0%}",
        "",
        "## Charts",
        "",
    ]
    for key, path in files.items():
        if path.suffix == ".png":
            rel = path.relative_to(reports_dir).as_posix()
            lines.append(f"- {key}: ![{key}]({rel})")
    if result.warnings:
        lines += ["", "## Warnings", "", *[f"- {w}" for w in result.warnings]]
    return "\n".join(lines) + "\n"
