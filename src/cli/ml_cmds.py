"""`train`, `evaluate` and `predict` commands."""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import pandas as pd

from src.cli.context import AppContext
from src.cli.prompts import interactive
from src.habits.service import HabitService
from src.journal.service import JournalService
from src.ml import pipeline
from src.ml.db_extract import daily_row, extract_daily_frame
from src.ml.features import MODES, REQUIRED_PREDICTION_INPUTS, feature_list, leakage_check
from src.ml.persistence import load_bundle
from src.ml.prediction import HISTORY_DAYS, INPUT_FLAGS, predict_day
from src.ml.report import write_evaluation_report
from src.planner.service import PlannerService
from src.utils.errors import ModelError, ValidationError
from src.utils.formatting import format_kv, format_table
from src.utils.validators import parse_date

MODE_CHOICES = (*MODES, "all")

# CLI flag (argparse dest) -> raw input name
OVERRIDE_DESTS = {flag.lstrip("-").replace("-", "_"): name for name, flag in INPUT_FLAGS.items()}


def register(sub: argparse._SubParsersAction) -> None:
    t = sub.add_parser("train", help="Train, compare and save the regression models.",
                       description="Train Linear Regression, Decision Tree and Random Forest models, "
                                   "select one by time-series cross-validation on the training period, "
                                   "and report held-out test metrics.")
    t.add_argument("--mode", choices=MODE_CHOICES, default="all",
                   help="planning (start-of-day forecast), retrospective (after-the-day analysis) or all.")
    t.add_argument("--source", choices=pipeline.SOURCES, default="seed",
                   help="seed = synthetic CSV (default), db = your logged days, combined = both.")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("evaluate", help="Evaluate saved models on the held-out test period and write charts.")
    e.add_argument("--mode", choices=MODE_CHOICES, default="all")
    e.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("predict", help="Predict a productivity score (0-100) for a date.",
                       description="Predict a day's productivity score. Inputs come from the database "
                                   "(daily record, time blocks, habits, previous days) and can be "
                                   "supplied or overridden with flags.")
    p.add_argument("--date", help="YYYY-MM-DD (default: today).")
    p.add_argument("--mode", choices=MODES, default="planning",
                   help="planning (default) uses only start-of-day information.")
    g = p.add_argument_group("inputs (override database values)")
    g.add_argument("--sleep", type=float, help="Hours slept last night.")
    g.add_argument("--energy", type=float, help="Morning energy level (1-10).")
    g.add_argument("--habits-planned", type=float, help="Number of habits scheduled today.")
    g.add_argument("--planned-hours", type=float, help="Total planned task hours.")
    g.add_argument("--planned-study", type=float, help="Planned study hours.")
    g.add_argument("--planned-deep", type=float, help="Planned deep-work hours.")
    g.add_argument("--prev-score", type=float, help="Yesterday's productivity score.")
    r = p.add_argument_group("retrospective-only inputs")
    r.add_argument("--habits-completed", type=float)
    r.add_argument("--completed-hours", type=float)
    r.add_argument("--study-hours", type=float)
    r.add_argument("--deep-hours", type=float)
    r.add_argument("--mood", type=float, help="Mood score (1-10).")
    r.add_argument("--interruptions", type=float)
    r.add_argument("--exercise", type=float, help="Exercise minutes.")
    p.set_defaults(func=cmd_predict)


def _modes(value: str) -> list[str]:
    return list(MODES) if value == "all" else [value]


def _db_loader(ctx: AppContext):
    return lambda: extract_daily_frame(ctx.conn, ctx.user_id)


def cmd_train(args, ctx: AppContext) -> None:
    loader = _db_loader(ctx) if args.source != "seed" else None
    raw = pipeline.load_raw(args.source, ctx.settings, loader)
    data = pipeline.prepare(raw, args.source)
    c = data.cleaning
    print(f"Data source: {args.source} | rows: {c.rows_in} | labelled rows used: {c.rows_out} "
          f"| dropped (no score): {c.dropped_missing_target}")
    for warning in data.report.warnings:
        print(f"  note: {warning} Missing values are median-imputed inside the model (fit on training rows only).")

    for mode in _modes(args.mode):
        leaks = leakage_check(feature_list(mode))
        print(f"\n=== {mode} model ({len(feature_list(mode))} features"
              + ("" if mode == "retrospective" else ", no post-day features") + ") ===")
        if mode == "planning" and leaks:
            raise ValidationError(f"Leakage guard failed: {leaks}")
        result = pipeline.train_mode(data, mode, ctx.settings.models_dir)
        meta = result.bundle.metadata
        print(f"Chronological split: train {meta['train_period'][0]}..{meta['train_period'][1]} "
              f"({result.n_train} rows) | test {meta['test_period'][0]}..{meta['test_period'][1]} "
              f"({result.n_test} rows)")
        rows = []
        for name, cv in result.cv.items():
            t = result.test_metrics[name]
            rows.append([name + (" *" if name == result.bundle.selected else ""),
                         f"{cv['rmse_mean']:.2f} +/- {cv['rmse_std']:.2f}", cv["r2_mean"],
                         t["mae"], t["rmse"], t["r2"]])
        b = result.baseline
        rows.append(["baseline (train mean)", "-", None, b["mae"], b["rmse"], b["r2"]])
        print(format_table(["model", "CV RMSE", "CV R2", "test MAE", "test RMSE", "test R2"], rows))
        print(f"* selected: {result.bundle.selected} (lowest mean CV RMSE on the training period; "
              "the test set was not used for selection)")
    print(f"\nModels saved to {ctx.settings.models_dir}. Next: `python -m src.main evaluate`.")


def cmd_evaluate(args, ctx: AppContext) -> None:
    modes = _modes(args.mode)
    evaluated = 0
    for mode in modes:
        try:
            load_bundle(ctx.settings.models_dir, mode)
        except ModelError:
            if args.mode == "all":
                print(f"Skipping {mode}: no trained model found.")
                continue
            raise
        loader = _db_loader(ctx) if ctx.db_available() else None
        result = pipeline.evaluate_saved(ctx.settings, mode, loader)
        files = write_evaluation_report(result, ctx.settings.reports_dir)
        evaluated += 1
        print(f"\n=== {mode} model: held-out test period from {result.split_date} "
              f"({result.n_test} days; trained on {result.n_train}) ===")
        rows = [[n + (" *" if n == result.selected else ""), m["mae"], m["rmse"], m["r2"]]
                for n, m in result.test_metrics.items()]
        rows.append(["baseline (train mean)", result.baseline["mae"], result.baseline["rmse"], result.baseline["r2"]])
        print(format_table(["model", "MAE", "RMSE", "R2"], rows))
        err = result.errors
        print(f"Selected model: {result.selected}. Residual mean {err['residual_mean']:+.2f}, "
              f"sd {err['residual_std']:.2f}; {err['within_5_points']:.0%} of days within +/-5 points, "
              f"{err['within_10_points']:.0%} within +/-10.")
        best_imp = result.importances.get(result.selected)
        if best_imp is not None:
            top = ", ".join(f"{k} ({v:.3f})" for k, v in best_imp.head(5).items())
            label = "|coef|" if result.selected == "linear_regression" else "importance"
            print(f"Top features by {label}: {top}")
        for warning in result.warnings:
            print(f"  warning: {warning}")
        print(f"Report: {files['markdown']}")
    if evaluated == 0:
        raise ModelError("No trained models found. Run `python -m src.main train` first.")
    print(f"\nCharts and metrics written to {ctx.settings.reports_dir}")


def _db_inputs(ctx: AppContext, day: date, mode: str) -> tuple[dict, pd.DataFrame | None]:
    """Raw inputs for ``day`` and recent history from the database (if initialised)."""
    if not ctx.db_available():
        return {}, None
    conn, uid = ctx.conn, ctx.user_id
    habits, planner = HabitService(conn, uid), PlannerService(conn, uid)
    row = daily_row(day, habits, planner, JournalService(conn, uid))
    has_blocks = bool(planner.for_day(day))
    if not has_blocks:  # no plan recorded: unknown, not zero
        for key in ("planned_task_hours", "completed_task_hours", "planned_study_hours",
                    "study_hours", "planned_deep_work_hours", "deep_work_hours"):
            row[key] = None
    if not habits.list(include_archived=True):
        row["habits_planned"] = row["habits_completed"] = None
    history = extract_daily_frame(conn, uid, day - timedelta(days=HISTORY_DAYS), day - timedelta(days=1))
    return row, history


def _prompt_missing(mode: str, today_raw: dict, overrides: dict) -> None:
    for name in REQUIRED_PREDICTION_INPUTS[mode]:
        if today_raw.get(name) is None and overrides.get(name) is None:
            while True:
                try:
                    answer = input(f"{name.replace('_', ' ')} ({INPUT_FLAGS[name]}): ").strip()
                except EOFError:
                    return
                if not answer:
                    return
                try:
                    overrides[name] = float(answer)
                    break
                except ValueError:
                    print("Please enter a number.")


def cmd_predict(args, ctx: AppContext) -> None:
    day = parse_date(args.date, "date") if args.date else date.today()
    if args.mode == "retrospective" and day > date.today():
        raise ValidationError("Retrospective predictions are only possible for days that have happened.")
    bundle = load_bundle(ctx.settings.models_dir, args.mode)
    overrides = {name: getattr(args, dest) for dest, name in OVERRIDE_DESTS.items()}
    today_raw, history = _db_inputs(ctx, day, args.mode)
    if interactive():
        _prompt_missing(args.mode, today_raw, overrides)

    result = predict_day(bundle, day, today_raw, history, overrides)
    band = f" (typical error +/-{result.test_mae:.1f})" if result.test_mae is not None else ""
    print(f"Predicted productivity score for {result.date} ({result.date:%A}): {result.score:.1f} / 100{band}")
    print(f"Model: {result.model} | mode: {result.mode}")
    print("\nInputs used:")
    print(format_kv([(k, v) for k, v in result.inputs.items()]))
    if result.imputed:
        print(f"\nNo history available for: {', '.join(result.imputed)}; "
              "the model used training-set medians instead.")
    if bundle.metadata.get("source") == "seed":
        print("\nNote: this model was trained on the synthetic seed dataset. Once you have logged enough "
              "days, retrain with `train --source combined` or `--source db`.")
    if args.mode == "planning" and day < date.today():
        print("Note: planning mode ignores anything that happened during that day.")
    print("This is a statistical estimate from productivity records, not an assessment of you.")
