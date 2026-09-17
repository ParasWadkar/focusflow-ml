"""`analytics` command."""

from __future__ import annotations

import argparse

import pandas as pd

from src.analytics import charts, reports
from src.cli.context import AppContext
from src.utils.formatting import format_kv, format_table, pct

VIEWS = ("summary", "daily", "weekly", "habits", "tasks", "focus", "model")


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("analytics", help="Productivity, habit, task and model statistics.",
                       description="Views: summary (default), daily, weekly, habits, tasks, focus, model.")
    p.add_argument("view", nargs="?", default="summary", choices=VIEWS, help="What to show (default: summary).")
    p.add_argument("--from", dest="start", help="Start date YYYY-MM-DD (default: 30 days before --to).")
    p.add_argument("--to", dest="end", help="End date YYYY-MM-DD (default: today).")
    p.add_argument("--plot", action="store_true", help="Also save charts to the reports folder.")
    p.set_defaults(func=cmd_analytics)


def cmd_analytics(args, ctx: AppContext) -> None:
    if args.view == "model":
        _model_view(ctx)
        return
    period = reports.resolve_period(args.start, args.end)
    print(f"Period: {period.start} to {period.end} ({period.days} days)\n")
    handler = {
        "summary": _summary_view,
        "daily": _daily_view,
        "weekly": _weekly_view,
        "habits": _habits_view,
        "tasks": _tasks_view,
        "focus": _focus_view,
    }[args.view]
    saved = handler(ctx, period, args.plot)
    for path in saved:
        print(f"Chart saved: {path}")


def _summary_view(ctx, period, plot) -> list:
    daily = reports.daily_table(ctx.conn, ctx.user_id, period)
    ov = reports.overview(daily)
    if ov["days_logged"] == 0:
        print("No data logged in this period. Try `init --demo-data` on an empty database, or start logging.")
        return []
    print(format_kv([
        ("days logged / scored", f"{ov['days_logged']} / {ov['days_scored']}"),
        ("avg productivity score", ov["avg_productivity_score"]),
        ("best day", ov["best_day"]),
        ("avg sleep (h)", ov["avg_sleep_hours"]),
        ("avg energy / mood", f"{_num(ov['avg_energy_level'])} / {_num(ov['avg_mood_score'])}"),
        ("task hours done / planned", f"{ov['completed_hours']:.1f} / {ov['planned_hours']:.1f}"),
        ("task completion", pct(ov["task_completion_ratio"])),
        ("habit completion", pct(ov["habit_completion_rate"])),
        ("study hours", ov["study_hours"]),
        ("deep-work hours", ov["deep_work_hours"]),
        ("interruptions", ov["total_interruptions"]),
    ]))
    saved = []
    if plot:
        saved.append(charts.plot_daily_scores(daily, ctx.settings.reports_dir / "daily_scores.png"))
        weekly = reports.weekly_table(daily)
        saved.append(charts.plot_weekly_hours(weekly, ctx.settings.reports_dir / "weekly_hours.png"))
    return saved


def _num(value) -> str:
    return "-" if value is None else f"{value:.1f}"


def _habit_ratio(done: float, planned: float) -> str:
    if pd.isna(planned):
        return "-"
    return f"?/{int(planned)}" if pd.isna(done) else f"{int(done)}/{int(planned)}"


def _ratio(value: float) -> str:
    return pct(None if pd.isna(value) else float(value))


def _daily_view(ctx, period, plot) -> list:
    daily = reports.daily_table(ctx.conn, ctx.user_id, period)
    if daily.empty:
        print("No data logged in this period.")
        return []
    print(format_table(
        ["date", "score", "sleep", "energy", "planned h", "done h", "done %", "deep h", "habits"],
        [[r.date, r.productivity_score, r.sleep_hours, r.energy_level, r.planned_task_hours,
          r.completed_task_hours, _ratio(r.task_completion),
          r.deep_work_hours, _habit_ratio(r.habits_completed, r.habits_planned)]
         for r in daily.itertuples()],
    ))
    if plot:
        return [charts.plot_daily_scores(daily, ctx.settings.reports_dir / "daily_scores.png")]
    return []


def _weekly_view(ctx, period, plot) -> list:
    weekly = reports.weekly_table(reports.daily_table(ctx.conn, ctx.user_id, period))
    if weekly.empty:
        print("No data logged in this period.")
        return []
    print(format_table(
        ["week of", "days", "avg score", "planned h", "done h", "done %", "study h", "deep h", "habits %", "sleep"],
        [[r.week_start, r.days_logged, r.avg_score, r.planned_hours, r.completed_hours,
          _ratio(r.completion_ratio),
          r.study_hours, r.deep_work_hours,
          _ratio(r.habit_rate), r.avg_sleep]
         for r in weekly.itertuples()],
    ))
    if plot:
        return [charts.plot_weekly_hours(weekly, ctx.settings.reports_dir / "weekly_hours.png")]
    return []


def _habits_view(ctx, period, plot) -> list:
    stats = reports.habit_stats(ctx.conn, ctx.user_id, period)
    if not stats:
        print("No habits yet.")
        return []
    print(format_table(
        ["habit", "scheduled", "completed", "rate", "current streak", "longest streak"],
        [[s.habit.name, s.scheduled_days, s.completed_days, pct(s.completion_rate),
          s.current_streak, s.longest_streak] for s in stats],
    ))
    rated = [s for s in stats if s.completion_rate is not None]
    if plot and rated:
        return [charts.plot_habit_rates([s.habit.name for s in rated], [s.completion_rate for s in rated],
                                        ctx.settings.reports_dir / "habit_completion.png")]
    return []


def _tasks_view(ctx, period, plot) -> list:
    s = reports.task_stats(ctx.conn, ctx.user_id, period)
    print(format_kv([
        ("blocks completed / total", f"{s.completed_count}/{s.block_count}"),
        ("planned hours", s.planned_hours),
        ("completed hours", s.completed_hours),
        ("completion ratio", pct(s.completion_ratio)),
    ]))
    if s.hours_by_category:
        print()
        print(format_table(["category", "planned h", "completed h", "done %"],
                           [[c, h, s.completed_hours_by_category.get(c, 0.0),
                             pct(s.completed_hours_by_category.get(c, 0.0) / h if h else None)]
                            for c, h in s.hours_by_category.items()]))
        if plot:
            return [charts.plot_category_hours(s.hours_by_category, s.completed_hours_by_category,
                                               ctx.settings.reports_dir / "category_hours.png")]
    return []


def _focus_view(ctx, period, plot) -> list:
    daily = reports.daily_table(ctx.conn, ctx.user_id, period)
    s = reports.task_stats(ctx.conn, ctx.user_id, period)
    pairs = [
        ("study hours done / planned", f"{s.study_hours:.2f} / {s.planned_study_hours:.2f}"),
        ("deep-work hours done / planned", f"{s.deep_work_hours:.2f} / {s.planned_deep_work_hours:.2f}"),
    ]
    if not daily.empty:
        best = daily.loc[daily["deep_work_hours"].idxmax()]
        pairs += [
            ("avg deep work per logged day (h)", float(daily["deep_work_hours"].mean())),
            ("avg study per logged day (h)", float(daily["study_hours"].mean())),
            ("longest deep-work day", f"{best['date']} ({best['deep_work_hours']:.2f} h)"),
        ]
        scored = daily.dropna(subset=["productivity_score"])
        if len(scored) >= 5 and scored["deep_work_hours"].std() > 0:
            corr = scored["deep_work_hours"].corr(scored["productivity_score"])
            pairs.append(("corr(deep work, score)", f"{corr:.2f} (association only, not causation)"))
    print(format_kv(pairs))
    if plot and s.hours_by_category:
        return [charts.plot_category_hours(s.hours_by_category, s.completed_hours_by_category,
                                           ctx.settings.reports_dir / "category_hours.png")]
    return []


def _model_view(ctx) -> None:
    summary = reports.model_summary(ctx.settings.models_dir)
    if not summary:
        print("No trained models. Run `python -m src.main train`.")
        return
    for mode, meta in summary.items():
        print(f"=== {mode} model (trained {meta.get('trained_at')}, source: {meta.get('source')}) ===")
        print(f"Selected: {meta.get('selected_model')} - {meta.get('selection_rule')}")
        print(f"Train {meta['train_period'][0]}..{meta['train_period'][1]} ({meta['n_train']} rows), "
              f"test {meta['test_period'][0]}..{meta['test_period'][1]} ({meta['n_test']} rows)")
        print(format_table(["model", "CV RMSE", "test MAE", "test RMSE", "test R2"],
                           [[n, meta["cv"][n]["rmse_mean"], m["mae"], m["rmse"], m["r2"]]
                            for n, m in meta["test_metrics"].items()]))
        print()
    print("Run `python -m src.main evaluate` for feature importance, error analysis and charts.")
