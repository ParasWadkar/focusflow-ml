"""`habit` command group."""

from __future__ import annotations

import argparse

from src.cli.context import AppContext
from src.cli.prompts import confirm, require
from src.habits.models import SCHEDULES
from src.habits.service import HabitService
from src.utils.formatting import format_table, pct


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("habit", help="Create, track and analyse habits.")
    hs = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    a = hs.add_parser("add", help="Create a habit.")
    a.add_argument("name", nargs="?", help="Habit name (prompted if omitted in a terminal).")
    a.add_argument("--description", help="Optional description.")
    a.add_argument("--schedule", default="daily", choices=SCHEDULES, help="Days the habit is planned (default: daily).")
    a.add_argument("--start-date", help="First tracked date, YYYY-MM-DD (default: today). Use a past date to backfill.")
    a.set_defaults(func=cmd_add)

    ls = hs.add_parser("list", help="List habits.")
    ls.add_argument("--all", action="store_true", help="Include archived habits.")
    ls.set_defaults(func=cmd_list)

    e = hs.add_parser("edit", help="Edit a habit.")
    e.add_argument("habit", help="Habit id or name.")
    e.add_argument("--name")
    e.add_argument("--description")
    e.add_argument("--schedule", choices=SCHEDULES)
    e.add_argument("--start-date", help="YYYY-MM-DD")
    arch = e.add_mutually_exclusive_group()
    arch.add_argument("--archive", dest="archived", action="store_const", const=True, help="Hide from active lists.")
    arch.add_argument("--unarchive", dest="archived", action="store_const", const=False)
    e.set_defaults(func=cmd_edit, archived=None)

    d = hs.add_parser("delete", help="Delete a habit and all its history.")
    d.add_argument("habit", help="Habit id or name.")
    d.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    d.set_defaults(func=cmd_delete)

    for name, completed, text in (("done", True, "Mark a habit completed for a date."),
                                  ("missed", False, "Explicitly mark a habit as not completed for a date.")):
        m = hs.add_parser(name, help=text)
        m.add_argument("habit", help="Habit id or name.")
        m.add_argument("--date", help="YYYY-MM-DD (default: today). Past dates allowed.")
        m.add_argument("--note", help="Optional note.")
        m.set_defaults(func=cmd_mark, completed=completed)

    u = hs.add_parser("undo", help="Remove the record for a date (back to unrecorded).")
    u.add_argument("habit", help="Habit id or name.")
    u.add_argument("--date", help="YYYY-MM-DD (default: today).")
    u.set_defaults(func=cmd_undo)

    h = hs.add_parser("history", help="Show day-by-day history for a habit.")
    h.add_argument("habit", help="Habit id or name.")
    h.add_argument("--from", dest="start", help="Start date YYYY-MM-DD (default: habit start).")
    h.add_argument("--to", dest="end", help="End date YYYY-MM-DD (default: today).")
    h.set_defaults(func=cmd_history)

    st = hs.add_parser("stats", help="Completion rates and streaks.")
    st.add_argument("habit", nargs="?", help="Habit id or name (default: all habits).")
    st.add_argument("--from", dest="start", help="Start date YYYY-MM-DD.")
    st.add_argument("--to", dest="end", help="End date YYYY-MM-DD (default: today).")
    st.set_defaults(func=cmd_stats)


def _svc(ctx: AppContext) -> HabitService:
    return HabitService(ctx.conn, ctx.user_id)


def cmd_add(args, ctx: AppContext) -> None:
    name = require(args.name, "Habit name", "Habit name")
    habit = _svc(ctx).add(name, args.description, args.schedule, args.start_date)
    print(f"Created habit #{habit.id} '{habit.name}' ({habit.schedule}, from {habit.start_date}).")


def cmd_list(args, ctx: AppContext) -> None:
    habits = _svc(ctx).list(include_archived=args.all)
    if not habits:
        print("No habits yet. Add one with `habit add \"Read 20 pages\"`.")
        return
    print(format_table(["id", "name", "schedule", "since", "archived", "description"],
                       [[h.id, h.name, h.schedule, h.start_date, h.archived, h.description] for h in habits]))


def cmd_edit(args, ctx: AppContext) -> None:
    habit = _svc(ctx).edit(args.habit, name=args.name, description=args.description, schedule=args.schedule,
                           start_date=args.start_date, archived=args.archived)
    print(f"Updated habit #{habit.id} '{habit.name}'.")


def cmd_delete(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    habit = svc.get(args.habit)
    confirm(f"Delete habit '{habit.name}' and all of its history?", args.yes)
    svc.delete(habit.id)
    print(f"Deleted habit #{habit.id} '{habit.name}'.")


def cmd_mark(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    result = svc.mark(args.habit, args.date, completed=args.completed, note=args.note)
    habit = svc.get(args.habit)
    state = "completed" if result.completed else "not completed"
    extra = "" if result.scheduled else " (note: not a scheduled day for this habit)"
    print(f"'{habit.name}' marked {state} on {result.date}{extra}.")


def cmd_undo(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    habit = svc.get(args.habit)
    if svc.clear(habit.id, args.date):
        print(f"Removed the record for '{habit.name}'.")
    else:
        print(f"'{habit.name}' had no record for that date.")


def cmd_history(args, ctx: AppContext) -> None:
    days = _svc(ctx).history(args.habit, args.start, args.end)
    rows = []
    for d in days:
        status = "done" if d.completed else ("missed" if d.scheduled else "-")
        if not d.recorded and d.scheduled:
            status = "missed (unrecorded)"
        rows.append([d.date, d.date.strftime("%a"), "yes" if d.scheduled else "no", status, d.note])
    print(format_table(["date", "day", "scheduled", "status", "note"], rows) if rows else "No days in range.")


def cmd_stats(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    stats = [svc.stats(args.habit, args.start, args.end)] if args.habit else svc.all_stats(args.start, args.end)
    if not stats:
        print("No habits yet.")
        return
    print(format_table(
        ["habit", "from", "to", "scheduled", "completed", "rate", "current streak", "longest streak"],
        [[s.habit.name, s.start, s.end, s.scheduled_days, s.completed_days, pct(s.completion_rate),
          s.current_streak, s.longest_streak] for s in stats],
    ))
