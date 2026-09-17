"""`block` command group (time blocking)."""

from __future__ import annotations

import argparse
from datetime import date

from src.cli.context import AppContext
from src.cli.prompts import confirm, require
from src.planner.models import CATEGORIES, PRIORITIES
from src.planner.service import PlannerService
from src.utils.formatting import format_kv, format_table, pct


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("block", help="Plan and track time blocks.")
    bs = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    a = bs.add_parser("add", help="Add a time block.")
    a.add_argument("task", nargs="?", help="Task name (prompted if omitted in a terminal).")
    a.add_argument("--date", help="YYYY-MM-DD (default: today). Future dates allowed for planning.")
    a.add_argument("--start", help="Start time HH:MM (24h).")
    a.add_argument("--end", help="End time HH:MM (24h), after the start time.")
    a.add_argument("--category", default="other", choices=CATEGORIES, help="Category (default: other).")
    a.add_argument("--priority", type=int, default=2, choices=sorted(PRIORITIES),
                   help="1=high, 2=medium, 3=low (default: 2).")
    a.add_argument("--completed", action="store_true", help="Record the block as already completed.")
    a.set_defaults(func=cmd_add)

    ls = bs.add_parser("list", help="List blocks for a date or date range.")
    _range_args(ls)
    ls.set_defaults(func=cmd_list)

    sh = bs.add_parser("show", help="Show one block.")
    sh.add_argument("id", type=int)
    sh.set_defaults(func=cmd_show)

    e = bs.add_parser("edit", help="Modify a block.")
    e.add_argument("id", type=int)
    e.add_argument("--task")
    e.add_argument("--date")
    e.add_argument("--start")
    e.add_argument("--end")
    e.add_argument("--category", choices=CATEGORIES)
    e.add_argument("--priority", type=int, choices=sorted(PRIORITIES))
    e.set_defaults(func=cmd_edit)

    dn = bs.add_parser("done", help="Mark a block completed.")
    dn.add_argument("id", type=int)
    dn.add_argument("--undo", action="store_true", help="Mark as not completed instead.")
    dn.set_defaults(func=cmd_done)

    d = bs.add_parser("delete", help="Delete a block.")
    d.add_argument("id", type=int)
    d.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    d.set_defaults(func=cmd_delete)

    st = bs.add_parser("stats", help="Planned/completed hours, study and deep-work hours, hours by category.")
    _range_args(st)
    st.set_defaults(func=cmd_stats)


def _range_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--date", help="Single date YYYY-MM-DD (default: today unless a range is given).")
    p.add_argument("--from", dest="start", help="Range start YYYY-MM-DD.")
    p.add_argument("--to", dest="end", help="Range end YYYY-MM-DD.")


def _range(args) -> tuple[str | date | None, str | date | None]:
    if args.date:
        return args.date, args.date
    if args.start or args.end:
        return args.start, args.end
    return date.today(), date.today()


def _svc(ctx: AppContext) -> PlannerService:
    return PlannerService(ctx.conn, ctx.user_id)


def _describe(b) -> str:
    return (f"#{b.id} {b.date} {b.start_time:%H:%M}-{b.end_time:%H:%M} '{b.task_name}' "
            f"[{b.category}, {PRIORITIES[b.priority]} priority]")


def cmd_add(args, ctx: AppContext) -> None:
    task = require(args.task, "Task name", "Task name")
    start = require(args.start, "--start", "Start time (HH:MM)")
    end = require(args.end, "--end", "End time (HH:MM)")
    block = _svc(ctx).add(args.date or date.today(), start, end, task, args.category, args.priority,
                          completed=args.completed)
    print(f"Added block {_describe(block)} ({block.duration_hours:.2f} h).")


def cmd_list(args, ctx: AppContext) -> None:
    start, end = _range(args)
    blocks = _svc(ctx).list(start, end)
    if not blocks:
        print("No time blocks in range.")
        return
    print(format_table(
        ["id", "date", "start", "end", "hours", "task", "category", "priority", "done"],
        [[b.id, b.date, f"{b.start_time:%H:%M}", f"{b.end_time:%H:%M}", b.duration_hours, b.task_name,
          b.category, PRIORITIES[b.priority], b.completed] for b in blocks],
    ))


def cmd_show(args, ctx: AppContext) -> None:
    b = _svc(ctx).get(args.id)
    print(format_kv([("id", b.id), ("date", b.date), ("time", f"{b.start_time:%H:%M}-{b.end_time:%H:%M}"),
                     ("duration (h)", b.duration_hours), ("task", b.task_name), ("category", b.category),
                     ("priority", PRIORITIES[b.priority]), ("completed", b.completed)]))


def cmd_edit(args, ctx: AppContext) -> None:
    block = _svc(ctx).edit(args.id, day=args.date, start=args.start, end=args.end, task_name=args.task,
                           category=args.category, priority=args.priority)
    print(f"Updated block {_describe(block)}.")


def cmd_done(args, ctx: AppContext) -> None:
    block = _svc(ctx).set_completed(args.id, not args.undo)
    print(f"Block {_describe(block)} marked {'completed' if block.completed else 'not completed'}.")


def cmd_delete(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    block = svc.get(args.id)
    confirm(f"Delete block {_describe(block)}?", args.yes)
    svc.delete(block.id)
    print(f"Deleted block #{block.id}.")


def cmd_stats(args, ctx: AppContext) -> None:
    start, end = _range(args)
    s = _svc(ctx).stats(start, end)
    print(format_kv([
        ("blocks (completed/total)", f"{s.completed_count}/{s.block_count}"),
        ("planned hours", s.planned_hours),
        ("completed hours", s.completed_hours),
        ("completion ratio", pct(s.completion_ratio)),
        ("study hours (done/planned)", f"{s.study_hours:.2f} / {s.planned_study_hours:.2f}"),
        ("deep-work hours (done/planned)", f"{s.deep_work_hours:.2f} / {s.planned_deep_work_hours:.2f}"),
    ]))
    if s.hours_by_category:
        print()
        print(format_table(["category", "planned h", "completed h"],
                           [[c, h, s.completed_hours_by_category.get(c, 0.0)]
                            for c, h in s.hours_by_category.items()]))
