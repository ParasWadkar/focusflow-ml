"""`day` command group (daily records and journal)."""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.cli.context import AppContext
from src.cli.prompts import confirm, require
from src.habits.service import HabitService
from src.journal.models import ALL_FIELDS
from src.journal.service import JournalService
from src.planner.service import PlannerService
from src.utils.errors import ValidationError
from src.utils.formatting import format_kv, format_table
from src.utils.validators import parse_date

FLAG_TO_FIELD = {
    "sleep": "sleep_hours",
    "energy": "energy_level",
    "mood": "mood_score",
    "exercise": "exercise_minutes",
    "interruptions": "interruptions",
    "score": "productivity_score",
}


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("day", help="Record and review daily records and the journal.")
    ds = p.add_subparsers(dest="action", required=True, metavar="ACTION")

    r = ds.add_parser("record", help="Create or update a day's record (only the given fields change).",
                      description="Create or update the record for a date. Fill it in gradually, "
                                  "e.g. sleep and energy in the morning and the score in the evening.")
    r.add_argument("--date", help="YYYY-MM-DD (default: today). Past dates can be edited.")
    r.add_argument("--sleep", type=float, help="Hours slept last night (0-24).")
    r.add_argument("--energy", type=int, help="Morning energy level (1-10).")
    r.add_argument("--mood", type=int, help="Mood score for the day (1-10).")
    r.add_argument("--exercise", type=float, help="Exercise minutes (>= 0).")
    r.add_argument("--interruptions", type=int, help="Number of interruptions (>= 0).")
    r.add_argument("--score", type=float, help="Your productivity score for the day (0-100).")
    r.add_argument("--journal", help="Journal text (replaces the existing text).")
    r.add_argument("--journal-file", type=Path, help="Read journal text from a UTF-8 file.")
    r.add_argument("--clear", nargs="+", choices=ALL_FIELDS, metavar="FIELD", default=[],
                   help=f"Reset fields to empty. Fields: {', '.join(ALL_FIELDS)}.")
    r.set_defaults(func=cmd_record)

    s = ds.add_parser("show", help="Show a day's record, time blocks summary and photos.")
    s.add_argument("--date", help="YYYY-MM-DD (default: today).")
    s.set_defaults(func=cmd_show)

    ls = ds.add_parser("list", help="List daily records.")
    ls.add_argument("--from", dest="start", help="Start date YYYY-MM-DD.")
    ls.add_argument("--to", dest="end", help="End date YYYY-MM-DD.")
    ls.set_defaults(func=cmd_list)

    d = ds.add_parser("delete", help="Delete a day's record (and its photo references).")
    d.add_argument("--date", required=True, help="YYYY-MM-DD")
    d.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    d.set_defaults(func=cmd_delete)

    ph = ds.add_parser("photo-add", help="Attach a local photo to a day (the file is copied into the data folder).")
    ph.add_argument("path", nargs="?", type=Path, help="Image file to attach.")
    ph.add_argument("--date", help="YYYY-MM-DD (default: today).")
    ph.add_argument("--caption", help="Optional caption.")
    ph.set_defaults(func=cmd_photo_add)


def _svc(ctx: AppContext) -> JournalService:
    return JournalService(ctx.conn, ctx.user_id, ctx.settings.photos_dir)


def cmd_record(args, ctx: AppContext) -> None:
    fields = {field: getattr(args, flag) for flag, field in FLAG_TO_FIELD.items()}
    if args.journal is not None and args.journal_file is not None:
        raise ValidationError("Use either --journal or --journal-file, not both.")
    if args.journal_file is not None:
        try:
            fields["journal"] = args.journal_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(f"Could not read journal file {args.journal_file}: {exc}") from exc
    else:
        fields["journal"] = args.journal
    record = _svc(ctx).record(args.date, clear=args.clear, **fields)
    print(f"Saved daily record for {record.date}.")
    _print_record(record)


def _print_record(record) -> None:
    print(format_kv([
        ("date", f"{record.date} ({record.date:%A})"),
        ("sleep hours", record.sleep_hours),
        ("energy (1-10)", record.energy_level),
        ("mood (1-10)", record.mood_score),
        ("exercise minutes", record.exercise_minutes),
        ("interruptions", record.interruptions),
        ("productivity score", record.productivity_score),
    ]))
    if record.journal:
        print("\nJournal:\n" + record.journal)
    if record.photos:
        print("\nPhotos:")
        for photo in record.photos:
            print(f"  - {photo.file_path}" + (f" ({photo.caption})" if photo.caption else ""))


def cmd_show(args, ctx: AppContext) -> None:
    target = parse_date(args.date, "date") if args.date else date.today()
    record = _svc(ctx).find(target)
    if record:
        _print_record(record)
    else:
        print(f"No daily record for {target}.")
    stats = PlannerService(ctx.conn, ctx.user_id).stats(target, target)
    planned, completed = HabitService(ctx.conn, ctx.user_id).day_summary(target)
    print(f"\nTime blocks: {stats.completed_hours:.2f} of {stats.planned_hours:.2f} planned hours completed "
          f"({stats.block_count} blocks). Habits: "
          + (f"{completed}/{planned} completed." if completed is not None else f"{planned} scheduled, none recorded yet."))


def cmd_list(args, ctx: AppContext) -> None:
    records = _svc(ctx).list(args.start, args.end)
    if not records:
        print("No daily records in range.")
        return
    print(format_table(
        ["date", "sleep", "energy", "mood", "exercise", "interrupts", "score", "journal"],
        [[r.date, r.sleep_hours, r.energy_level, r.mood_score, r.exercise_minutes, r.interruptions,
          r.productivity_score, (r.journal or "")[:30]] for r in records],
    ))


def cmd_delete(args, ctx: AppContext) -> None:
    svc = _svc(ctx)
    record = svc.get(args.date)
    confirm(f"Delete the daily record for {record.date}?", args.yes)
    svc.delete(record.date)
    print(f"Deleted the daily record for {record.date}.")


def cmd_photo_add(args, ctx: AppContext) -> None:
    path = require(str(args.path) if args.path else None, "Photo path", "Path to image")
    photo = _svc(ctx).add_photo(args.date or date.today(), path, args.caption)
    print(f"Attached photo #{photo.id}: {ctx.settings.photos_dir / photo.file_path}")
