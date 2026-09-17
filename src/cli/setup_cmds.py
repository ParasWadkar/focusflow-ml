"""`init` and `dataset` commands."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.cli.context import AppContext
from src.database.connection import get_user_id, initialize
from src.demo import seed_demo
from src.ml.cleaning import clean
from src.ml.dataset import DEFAULT_ROWS, DEFAULT_SEED, generate_dataset, load_dataset, save_dataset
from src.ml.validation import validate_raw
from src.utils.errors import ValidationError
from src.utils.formatting import format_kv


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Create the database and folders, and generate the seed dataset if missing.",
                       description="Initialise FocusFlow ML. Safe to run more than once.")
    p.add_argument("--demo-data", action="store_true",
                   help="Also fill an EMPTY database with synthetic demo history (marked [demo]).")
    p.add_argument("--demo-days", type=int, default=60, help="Days of demo history (default: 60).")
    p.set_defaults(func=cmd_init)

    ds = sub.add_parser("dataset", help="Generate, validate or inspect the synthetic seed dataset.")
    dsub = ds.add_subparsers(dest="action", required=True, metavar="ACTION")
    g = dsub.add_parser("generate", help="Generate the synthetic seed dataset CSV.")
    g.add_argument("--rows", type=int, default=DEFAULT_ROWS, help=f"Number of days (default: {DEFAULT_ROWS}).")
    g.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED}).")
    g.add_argument("--output", type=Path, help="Output path (default: configured dataset path).")
    g.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    g.set_defaults(func=cmd_dataset_generate)
    for name, func, text in (("validate", cmd_dataset_validate, "Validate a dataset CSV."),
                             ("info", cmd_dataset_info, "Show dataset statistics.")):
        sp = dsub.add_parser(name, help=text)
        sp.add_argument("--path", type=Path, help="Dataset path (default: configured dataset path).")
        sp.set_defaults(func=func)


def cmd_init(args: argparse.Namespace, ctx: AppContext) -> None:
    s = ctx.settings
    s.ensure_dirs()
    conn = initialize(s.db_path, s.user_name)
    try:
        print(f"Database ready: {s.db_path}")
        if s.dataset_path.exists():
            print(f"Seed dataset found: {s.dataset_path}")
        else:
            save_dataset(generate_dataset(DEFAULT_ROWS, DEFAULT_SEED), s.dataset_path)
            print(f"Generated synthetic seed dataset ({DEFAULT_ROWS} days, seed {DEFAULT_SEED}): {s.dataset_path}")
        if args.demo_data:
            summary = seed_demo(conn, get_user_id(conn, s.user_name), days=args.demo_days)
            print(f"Added synthetic demo data: {summary.days} past days + today "
                  f"({summary.start} to {summary.end}), {summary.habits} habits, {summary.blocks} time blocks.")
        print("Next: `python -m src.main train`, then `python -m src.main evaluate` and `predict`.")
    finally:
        conn.close()


def cmd_dataset_generate(args: argparse.Namespace, ctx: AppContext) -> None:
    path = args.output or ctx.settings.dataset_path
    if path.exists() and not args.force:
        raise ValidationError(f"{path} already exists. Use --force to overwrite it.")
    if args.rows < 50:
        raise ValidationError("--rows must be at least 50 for a useful dataset.")
    df = generate_dataset(args.rows, args.seed)
    save_dataset(df, path)
    print(f"Wrote {len(df)} synthetic rows (seed {args.seed}) to {path}")


def cmd_dataset_validate(args: argparse.Namespace, ctx: AppContext) -> None:
    path = args.path or ctx.settings.dataset_path
    df = load_dataset(path)
    report = validate_raw(df)
    _, summary = clean(df)
    print(f"{path}: {report.rows} rows, structurally valid.")
    print(f"Rows usable for training (with a score): {summary.rows_out}")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    if report.ok:
        print("No issues found.")


def cmd_dataset_info(args: argparse.Namespace, ctx: AppContext) -> None:
    path = args.path or ctx.settings.dataset_path
    df = load_dataset(path)
    validate_raw(df)
    cleaned, _ = clean(df, require_target=False)
    print(format_kv([
        ("path", path),
        ("rows", len(df)),
        ("date range", f"{cleaned['date'].min():%Y-%m-%d} to {cleaned['date'].max():%Y-%m-%d}"),
        ("missing values", int(cleaned.drop(columns='date').isna().sum().sum())),
    ]))
    print()
    stats = cleaned.drop(columns="date").describe().T[["count", "mean", "std", "min", "max"]]
    print(stats.round(2).to_string())
