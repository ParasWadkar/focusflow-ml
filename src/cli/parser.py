"""Top-level argument parser."""

from __future__ import annotations

import argparse

from src.cli import analytics_cmds, block_cmds, day_cmds, habit_cmds, ml_cmds, setup_cmds

EPILOG = """examples:
  python -m src.main init
  python -m src.main habit add "Read 20 pages" --schedule daily
  python -m src.main habit done "Read 20 pages" --date 2026-09-15
  python -m src.main block add "Thesis chapter" --start 09:00 --end 11:00 --category deep_work
  python -m src.main day record --sleep 7.5 --energy 7
  python -m src.main train
  python -m src.main evaluate
  python -m src.main predict
  python -m src.main analytics weekly --plot
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.main",
        description="FocusFlow ML: habits, time blocking, daily records and "
                    "productivity-score prediction, all from the command line.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--debug", action="store_true", help="Show full tracebacks on errors.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for module in (setup_cmds, habit_cmds, block_cmds, day_cmds, analytics_cmds, ml_cmds):
        module.register(sub)
    return parser
