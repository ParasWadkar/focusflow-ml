"""FocusFlow ML command-line entry point: ``python -m src.main --help``."""

from __future__ import annotations

import sys
import traceback
from typing import Sequence

from src.cli.context import AppContext
from src.cli.parser import build_parser
from src.config import Settings, load_settings
from src.utils.errors import FocusFlowError


def _safe_console() -> None:
    """Avoid crashes on consoles whose encoding lacks symbols such as '±'."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")


def main(argv: Sequence[str] | None = None, settings: Settings | None = None) -> int:
    _safe_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    ctx = None
    try:
        ctx = AppContext(settings or load_settings())
        args.func(args, ctx)
        return 0
    except FocusFlowError as exc:
        if args.debug:
            traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    finally:
        if ctx is not None:
            ctx.close()


if __name__ == "__main__":
    sys.exit(main())
