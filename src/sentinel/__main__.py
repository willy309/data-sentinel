"""Command line entry point: ``python -m sentinel run``."""

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from sentinel import monitoring, report
from sentinel.pipeline import run


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the requested command, and return an exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m sentinel",
        description="Run the anomaly detection pipeline over synthetic transactions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="generate data, detect, score, report")
    run_command.add_argument("--n", type=int, default=50_000, help="transactions to generate")
    run_command.add_argument("--seed", type=int, default=42, help="seed for the whole run")
    run_command.add_argument("--users", type=int, default=500, help="distinct cardholders")
    run_command.add_argument(
        "--report-dir",
        type=Path,
        default=report.DEFAULT_REPORT_DIR,
        help="where the JSON artifact is written",
    )
    run_command.add_argument(
        "--no-report", action="store_true", help="print the summary without writing an artifact"
    )
    run_command.add_argument(
        "--verbose", action="store_true", help="emit structured JSON logs to stderr"
    )

    args = parser.parse_args(argv)
    if args.verbose:
        monitoring.configure(level=logging.INFO)

    result = run(n=args.n, seed=args.seed, n_users=args.users)
    print(report.format_console(result))
    if not args.no_report:
        print(f"\nreport written to {report.write(report.build(result), args.report_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
