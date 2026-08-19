"""Command line entry point: ``python -m sentinel run``."""

import argparse
from collections.abc import Sequence

from sentinel.pipeline import format_summary, run


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, execute the requested command, and return an exit code."""
    parser = argparse.ArgumentParser(
        prog="python -m sentinel",
        description="Run the anomaly detection pipeline over synthetic transactions.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run_command = commands.add_parser("run", help="generate data, detect, and score")
    run_command.add_argument("--n", type=int, default=50_000, help="transactions to generate")
    run_command.add_argument("--seed", type=int, default=42, help="seed for the whole run")
    run_command.add_argument("--users", type=int, default=500, help="distinct cardholders")

    args = parser.parse_args(argv)
    print(format_summary(run(n=args.n, seed=args.seed, n_users=args.users)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
