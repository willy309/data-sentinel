"""Structured logging, stage timing, and the run summary.

Everything here exists to answer questions about a run *after* it has finished, from a
log file, without a debugger and without rerunning it. That constraint is what makes the
logs JSON rather than prose: one object per line survives grep, jq, and whatever ships
them somewhere central, whereas a nicely worded sentence has to be re-parsed by hand the
first time anyone wants to plot stage durations.

Logging is off by default. A library that writes to stderr the moment it is imported is
a library that has decided something on its caller's behalf, so the CLI opts in and
importers stay quiet.
"""

import json
import logging
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TextIO

import pandas as pd

LOGGER_NAME = "sentinel"
logger = logging.getLogger(LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Render each record as one JSON object on one line.

    Anything passed as a keyword to :func:`log` arrives here under ``fields`` and is
    merged into the top level, so a stage duration is queryable as ``.duration_ms``
    rather than buried in a message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def configure(*, stream: TextIO | None = None, level: int = logging.INFO) -> None:
    """Send this package's structured logs to ``stream`` (default stderr).

    Replaces its own handler rather than adding one, so calling it twice — in a test,
    say — does not double every line.
    """
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(level)
    # Stay out of the root logger; an application that configures its own logging should
    # not suddenly find sentinel's records duplicated into it.
    logger.propagate = False


def log(event: str, **fields: object) -> None:
    """Emit one structured event. A no-op unless :func:`configure` has been called."""
    logger.info(event, extra={"fields": fields})


@contextmanager
def stage(name: str, timings: dict[str, float]) -> Iterator[None]:
    """Time a block of work, record it in ``timings``, and log its completion.

    The duration is written in a ``finally``, so a stage that raises still leaves its
    timing behind — which is exactly when you want to know how far it got.
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        timings[name] = round((time.perf_counter() - start) * 1000, 1)
        log("stage.complete", stage=name, duration_ms=timings[name])


def run_summary(frame: pd.DataFrame, detectors: Sequence[str]) -> dict[str, Any]:
    """Headline counts for one run, including how far the detectors agree.

    Args:
        frame: A scored frame carrying ``is_anomaly`` and one ``<name>_flag`` per
            detector.
        detectors: Detector names, in report order.

    Returns:
        A JSON-safe dict of plain Python numbers, ready for a log line or an artifact.
    """
    flags = {name: frame[f"{name}_flag"] for name in detectors}
    any_flag = pd.concat(flags.values(), axis=1).any(axis=1)
    all_flags = pd.concat(flags.values(), axis=1).all(axis=1)

    union, intersection = int(any_flag.sum()), int(all_flags.sum())
    return {
        "rows_processed": int(len(frame)),
        "anomalies_injected": int(frame["is_anomaly"].sum()),
        "flagged_by_detector": {name: int(flag.sum()) for name, flag in flags.items()},
        "flagged_by_any": union,
        "flagged_by_all": intersection,
        # Jaccard, not raw row-by-row agreement. Both detectors call ~99% of rows normal,
        # so plain agreement is ~0.99 for any two detectors ever written and tells you
        # nothing. Overlap of what they actually alert on is the number worth watching:
        # near 1 means the second detector is redundant, near 0 means they are covering
        # different fraud and both earn their keep.
        "agreement": round(intersection / union, 4) if union else 1.0,
    }
