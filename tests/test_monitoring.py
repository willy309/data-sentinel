"""Tests for logging, stage timing and the run summary.

Logs are only worth having if they can be parsed by a machine later, so the formatter
tests assert real ``json.loads`` round trips rather than substring matches. The summary
tests pin the agreement metric hardest: it is the one number here that is easy to define
plausibly and wrongly.
"""

import io
import json
import logging

import pandas as pd
import pytest

from sentinel import monitoring


@pytest.fixture
def captured() -> io.StringIO:
    """Point the package logger at a buffer, and put it back afterwards."""
    stream = io.StringIO()
    monitoring.configure(stream=stream, level=logging.INFO)
    yield stream
    monitoring.logger.handlers = []


def records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def scored(flags: dict[str, list[bool]], labels: list[bool]) -> pd.DataFrame:
    """A minimal scored frame: just the label and the flag columns the summary reads."""
    return pd.DataFrame({"is_anomaly": labels, **{f"{k}_flag": v for k, v in flags.items()}})


def test_each_log_line_is_one_json_object(captured):
    monitoring.log("thing.happened", count=3, label="x")
    (line,) = records(captured)
    assert line["event"] == "thing.happened"
    assert line["level"] == "info"
    assert line["count"] == 3 and line["label"] == "x"
    assert "ts" in line


def test_unserialisable_fields_do_not_crash_the_run(captured):
    """A log line is never worth taking the pipeline down for."""
    monitoring.log("odd.value", when=pd.Timestamp("2025-01-01"))
    assert records(captured)[0]["when"].startswith("2025-01-01")


def test_configure_replaces_its_handler_rather_than_stacking(captured):
    monitoring.configure(stream=captured, level=logging.INFO)
    monitoring.log("once")
    assert len(records(captured)) == 1


def test_logging_is_silent_until_configured(capsys):
    monitoring.logger.handlers = []
    monitoring.log("should.not.appear")
    out = capsys.readouterr()
    assert "should.not.appear" not in out.out + out.err


def test_stage_records_a_duration_and_logs_it(captured):
    timings: dict[str, float] = {}
    with monitoring.stage("work", timings):
        pass
    assert timings["work"] >= 0.0
    assert records(captured)[0]["stage"] == "work"


def test_stage_records_a_duration_even_when_the_block_raises(captured):
    """How far a failing run got is exactly when the timing matters."""
    timings: dict[str, float] = {}
    with pytest.raises(RuntimeError), monitoring.stage("boom", timings):
        raise RuntimeError("nope")
    assert "boom" in timings


def test_summary_counts_rows_labels_and_flags():
    frame = scored(
        {"a": [True, False, False, True], "b": [True, True, False, False]},
        labels=[True, True, False, False],
    )
    summary = monitoring.run_summary(frame, ["a", "b"])
    assert summary["rows_processed"] == 4
    assert summary["anomalies_injected"] == 2
    assert summary["flagged_by_detector"] == {"a": 2, "b": 2}
    assert summary["flagged_by_any"] == 3
    assert summary["flagged_by_all"] == 1


def test_agreement_is_overlap_of_alerts_not_of_all_rows():
    """The whole point of the metric.

    Both detectors call almost every row normal, so row-by-row agreement here would be
    0.5 and in a real run ~0.99 — a number that is identical for every pair of detectors
    ever written. Overlap of what they actually alert on is 1 of 3.
    """
    frame = scored(
        {"a": [True, False, False, True], "b": [True, True, False, False]},
        labels=[True, True, False, False],
    )
    # Rounded to four places on the way out, so compare at that resolution.
    assert monitoring.run_summary(frame, ["a", "b"])["agreement"] == pytest.approx(1 / 3, abs=1e-4)


def test_identical_detectors_agree_completely():
    frame = scored({"a": [True, False], "b": [True, False]}, labels=[True, False])
    assert monitoring.run_summary(frame, ["a", "b"])["agreement"] == 1.0


def test_agreement_is_defined_when_nothing_is_flagged():
    """Two detectors that both flagged nothing are vacuously in complete agreement."""
    frame = scored({"a": [False, False], "b": [False, False]}, labels=[True, False])
    assert monitoring.run_summary(frame, ["a", "b"])["agreement"] == 1.0


def test_summary_is_json_serialisable():
    """It goes straight into a log line and an artifact, so numpy scalars would break it."""
    frame = scored({"a": [True, False]}, labels=[True, False])
    json.dumps(monitoring.run_summary(frame, ["a"]))
