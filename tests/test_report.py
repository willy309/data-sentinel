"""Tests for the console summary and the JSON artifact.

The artifact is the half that outlives the terminal, so most of these guard the promises
it makes to a future reader: that it is genuinely serialisable, that a truncated alert
list still records how many alerts there really were, and that writing twice never
destroys the earlier evidence.
"""

import json

import pytest

from sentinel import report
from sentinel.pipeline import run

SMALL = 4_000


@pytest.fixture(scope="module")
def result():
    return run(n=SMALL, seed=7, n_users=50)


@pytest.fixture(scope="module")
def built(result):
    return report.build(result)


def test_report_is_json_serialisable(built):
    """numpy scalars and Timestamps are the usual way an artifact silently fails."""
    assert json.loads(json.dumps(built))["params"]["seed"] == 7


def test_report_records_what_produced_it(built):
    assert built["params"]["n"] == SMALL
    assert built["generated_at"].startswith("20")
    assert set(built["timings_ms"]) == {"generate", "features", "detect", "score"}
    assert set(built["metrics"]) == {"zscore", "isoforest"}


def test_report_carries_the_summary_counts(built):
    assert built["summary"]["rows_processed"] == SMALL
    assert 0.0 <= built["summary"]["agreement"] <= 1.0


def test_alerts_are_capped_but_the_true_total_is_kept(built):
    assert built["alerts_included"] == min(built["alerts_total"], report.MAX_ALERTS)
    assert len(built["alerts"]) == built["alerts_included"]
    # A cap that hid the real number would read as "these are all the alerts".
    assert built["alerts_total"] >= built["alerts_included"]


def test_every_alert_names_the_detectors_that_raised_it(built):
    assert built["alerts"], "the run should produce at least one alert"
    for alert in built["alerts"]:
        assert alert["detectors"], "an alert nobody raised should not be in the list"
        assert set(alert["detectors"]) <= {"zscore", "isoforest"}
        assert set(report.ALERT_FIELDS) <= set(alert)


def test_alerts_are_ranked_so_the_worst_is_first(built):
    scores = [alert["amount_zscore"] for alert in built["alerts"]]
    assert scores == sorted(scores, reverse=True)


def test_write_creates_the_directory_and_names_the_seed(built, tmp_path):
    path = report.write(built, tmp_path / "nested" / "reports")
    assert path.exists()
    assert "seed7" in path.name
    assert json.loads(path.read_text())["params"]["seed"] == 7


def test_writing_twice_keeps_both_runs(built, tmp_path):
    """The stamp only resolves to the second, so this would silently overwrite."""
    first = report.write(built, tmp_path)
    second = report.write(built, tmp_path)
    assert first != second
    assert first.exists() and second.exists()


def test_console_summary_reports_the_numbers(result):
    text = report.format_console(result)
    assert f"{SMALL:,} transactions" in text
    assert "seed 7" in text
    assert "zscore" in text and "isoforest" in text
    assert "recall by anomaly type" in text
    assert "detector agreement" in text
    assert "stage timings" in text


def test_console_summary_names_the_blind_spot(result):
    """Neither detector reads the clock, so the summary has to say so out loud."""
    assert "caught by no detector: odd_hours" in report.format_console(result)
