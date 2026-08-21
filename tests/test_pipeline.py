"""Tests for the end-to-end run and its CLI.

The pipeline's job is to be honest, so most of these assert that the scoring says what
the data says: metrics that match a direct call to the detector, a per-type breakdown
that covers every injected type, and a summary that names the anomaly nothing catches
rather than quietly averaging it away.
"""

import pandas as pd
import pytest

from sentinel.__main__ import main
from sentinel.detectors import zscore
from sentinel.generate import ANOMALY_TYPES
from sentinel.pipeline import METRIC_COLUMNS, run

SMALL = 8_000


@pytest.fixture(scope="module")
def result():
    return run(n=SMALL, seed=7, n_users=50)


def test_frame_carries_features_and_a_flag_per_detector(result):
    assert len(result.frame) == SMALL
    for column in ("amount_zscore", "txn_count_1h", "zscore_flag", "isoforest_flag"):
        assert column in result.frame.columns
    assert result.frame["zscore_flag"].dtype == bool
    assert result.frame["isoforest_flag"].dtype == bool


def test_metrics_cover_every_detector(result):
    assert list(result.metrics.columns) == list(METRIC_COLUMNS)
    assert set(result.metrics.index) == {"zscore", "isoforest"}
    scores = result.metrics[["precision", "recall", "f1"]]
    assert scores.ge(0.0).all().all() and scores.le(1.0).all().all()
    assert (result.metrics["flagged"] > 0).all()


def test_metrics_agree_with_calling_the_detector_directly(result):
    """The pipeline must not quietly score something other than what it ran."""
    direct = zscore.detect(result.frame)
    pd.testing.assert_series_equal(direct, result.frame["zscore_flag"], check_names=False)
    assert result.metrics.loc["zscore", "flagged"] == direct.sum()


def test_precision_matches_the_flag_column(result):
    frame = result.frame
    for name in ("zscore", "isoforest"):
        flagged = frame[frame[f"{name}_flag"]]
        assert result.metrics.loc[name, "precision"] == pytest.approx(flagged["is_anomaly"].mean())


def test_recall_is_broken_down_over_every_injected_type(result):
    assert set(result.recall_by_type.index) == set(ANOMALY_TYPES)
    assert set(result.recall_by_type.columns) == {"zscore", "isoforest"}
    assert result.recall_by_type.ge(0.0).all().all()
    assert result.recall_by_type.le(1.0).all().all()


def test_the_two_detectors_are_complementary(result):
    """The point of running both: each covers a type the other misses entirely."""
    by_type = result.recall_by_type
    assert by_type.loc["amount_spike", "zscore"] > 0.4
    assert by_type.loc["velocity_burst", "zscore"] == 0.0
    assert by_type.loc["velocity_burst", "isoforest"] > 0.4


def test_runs_are_reproducible():
    pd.testing.assert_frame_equal(
        run(n=2_000, seed=5, n_users=20).metrics,
        run(n=2_000, seed=5, n_users=20).metrics,
    )


def test_cli_runs_and_prints_a_summary(capsys):
    assert main(["run", "--n", "2000", "--seed", "1", "--users", "20"]) == 0
    assert "transactions" in capsys.readouterr().out


def test_cli_requires_a_command():
    with pytest.raises(SystemExit):
        main([])
