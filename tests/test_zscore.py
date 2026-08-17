"""Tests for the z-score baseline.

The detector itself is one comparison, so the interesting assertions are not about the
arithmetic but about the operating point: it has to catch most amount spikes without
burying an analyst in false positives, and it has to fail loudly when handed a frame
that never went through the feature step. The recall and precision floors below sit
comfortably under the measured values so ordinary noise does not fail CI, while a real
regression in either the features or the threshold still does.
"""

import pandas as pd
import pytest

from sentinel.detectors.zscore import DEFAULT_THRESHOLD, detect
from sentinel.features import add_features
from sentinel.generate import generate_transactions

SMALL = 6_000


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    return add_features(generate_transactions(SMALL, seed=7, n_users=50))


def test_returns_an_aligned_boolean_series(featured):
    flags = detect(featured)
    assert flags.dtype == bool
    assert flags.name == "zscore_flag"
    assert flags.index.equals(featured.index)


def test_flags_exactly_the_rows_above_the_threshold(featured):
    flags = detect(featured, threshold=3.0)
    assert (featured.loc[flags, "amount_zscore"] >= 3.0).all()
    assert (featured.loc[~flags, "amount_zscore"] < 3.0).all()


def test_catches_most_amount_spikes(featured):
    spikes = featured["anomaly_type"] == "amount_spike"
    recall = (detect(featured) & spikes).sum() / spikes.sum()
    assert recall > 0.5


def test_most_flags_are_real_spikes(featured):
    """Precision matters more than recall here: alerts cost an analyst's attention."""
    flags = detect(featured)
    assert flags.sum() > 0
    assert (flags & (featured["anomaly_type"] == "amount_spike")).sum() / flags.sum() > 0.5


def test_clean_data_produces_almost_no_alerts():
    clean = add_features(generate_transactions(SMALL, seed=21, n_users=50, anomaly_rate=0.0))
    assert detect(clean).mean() < 0.005


def test_raising_the_threshold_only_removes_flags(featured):
    loose, strict = detect(featured, threshold=3.0), detect(featured, threshold=6.0)
    assert strict.sum() < loose.sum()
    assert not (strict & ~loose).any()


def test_low_amounts_are_not_flagged(featured):
    """The detector is one-sided; an unusually small charge is not fraud."""
    assert not detect(featured)[featured["amount_zscore"] < 0].any()


def test_default_threshold_is_the_calibrated_one():
    # Guards against a stray edit to the constant; the value is justified in its comment.
    assert DEFAULT_THRESHOLD == 4.0


def test_unfeaturised_frame_is_rejected():
    raw = generate_transactions(200, seed=3, n_users=10)
    with pytest.raises(KeyError, match="add_features"):
        detect(raw)


@pytest.mark.parametrize("threshold", [0.0, -1.0])
def test_invalid_threshold_is_rejected(featured, threshold):
    with pytest.raises(ValueError):
        detect(featured, threshold=threshold)
