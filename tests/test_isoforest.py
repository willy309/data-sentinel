"""Tests for the IsolationForest detector.

A learned detector is easy to test badly: assert an exact flag count and the suite
breaks on the next scikit-learn release, assert nothing and a model that flags at random
still passes. So these check the contract (shape, determinism, validation) exactly, and
the behaviour (does it beat chance, on which anomaly types) with floors set well under
the measured values.
"""

import pandas as pd
import pytest

from sentinel.detectors.isoforest import (
    DEFAULT_CONTAMINATION,
    MODEL_FEATURES,
    detect,
)
from sentinel.features import FEATURE_COLUMNS, add_features
from sentinel.generate import generate_transactions

SMALL = 8_000


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    return add_features(generate_transactions(SMALL, seed=7, n_users=50))


def test_returns_an_aligned_boolean_series(featured):
    flags = detect(featured)
    assert flags.dtype == bool
    assert flags.name == "isoforest_flag"
    assert flags.index.equals(featured.index)


def test_flags_roughly_the_contamination_share(featured):
    assert detect(featured, contamination=0.02).mean() == pytest.approx(0.02, abs=0.005)


def test_more_contamination_flags_more_rows(featured):
    assert detect(featured, contamination=0.02).sum() > detect(featured, contamination=0.005).sum()


def test_same_seed_reproduces_flags(featured):
    pd.testing.assert_series_equal(detect(featured, seed=3), detect(featured, seed=3))


def test_model_features_exclude_the_ones_measured_to_hurt():
    """Guards a decision that cost real measurement to reach.

    Adding the hour columns halved precision and recall on the default frame and caught
    no odd-hours anomalies, because the generator's dead zone holds more legitimate
    charges than injected ones. merchant_freq moved nothing. Anyone widening this list
    should have to change this test on purpose, with fresh numbers in hand.
    """
    assert set(MODEL_FEATURES) == {"amount_zscore", "txn_count_1h"}
    assert set(MODEL_FEATURES) < set(FEATURE_COLUMNS)


def test_beats_chance_on_the_injected_labels(featured):
    flags = detect(featured)
    lift = featured.loc[flags, "is_anomaly"].mean() / featured["is_anomaly"].mean()
    # Anything near 1.0 would mean the forest is flagging at random.
    assert lift > 10


def test_catches_both_amount_spikes_and_velocity_bursts(featured):
    """The reason for having it: one detector covering two unrelated failure modes."""
    flags = detect(featured)
    recall = (
        featured.loc[featured["is_anomaly"]].assign(f=flags).groupby("anomaly_type")["f"].mean()
    )
    assert recall["amount_spike"] > 0.4
    assert recall["velocity_burst"] > 0.4


def test_unfeaturised_frame_is_rejected():
    raw = generate_transactions(200, seed=3, n_users=10)
    with pytest.raises(KeyError, match="add_features"):
        detect(raw)


def test_default_contamination_is_the_calibrated_one():
    # Justified in the constant's comment; guards against a stray edit.
    assert DEFAULT_CONTAMINATION == 0.0075


@pytest.mark.parametrize("contamination", [0.0, -0.1, 0.6, 1.0])
def test_invalid_contamination_is_rejected(featured, contamination):
    with pytest.raises(ValueError):
        detect(featured, contamination=contamination)


def test_empty_feature_list_is_rejected(featured):
    with pytest.raises(ValueError, match="at least one column"):
        detect(featured, features=())
