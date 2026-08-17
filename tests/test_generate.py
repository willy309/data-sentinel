"""Tests for the synthetic generator.

The generator is the ground truth for every later evaluation number, so these tests
guard the properties the rest of the project assumes: the seed is honoured, the label
column matches what was actually injected, and each anomaly type is genuinely visible
in the data rather than only in the label.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.generate import ANOMALY_TYPES, COLUMNS, LOCATIONS, generate_transactions

SMALL = 4_000
# Velocity bursts are injected in whole groups of eight, so the realised anomaly count
# can land a couple of groups under the request. Anything larger is a real bug.
BURST_TOLERANCE = 16


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """A single small frame shared by the read-only assertions below."""
    return generate_transactions(SMALL, seed=7, n_users=50)


def test_schema_and_dtypes(frame):
    assert list(frame.columns) == list(COLUMNS)
    assert len(frame) == SMALL
    assert frame["timestamp"].dtype == np.dtype("datetime64[ns]")
    assert frame["amount"].dtype == np.float64
    assert frame["is_anomaly"].dtype == np.bool_
    assert not frame.isna().any().any()


def test_same_seed_reproduces_frame():
    a = generate_transactions(SMALL, seed=42, n_users=50)
    b = generate_transactions(SMALL, seed=42, n_users=50)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_changes_frame():
    a = generate_transactions(SMALL, seed=42, n_users=50)
    b = generate_transactions(SMALL, seed=43, n_users=50)
    assert not a["amount"].equals(b["amount"])


def test_anomaly_rate_is_close_to_requested(frame):
    assert frame["is_anomaly"].sum() == pytest.approx(SMALL * 0.01, abs=BURST_TOLERANCE)


def test_zero_anomaly_rate_produces_clean_data():
    clean = generate_transactions(1_000, seed=1, n_users=20, anomaly_rate=0.0)
    assert not clean["is_anomaly"].any()
    assert (clean["anomaly_type"] == "").all()


def test_label_and_type_columns_agree(frame):
    assert (frame.loc[frame["is_anomaly"], "anomaly_type"] != "").all()
    assert (frame.loc[~frame["is_anomaly"], "anomaly_type"] == "").all()


def test_all_three_anomaly_types_are_injected(frame):
    assert set(frame.loc[frame["is_anomaly"], "anomaly_type"]) == set(ANOMALY_TYPES)


def test_rows_are_sorted_by_timestamp(frame):
    assert frame["timestamp"].is_monotonic_increasing
    assert frame["transaction_id"].is_monotonic_increasing


def test_window_is_respected():
    days = 5
    windowed = generate_transactions(2_000, seed=3, n_users=20, days=days, start="2025-06-01")
    # Odd-hours injection only shifts the hour within a row's own day, so the whole
    # frame must still sit inside the requested window.
    assert windowed["timestamp"].min() >= pd.Timestamp("2025-06-01")
    assert windowed["timestamp"].max() < pd.Timestamp("2025-06-01") + pd.Timedelta(days=days)


def test_categorical_values_come_from_the_declared_vocabularies(frame):
    assert set(frame["location"]) <= set(LOCATIONS)
    assert frame["user_id"].nunique() <= 50
    assert frame["user_id"].str.match(r"^u\d{4}$").all()


def test_amount_spikes_are_large_relative_to_the_user(frame):
    spikes = frame["anomaly_type"] == "amount_spike"
    per_user_median = frame.groupby("user_id")["amount"].transform("median")
    # Spikes multiply the drawn amount by at least 8x, so even against a median that
    # the spike itself inflates, every one should clear a factor of three.
    assert (frame.loc[spikes, "amount"] > 3 * per_user_median[spikes]).all()


def test_velocity_bursts_are_tightly_clustered_in_time(frame):
    bursts = frame[frame["anomaly_type"] == "velocity_burst"]
    spans = bursts.groupby("user_id")["timestamp"].agg(lambda s: s.max() - s.min())
    # Eight charges with gaps of at most two minutes cannot span more than 15 minutes.
    assert (spans <= pd.Timedelta(minutes=15)).all()


def test_odd_hours_rows_land_in_the_dead_zone(frame):
    odd = frame.loc[frame["anomaly_type"] == "odd_hours", "timestamp"].dt.hour
    assert odd.between(1, 4).all()


def test_normal_activity_mostly_avoids_the_dead_zone(frame):
    normal_hours = frame.loc[~frame["is_anomaly"], "timestamp"].dt.hour
    # If the diurnal shape were lost, odd-hours anomalies would stop being anomalous.
    assert normal_hours.between(1, 4).mean() < 0.05


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n": 0},
        {"n": -1},
        {"n_users": 0},
        {"days": 0},
        {"anomaly_rate": -0.1},
        {"anomaly_rate": 1.0},
    ],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**{"n": 100, "n_users": 5, **kwargs})
