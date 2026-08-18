"""Tests for the feature layer.

The features decide what any detector can possibly see, so these tests check two
different kinds of property. First, that each feature actually separates the anomaly
type it is meant to expose — a feature that does not is dead weight. Second, and more
importantly, that no feature reads the present or the future: leakage would make every
later precision/recall number a lie, and it is invisible in the metrics themselves.
"""

import numpy as np
import pandas as pd
import pytest

from sentinel.features import FEATURE_COLUMNS, add_features
from sentinel.generate import COLUMNS, generate_transactions

SMALL = 6_000


@pytest.fixture(scope="module")
def featured() -> pd.DataFrame:
    """One featured frame shared by the read-only assertions below."""
    return add_features(generate_transactions(SMALL, seed=7, n_users=50))


def hand_built(users, amounts, categories, timestamps=None) -> pd.DataFrame:
    """A minimal frame with known values, for properties the generator cannot pin down."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps)
            if timestamps is not None
            else pd.date_range("2025-01-01", periods=len(users), freq="h"),
            "user_id": users,
            "amount": amounts,
            "merchant_category": categories,
        }
    )


def test_original_columns_survive_and_features_are_added(featured):
    assert list(featured.columns) == list(COLUMNS) + list(FEATURE_COLUMNS)
    assert len(featured) == SMALL
    assert not featured[list(FEATURE_COLUMNS)].isna().any().any()
    assert np.isfinite(featured[list(FEATURE_COLUMNS)]).all().all()


def test_input_frame_is_not_mutated():
    raw = generate_transactions(1_000, seed=5, n_users=20)
    before = raw.copy()
    add_features(raw)
    pd.testing.assert_frame_equal(raw, before)


def test_zscore_ignores_the_current_row():
    """The row's own amount must not enter its baseline, or a spike hides itself.

    Rewriting one charge to an absurd value may only change that row's z-score and the
    scores of rows after it. If the preceding rows move at all, the window is reading
    forwards.
    """
    raw = generate_transactions(2_000, seed=11, n_users=10, anomaly_rate=0.0)
    target = raw.index[raw["user_id"] == raw.loc[1_500, "user_id"]][-1]

    baseline = add_features(raw)["amount_zscore"]
    tampered = raw.copy()
    tampered.loc[target, "amount"] = 999_999.0
    after = add_features(tampered)["amount_zscore"]

    pd.testing.assert_series_equal(baseline[:target], after[:target])
    assert after[target] > baseline[target]


def test_zscore_is_measured_per_user_not_globally():
    """A big spender's ordinary charge must not look anomalous next to a small one."""
    rows = 60
    # Alternating users two orders of magnitude apart. Every charge is routine for
    # whoever made it, so no row should score highly.
    frame = hand_built(
        users=["rich", "thrifty"] * (rows // 2),
        amounts=[500.0, 5.0] * (rows // 2),
        categories=["groceries"] * rows,
    )
    assert add_features(frame)["amount_zscore"].abs().max() < 1.0


def test_zscore_says_nothing_until_history_exists(featured):
    """Early rows score exactly 0.0 rather than an invented deviation."""
    first_five = featured.groupby("user_id").head(5)
    assert (first_five["amount_zscore"] == 0.0).all()


def test_zscore_separates_amount_spikes(featured):
    spikes = featured["anomaly_type"] == "amount_spike"
    normal = ~featured["is_anomaly"]
    # Scored rows only: a spike among a user's first few charges has no baseline yet and
    # legitimately scores 0. Of the rest, nearly all should clear the normal population.
    scored = featured.loc[spikes & (featured["amount_zscore"] != 0.0), "amount_zscore"]
    assert scored.median() > 4.0
    assert featured.loc[normal, "amount_zscore"].quantile(0.99) < scored.median()


def test_velocity_count_separates_bursts(featured):
    bursts = featured["anomaly_type"] == "velocity_burst"
    assert featured.loc[bursts, "txn_count_1h"].mean() > 3.0
    assert featured.loc[~featured["is_anomaly"], "txn_count_1h"].mean() < 2.0


def test_velocity_count_is_per_user_and_includes_the_current_row():
    """Two users transacting in lockstep must not inflate each other's counts."""
    frame = hand_built(
        users=["a", "a", "b", "b"],
        amounts=[10.0, 20.0, 30.0, 40.0],
        categories=["groceries"] * 4,
        timestamps=["2025-01-01 09:00", "2025-01-01 09:01"] * 2,
    ).sort_values("timestamp", kind="stable", ignore_index=True)
    assert add_features(frame)["txn_count_1h"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_velocity_count_forgets_transactions_outside_the_window():
    frame = hand_built(
        users=["a", "a"],
        amounts=[10.0, 20.0],
        categories=["groceries"] * 2,
        timestamps=["2025-01-01 09:00", "2025-01-01 11:00"],
    )
    assert add_features(frame)["txn_count_1h"].tolist() == [1.0, 1.0]


def test_merchant_freq_starts_at_the_population_base_rate():
    """A cardholder's first ever charge is judged against strangers, not against 0/0.

    Three quarters of these rows are groceries, and every user appears exactly once, so
    every row is a cold start and must read back the base rate of its own category.
    """
    frame = hand_built(
        users=["a", "b", "c", "d"],
        amounts=[10.0] * 4,
        categories=["groceries", "groceries", "groceries", "travel"],
    )
    freq = add_features(frame)["merchant_freq"]
    assert freq.tolist() == pytest.approx([0.75, 0.75, 0.75, 0.25])


def test_merchant_freq_collapses_for_a_category_the_user_has_never_used():
    """The point of the feature: first electronics purchase on a groceries-only card."""
    frame = hand_built(
        users=["a"] * 21,
        amounts=[10.0] * 21,
        categories=["groceries"] * 20 + ["electronics"],
    )
    freq = add_features(frame)["merchant_freq"]
    assert freq.iloc[19] > 0.8
    assert freq.iloc[20] < 0.05


def test_merchant_freq_excludes_the_current_charge():
    """Counting the charge itself would make every first-of-a-kind look established.

    Note there is deliberately no whole-frame leakage test here: the base rate is a
    population aggregate over every row, a documented exception, so editing one row's
    category does shift every row's smoothing slightly. The per-row history is what must
    stay strictly backward-looking, and that is what this checks.
    """
    frame = hand_built(users=["a", "a"], amounts=[10.0, 20.0], categories=["travel"] * 2)
    freq = add_features(frame)["merchant_freq"]
    # Both rows are travel and travel is 100% of the frame, so the prior is 1.0 for both.
    # The first row has no history (0 + 5*1.0)/(0 + 5) == 1.0; the second has one prior
    # travel charge, (1 + 5*1.0)/(1 + 5) == 1.0. Equal only because the prior agrees with
    # the history — what matters is that row one counts zero priors, not one.
    assert freq.tolist() == pytest.approx([1.0, 1.0])
    mixed = hand_built(users=["a", "a"], amounts=[10.0, 20.0], categories=["travel", "fuel"])
    # travel and fuel are each half the frame; row two has one prior travel charge and
    # zero prior fuel charges, so (0 + 5*0.5)/(1 + 5) == 0.4167, not 0.5.
    assert add_features(mixed)["merchant_freq"].iloc[1] == pytest.approx(2.5 / 6)


def test_merchant_freq_is_a_proportion(featured):
    assert featured["merchant_freq"].between(0.0, 1.0, inclusive="right").all()


def test_hour_matches_the_timestamp(featured):
    pd.testing.assert_series_equal(
        featured["hour"], featured["timestamp"].dt.hour, check_names=False
    )
    odd = featured.loc[featured["anomaly_type"] == "odd_hours", "hour"]
    assert odd.between(1, 4).all()


def test_cyclical_hour_lies_on_the_unit_circle(featured):
    radius = featured["hour_sin"] ** 2 + featured["hour_cos"] ** 2
    assert radius.round(9).eq(1.0).all()


def test_cyclical_hour_puts_midnight_next_to_late_evening():
    """The whole reason the encoding exists: 23:00 and 00:00 are an hour apart."""
    frame = hand_built(
        users=["a"] * 3,
        amounts=[10.0] * 3,
        categories=["groceries"] * 3,
        timestamps=["2025-01-01 22:00", "2025-01-01 23:00", "2025-01-02 00:00"],
    )
    out = add_features(frame)
    point = out[["hour_sin", "hour_cos"]].to_numpy()
    wraparound = np.linalg.norm(point[2] - point[1])
    ordinary = np.linalg.norm(point[1] - point[0])
    assert wraparound == pytest.approx(ordinary)
    # Raw hour would have called these 23 apart, which is the trap being avoided.
    assert abs(out["hour"].iloc[2] - out["hour"].iloc[1]) == 23


def test_features_are_reproducible():
    raw = generate_transactions(1_000, seed=13, n_users=20)
    pd.testing.assert_frame_equal(add_features(raw), add_features(raw))


def test_unsorted_frame_is_rejected():
    raw = generate_transactions(500, seed=2, n_users=10)
    with pytest.raises(ValueError, match="sorted by timestamp"):
        add_features(raw.iloc[::-1])


@pytest.mark.parametrize(
    "kwargs",
    [{"window": 1}, {"min_history": 1}, {"window": 5, "min_history": 6}, {"category_prior": 0}],
)
def test_invalid_arguments_are_rejected(kwargs):
    raw = generate_transactions(200, seed=4, n_users=5)
    with pytest.raises(ValueError):
        add_features(raw, **kwargs)
