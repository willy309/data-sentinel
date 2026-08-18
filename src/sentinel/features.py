"""Per-user behavioural features.

Every feature here is computed *per cardholder* and from that cardholder's own past.
Both properties matter:

* per user, because the generator gives each user their own spending scale — a global
  z-score would just rediscover which users are rich;
* from the past only, because a fraudulent charge that is allowed into its own baseline
  drags the mean toward itself and hides. The rolling statistics are therefore shifted
  by one row, which also makes the features honest about what a live scorer would know
  at the moment the transaction arrives.

Between them the columns cover the three anomaly types the generator injects, so a
detector can be scored against a label it actually had a chance to see.
"""

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "amount_zscore",
    "txn_count_1h",
    "merchant_freq",
    "hour",
    "hour_sin",
    "hour_cos",
)

# Twenty prior charges is roughly two-thirds of a user's month in the default 50k frame:
# long enough for a stable mean, short enough to follow a genuine change in habits.
DEFAULT_WINDOW = 20
# Below this many prior charges the deviation is noise, so the feature says nothing (0.0)
# rather than guessing. std() also needs at least two points to exist at all.
DEFAULT_MIN_HISTORY = 5
DEFAULT_VELOCITY_WINDOW = "1h"
# Pseudo-count pulling a thin merchant history toward the population base rate. At 5, a
# cardholder needs roughly five charges before their own habits outweigh the prior,
# which stops a brand new card from looking suspicious on its very first purchase.
DEFAULT_CATEGORY_PRIOR = 5.0


def add_features(
    frame: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    min_history: int = DEFAULT_MIN_HISTORY,
    velocity_window: str = DEFAULT_VELOCITY_WINDOW,
    category_prior: float = DEFAULT_CATEGORY_PRIOR,
) -> pd.DataFrame:
    """Return ``frame`` with :data:`FEATURE_COLUMNS` appended.

    Args:
        frame: Rows as produced by :func:`sentinel.generate.generate_transactions`,
            sorted by ``timestamp``. Sorting is required, not merely assumed: the
            velocity window rolls over time and pandas rejects an unordered index.
        window: Number of prior charges the amount baseline is drawn from.
        min_history: Prior charges required before the baseline is trusted.
        velocity_window: Pandas offset string for the transaction-count window.
        category_prior: Pseudo-count weighting the population base rate in
            ``merchant_freq``. Larger values distrust short histories for longer.

    Returns:
        A copy of ``frame`` with six added columns:

        ``amount_zscore``
            Deviation of the charge from the user's recent norm, in standard deviations.
            Computed on ``log1p(amount)``: spending is lognormal, so in raw dollars an
            ordinary large purchase already scores several sigma and the threshold has
            no calibrated meaning. ``0.0`` where history is too short.
        ``txn_count_1h``
            Charges by this user in the trailing ``velocity_window``, inclusive of the
            current row, so a lone transaction scores 1.
        ``merchant_freq``
            Share of this user's prior charges that fell in this charge's category,
            smoothed toward the population base rate. Near zero means the cardholder
            has never shopped this category before.
        ``hour``
            Hour of day, 0-23. Kept raw for reports and for any rule a human wants to
            read, but note it is a poor model input on its own: 23 and 0 are one minute
            apart and numerically as far apart as the column allows.
        ``hour_sin``, ``hour_cos``
            The same clock placed on a circle, so midnight is adjacent to 23:00. This is
            the pair to feed a model. Neither encodes *which* hours are suspicious —
            hard-coding the generator's injection window would leak the answer into the
            input, so that judgement stays with the detector.

    Raises:
        ValueError: If ``frame`` is not sorted by timestamp, or an argument is invalid.
    """
    if window < 2:
        raise ValueError(f"window must be at least 2, got {window}")
    if not 2 <= min_history <= window:
        raise ValueError(f"min_history must be in [2, {window}], got {min_history}")
    if category_prior <= 0:
        raise ValueError(f"category_prior must be positive, got {category_prior}")
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("frame must be sorted by timestamp")

    out = frame.copy()
    # log1p rather than log: harmless at real transaction sizes and defined at zero, so
    # a degenerate amount cannot turn one row into -inf and poison the whole user.
    log_amount = np.log1p(frame["amount"])
    grouped = log_amount.groupby(frame["user_id"])
    prior_mean = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_history).mean()
    )
    prior_std = grouped.transform(
        lambda s: s.shift(1).rolling(window, min_periods=min_history).std()
    )

    # A user who has spent exactly the same amount every time has no scale to measure
    # against; NaN here becomes 0.0 below, i.e. "no opinion", not "infinitely unusual".
    out["amount_zscore"] = ((log_amount - prior_mean) / prior_std.where(prior_std > 0)).fillna(0.0)
    out["txn_count_1h"] = frame.groupby("user_id")["timestamp"].transform(
        lambda times: _rolling_count(times, velocity_window)
    )
    out["merchant_freq"] = _merchant_frequency(frame, category_prior)

    hour = frame["timestamp"].dt.hour
    radians = 2 * np.pi * hour / 24
    out["hour"] = hour
    out["hour_sin"] = np.sin(radians)
    out["hour_cos"] = np.cos(radians)
    return out


def _rolling_count(times: pd.Series, window: str) -> pd.Series:
    """Count timestamps in the trailing ``window`` ending at each of ``times``.

    A time-based ``rolling`` needs a DatetimeIndex, but the caller needs the result back
    on the original row index, so the index is swapped out and restored positionally.
    """
    counts = pd.Series(1.0, index=pd.DatetimeIndex(times)).rolling(window).sum()
    return pd.Series(counts.to_numpy(), index=times.index)


def _merchant_frequency(frame: pd.DataFrame, prior_weight: float) -> pd.Series:
    """Smoothed share of a user's prior charges falling in each charge's own category.

    Additive smoothing toward the population base rate, which is what makes the cold
    start behave: with no history the feature returns the base rate — what you would
    expect of a stranger — and slides toward the cardholder's own habits as they shop.
    A hard count would instead read 0/0 on every card's first purchase.
    """
    categories = frame["merchant_category"]
    seen = pd.get_dummies(categories, dtype=float)
    # cumsum() includes the current row, so subtract it straight back off: a charge must
    # never be admitted as evidence for its own ordinariness.
    prior_counts = seen.groupby(frame["user_id"]).cumsum() - seen
    prior_in_category = prior_counts.to_numpy()[
        np.arange(len(frame)), seen.columns.get_indexer(categories)
    ]
    prior_total = frame.groupby("user_id").cumcount().to_numpy()

    # The base rate is a whole-frame aggregate, so strictly it peeks past the current
    # row. That is a deliberate and narrow exception: it is one number per category over
    # tens of thousands of rows, carries no per-row information, and in a live system
    # would simply be fitted once on historical data and shipped as a constant.
    base_rate = categories.map(categories.value_counts(normalize=True)).to_numpy()
    return pd.Series(
        (prior_in_category + prior_weight * base_rate) / (prior_total + prior_weight),
        index=frame.index,
    )
