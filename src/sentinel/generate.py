"""Seeded synthetic transaction generator.

Real card data cannot be published, so every number in this project is manufactured
here. Beyond avoiding licensing and privacy problems, generating the data means we
know which rows are bad: the generator labels exactly the rows it corrupts, and those
labels are the only reason the detectors can be scored with precision and recall
instead of eyeballed.

Two deliberate properties of the "normal" population exist to make the later feature
work meaningful:

* each user gets their own spending scale, so a $400 charge is routine for one user
  and alarming for another (this is why features are computed per user, not globally);
* activity is concentrated in waking hours, so a 3am cluster is genuinely unusual
  rather than an artefact of uniform sampling.
"""

import numpy as np
import pandas as pd

MERCHANT_CATEGORIES: tuple[str, ...] = (
    "groceries",
    "restaurants",
    "fuel",
    "retail",
    "online",
    "travel",
    "entertainment",
    "electronics",
)
LOCATIONS: tuple[str, ...] = (
    "Cupertino, CA",
    "Seattle, WA",
    "Austin, TX",
    "Denver, CO",
    "Chicago, IL",
    "Atlanta, GA",
    "Boston, MA",
    "Portland, OR",
)
ANOMALY_TYPES: tuple[str, ...] = ("amount_spike", "velocity_burst", "odd_hours")
COLUMNS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "user_id",
    "amount",
    "merchant_category",
    "location",
    "is_anomaly",
    "anomaly_type",
)

# Relative volume by hour of day (index 0 == midnight). Hand-tuned rather than fitted:
# the shape only has to be plausibly diurnal for odd-hours activity to stand out.
# fmt: off
_HOUR_WEIGHTS = np.array([
    0.4, 0.25, 0.2, 0.2, 0.3, 0.8,  # 00:00-05:59
    2.0, 4.0, 6.0, 7.0, 7.0, 8.0,   # 06:00-11:59
    8.0, 7.0, 7.0, 7.0, 7.0, 7.0,   # 12:00-17:59
    7.0, 6.0, 5.0, 4.0, 2.0, 1.0,   # 18:00-23:59
])
# fmt: on
_CATEGORY_WEIGHTS = np.array([0.24, 0.20, 0.13, 0.14, 0.15, 0.04, 0.07, 0.03])
_SECONDS_PER_DAY = 86_400
# A burst is a run of charges on one card inside a few minutes. Eight is enough for a
# rolling velocity feature to notice without making the label set trivially separable.
_BURST_SIZE = 8
_HOME_LOCATION_SHARE = 0.9


def generate_transactions(
    n: int = 50_000,
    *,
    seed: int = 42,
    n_users: int = 500,
    anomaly_rate: float = 0.01,
    days: int = 30,
    start: str = "2025-01-01",
) -> pd.DataFrame:
    """Generate ``n`` synthetic transactions with a labelled anomaly subset.

    The same ``seed`` always produces an identical frame; tests and the README's sample
    output both depend on that, so treat the seeded output as an interface.

    Args:
        n: Total rows returned. Anomalies are corrupted rows inside this total, not
            extra rows, which keeps the label rate at ``anomaly_rate``.
        seed: Seed for the underlying ``numpy`` generator.
        n_users: Distinct cardholders to spread the rows across.
        anomaly_rate: Fraction of rows to corrupt, split across ``ANOMALY_TYPES``.
        days: Length of the observation window.
        start: Window start, parsed by ``pandas`` and treated as UTC.

    Returns:
        A frame with the columns in ``COLUMNS``, sorted by ``timestamp``. ``is_anomaly``
        marks injected rows and ``anomaly_type`` names the injection (empty string for
        normal rows). Note that a normal row can still look odd by chance; the labels
        record what was injected, not every row a human would question.

    Raises:
        ValueError: If any argument is outside its supported range.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if n_users <= 0:
        raise ValueError(f"n_users must be positive, got {n_users}")
    if not 0.0 <= anomaly_rate < 1.0:
        raise ValueError(f"anomaly_rate must be in [0, 1), got {anomaly_rate}")
    if days <= 0:
        raise ValueError(f"days must be positive, got {days}")

    rng = np.random.default_rng(seed)
    user_of_row = rng.integers(0, n_users, size=n)

    user_log_mean = rng.uniform(2.6, 4.4, size=n_users)
    amount = np.round(rng.lognormal(user_log_mean[user_of_row], 0.55), 2)

    # Timestamps are carried as integer epoch seconds until the very end: injection
    # rewrites them, and integer arithmetic is easier to reason about than datetime64
    # offsets. Second-of-day is therefore always ``timestamp % 86_400``.
    start_epoch = int(pd.Timestamp(start).value // 10**9)
    hour = rng.choice(24, size=n, p=_HOUR_WEIGHTS / _HOUR_WEIGHTS.sum())
    timestamp = (
        start_epoch
        + rng.integers(0, days, size=n) * _SECONDS_PER_DAY
        + hour * 3600
        + rng.integers(0, 3600, size=n)
    )

    category = rng.choice(np.array(MERCHANT_CATEGORIES), size=n, p=_CATEGORY_WEIGHTS)
    home = rng.integers(0, len(LOCATIONS), size=n_users)
    away = rng.integers(0, len(LOCATIONS), size=n)
    location = np.array(LOCATIONS)[
        np.where(rng.random(n) < _HOME_LOCATION_SHARE, home[user_of_row], away)
    ]

    anomaly_type = np.full(n, "", dtype=object)
    _inject_anomalies(
        rng,
        amount=amount,
        timestamp=timestamp,
        user_of_row=user_of_row,
        anomaly_type=anomaly_type,
        n_users=n_users,
        n_anomalies=int(round(n * anomaly_rate)),
    )

    user_ids = np.array([f"u{i:04d}" for i in range(n_users)])
    frame = pd.DataFrame(
        {
            # pandas 2 and 3 disagree on the unit inferred from epoch seconds, so pin
            # it: downstream rolling windows and the tests both want one stable dtype.
            "timestamp": pd.to_datetime(timestamp, unit="s").astype("datetime64[ns]"),
            "user_id": user_ids[user_of_row],
            "amount": amount,
            "merchant_category": category,
            "location": location,
            "is_anomaly": anomaly_type != "",
            "anomaly_type": anomaly_type.astype(str),
        }
    )
    frame = frame.sort_values("timestamp", kind="stable", ignore_index=True)
    # IDs are assigned after sorting so they read in chronological order, the way a
    # sequential transaction ID would.
    frame.insert(0, "transaction_id", [f"t{i:06d}" for i in range(n)])
    return frame[list(COLUMNS)]


def _inject_anomalies(
    rng: np.random.Generator,
    *,
    amount: np.ndarray,
    timestamp: np.ndarray,
    user_of_row: np.ndarray,
    anomaly_type: np.ndarray,
    n_users: int,
    n_anomalies: int,
) -> None:
    """Corrupt ``n_anomalies`` rows in place, split across the three anomaly types.

    Velocity bursts go first because they are the constrained case: they need several
    rows belonging to the *same* user, whereas spikes and odd-hours rows can use any
    row that is still untouched.
    """
    if n_anomalies == 0:
        return

    per_type = n_anomalies // 3
    n_velocity = _inject_velocity_bursts(
        rng,
        timestamp=timestamp,
        user_of_row=user_of_row,
        anomaly_type=anomaly_type,
        n_users=n_users,
        # Bursts are injected whole, so only ask for a multiple of the burst size.
        n_target=(per_type // _BURST_SIZE) * _BURST_SIZE,
    )

    remaining = n_anomalies - n_velocity
    free = np.flatnonzero(anomaly_type == "")
    chosen = rng.choice(free, size=min(remaining, free.size), replace=False)
    spikes, odd_hours = chosen[: remaining // 2], chosen[remaining // 2 :]

    # A spike is a single charge far above the cardholder's own normal, which is the
    # textbook case a rolling z-score should catch.
    amount[spikes] = np.round(amount[spikes] * rng.uniform(8.0, 30.0, size=spikes.size), 2)
    anomaly_type[spikes] = "amount_spike"

    # Move the row into the 01:00-04:59 dead zone, keeping its date.
    current_hour = timestamp[odd_hours] % _SECONDS_PER_DAY // 3600
    target_hour = rng.integers(1, 5, size=odd_hours.size)
    timestamp[odd_hours] += (target_hour - current_hour) * 3600
    anomaly_type[odd_hours] = "odd_hours"


def _inject_velocity_bursts(
    rng: np.random.Generator,
    *,
    timestamp: np.ndarray,
    user_of_row: np.ndarray,
    anomaly_type: np.ndarray,
    n_users: int,
    n_target: int,
) -> int:
    """Compress groups of ``_BURST_SIZE`` same-user rows into a few minutes.

    Returns the number of rows actually marked, which can fall short of ``n_target``
    when too few users own enough rows to form a burst.
    """
    if n_target < _BURST_SIZE:
        return 0

    # One stable sort gives contiguous per-user slices, which is cheaper and clearer
    # than grouping a frame that does not exist yet.
    order = np.argsort(user_of_row, kind="stable")
    bounds = np.searchsorted(user_of_row[order], np.arange(n_users + 1))
    eligible = rng.permutation(np.flatnonzero(np.diff(bounds) >= _BURST_SIZE))

    marked = 0
    for user in eligible:
        if marked >= n_target:
            break
        rows = rng.choice(order[bounds[user] : bounds[user + 1]], size=_BURST_SIZE, replace=False)
        # Anchor on one of the user's own timestamps so the burst lands inside their
        # real activity window rather than at an arbitrary hour.
        gaps = np.cumsum(rng.integers(15, 120, size=_BURST_SIZE - 1))
        timestamp[rows[1:]] = timestamp[rows[0]] + gaps
        anomaly_type[rows] = "velocity_burst"
        marked += _BURST_SIZE
    return marked
