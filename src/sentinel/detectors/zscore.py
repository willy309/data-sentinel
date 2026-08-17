"""Rolling z-score detector: the baseline every later method has to beat.

Deliberately the dumbest thing that works. It reads one feature and compares it to one
number, which makes it fast, explainable to a fraud analyst ("this charge sat six sigma
above your own norm"), and a fair yardstick — a model that cannot beat a threshold on
one column is not earning its complexity.

It only claims amount spikes. Velocity bursts and odd-hours activity are visible in
``txn_count_1h`` and ``hour``, and belong to detectors that read those columns.
"""

import pandas as pd

from sentinel.features import FEATURE_COLUMNS

FEATURE = "amount_zscore"
# Chosen by sweeping the default 50k frame, not by intuition: 3.0 gives 0.87 recall at
# 0.36 precision, 5.0 gives 0.55 at 0.87, and 4.0 sits at 0.76/0.74 — the best F1 of the
# three. Because the feature is z-scored in log space it is nearly standard normal, so
# the number also has its textbook meaning. Re-sweep whenever the generator changes.
DEFAULT_THRESHOLD = 4.0


def detect(frame: pd.DataFrame, *, threshold: float = DEFAULT_THRESHOLD) -> pd.Series:
    """Flag transactions whose amount is far above the cardholder's recent norm.

    Args:
        frame: A frame carrying :data:`FEATURE`, i.e. the output of
            :func:`sentinel.features.add_features`.
        threshold: Standard deviations above the user's rolling mean at which a charge
            is flagged.

    Returns:
        A boolean Series named ``zscore_flag``, aligned to ``frame``'s index.

    Raises:
        KeyError: If ``frame`` has not been through the feature step.
        ValueError: If ``threshold`` is not positive.
    """
    if FEATURE not in frame.columns:
        raise KeyError(
            f"{FEATURE!r} missing; pass the frame through sentinel.features.add_features "
            f"first (it adds {', '.join(FEATURE_COLUMNS)})"
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")

    # One-sided on purpose. A charge far *below* the norm is a coffee, not a theft, and
    # flagging it would cost precision to catch nothing this dataset calls fraud.
    return (frame[FEATURE] >= threshold).rename("zscore_flag")
