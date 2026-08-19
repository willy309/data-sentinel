"""IsolationForest detector: the learned counterpart to the z-score baseline.

Where the z-score detector is told what "unusual" means, this one is shown the feature
space and left to find sparse regions itself. That buys two things the baseline cannot
have: it combines features, so a moderately large charge during a moderately fast burst
is caught by the pair even though neither alone crosses a threshold; and it needs no
per-feature threshold, only a rough guess at how much fraud exists.

It is unsupervised on purpose. The labels exist to score it afterwards, never to train
it — a supervised model on 1% positives would be the more accurate and far less
interesting answer, and it would not survive contact with a fraud pattern nobody has
labelled yet.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

# Deliberately two columns, not the six that features.py produces. Measured on the
# default 50k frame, adding hour_sin/hour_cos drops precision and recall from 0.46 to
# 0.26 while catching no odd-hours anomalies at all, and merchant_freq moves nothing
# either way. The generator puts 474 legitimate charges in the 01:00-04:59 window
# against 170 injected ones, so there is genuinely no separation to find there — the
# extra dimensions only give the trees somewhere useless to split. Revisit this list if
# the generator ever grows per-user category habits or a quieter dead zone.
MODEL_FEATURES: tuple[str, ...] = ("amount_zscore", "txn_count_1h")

# Below the true 1% anomaly rate on purpose. Contamination is the share of rows the
# model is allowed to flag, and a third of the injected anomalies (odd hours) are
# invisible in these two features, so budgeting for all of them just spends the budget
# on false positives. Two-thirds of 1% is both the honest estimate and the best measured
# F1 (0.491 at 0.0075, against 0.461 at 0.01).
DEFAULT_CONTAMINATION = 0.0075
# Enough trees for the flag set to be stable run to run; past this it only costs time.
DEFAULT_N_ESTIMATORS = 200


def detect(
    frame: pd.DataFrame,
    *,
    contamination: float = DEFAULT_CONTAMINATION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    seed: int = 42,
    features: tuple[str, ...] = MODEL_FEATURES,
) -> pd.Series:
    """Flag transactions the forest finds easy to isolate.

    The model is fitted on the same rows it scores. That is normal for unsupervised
    outlier detection — there is no held-out notion of "clean" data to fit on — but it
    does mean the numbers describe how well the method separates *this* population, not
    how it would generalise to next month's.

    Args:
        frame: A frame carrying ``features``, i.e. the output of
            :func:`sentinel.features.add_features`.
        contamination: Expected share of anomalous rows; sets how many get flagged.
        n_estimators: Number of trees in the forest.
        seed: Seed for the forest's own randomness, so a run is reproducible.
        features: Columns the model sees. Defaults to :data:`MODEL_FEATURES`.

    Returns:
        A boolean Series named ``isoforest_flag``, aligned to ``frame``'s index.

    Raises:
        KeyError: If any of ``features`` is missing from ``frame``.
        ValueError: If ``contamination`` is outside ``(0, 0.5]`` or ``features`` is empty.
    """
    if not features:
        raise ValueError("features must name at least one column")
    missing = [column for column in features if column not in frame.columns]
    if missing:
        raise KeyError(
            f"{missing} missing; pass the frame through sentinel.features.add_features first"
        )
    if not 0.0 < contamination <= 0.5:
        raise ValueError(f"contamination must be in (0, 0.5], got {contamination}")

    forest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
    )
    # fit_predict returns -1 for outliers and 1 for inliers.
    flags = forest.fit_predict(frame[list(features)]) == -1
    return pd.Series(flags, index=frame.index, name="isoforest_flag")
