"""End-to-end run: generate, engineer features, detect, score.

This is the module that turns a pile of functions into a thing you can run and argue
with. The scoring half matters more than the orchestration half: a detector that is
never measured against the labels is just a random flag generator with good intentions,
and the per-anomaly-type breakdown is what stops a single headline number from hiding
which kinds of fraud the pipeline is blind to.
"""

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from sentinel.detectors import isoforest, zscore
from sentinel.features import add_features
from sentinel.generate import generate_transactions

METRIC_COLUMNS: tuple[str, ...] = ("flagged", "precision", "recall", "f1")


@dataclass(frozen=True)
class RunResult:
    """Everything one run produced, kept together so callers need not re-derive it.

    Attributes:
        frame: Transactions, features, and one ``<detector>_flag`` column per detector.
        metrics: One row per detector, columns :data:`METRIC_COLUMNS`, scored against
            ``is_anomaly``.
        recall_by_type: One row per injected anomaly type, one column per detector.
            Precision has no meaning per type — a false positive belongs to no type — so
            only recall is broken down.
    """

    frame: pd.DataFrame
    metrics: pd.DataFrame
    recall_by_type: pd.DataFrame


def run(
    *,
    n: int = 50_000,
    seed: int = 42,
    n_users: int = 500,
    anomaly_rate: float = 0.01,
    zscore_threshold: float = zscore.DEFAULT_THRESHOLD,
    contamination: float = isoforest.DEFAULT_CONTAMINATION,
) -> RunResult:
    """Generate a dataset, score every detector against it, and return the lot.

    Args:
        n: Transactions to generate.
        seed: Seed shared by the generator and the forest, so a run is reproducible end
            to end from this one number.
        n_users: Distinct cardholders.
        anomaly_rate: Fraction of rows the generator corrupts.
        zscore_threshold: Sigma at which the baseline detector flags.
        contamination: Share of rows the forest is allowed to flag.

    Returns:
        A :class:`RunResult`.
    """
    frame = add_features(
        generate_transactions(n, seed=seed, n_users=n_users, anomaly_rate=anomaly_rate)
    )
    flags = {
        "zscore": zscore.detect(frame, threshold=zscore_threshold),
        "isoforest": isoforest.detect(frame, contamination=contamination, seed=seed),
    }
    for name, flag in flags.items():
        frame[f"{name}_flag"] = flag

    return RunResult(
        frame=frame,
        metrics=_score(frame["is_anomaly"], flags),
        recall_by_type=_recall_by_type(frame, flags),
    )


def _score(labels: pd.Series, flags: dict[str, pd.Series]) -> pd.DataFrame:
    """Precision, recall and F1 for each detector against the injected labels."""
    metrics = pd.DataFrame(
        {
            name: {
                "flagged": int(flag.sum()),
                # zero_division=0: a detector that flags nothing has undefined precision,
                # and scoring it 0 is the reading that will not flatter it.
                "precision": precision_score(labels, flag, zero_division=0),
                "recall": recall_score(labels, flag, zero_division=0),
                "f1": f1_score(labels, flag, zero_division=0),
            }
            for name, flag in flags.items()
        }
    ).T[list(METRIC_COLUMNS)]
    return metrics.astype({"flagged": int})


def _recall_by_type(frame: pd.DataFrame, flags: dict[str, pd.Series]) -> pd.DataFrame:
    """Share of each injected anomaly type that each detector caught."""
    anomalies = frame[frame["is_anomaly"]]
    return pd.DataFrame(
        {name: anomalies.groupby("anomaly_type")[f"{name}_flag"].mean() for name in flags}
    )


def format_summary(result: RunResult) -> str:
    """Render a run as the console summary, kept a plain string so it can be tested.

    Phase 4 replaces this with a proper report module; until then it exists so the CLI
    has something honest to print rather than a bare frame dump.
    """
    frame = result.frame
    anomalies = int(frame["is_anomaly"].sum())
    lines = [
        f"data-sentinel — {len(frame):,} transactions, {frame['user_id'].nunique():,} users",
        f"{anomalies:,} injected anomalies ({anomalies / len(frame):.2%})",
        "",
        result.metrics.round(3).to_string(),
        "",
        "recall by anomaly type",
        result.recall_by_type.round(3).to_string(),
    ]
    # Naming the blind spots outright: a type nothing catches is the most useful line in
    # the report and the easiest one to miss in a table of small numbers.
    missed = result.recall_by_type.index[result.recall_by_type.max(axis=1) == 0]
    if len(missed):
        lines += ["", f"caught by no detector: {', '.join(missed)}"]
    return "\n".join(lines)
