"""Console summary and the JSON run artifact.

Two audiences, one set of numbers. The console table is for whoever just typed the
command; the JSON artifact is for everything else — diffing two runs, charting precision
over time, or attaching evidence to a pull request. They are built from the same
:class:`~sentinel.pipeline.RunResult` so they cannot drift apart.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sentinel.monitoring import run_summary
from sentinel.pipeline import RunResult

DEFAULT_REPORT_DIR = Path("reports")
# Alert lists are for reading, not for archiving every hit: a 50k run flags a few
# hundred rows and nobody opens item 300. The report always records the true total
# alongside the truncated list so the cap can never be mistaken for the whole story.
MAX_ALERTS = 100
ALERT_FIELDS: tuple[str, ...] = (
    "transaction_id",
    "timestamp",
    "user_id",
    "amount",
    "merchant_category",
    "amount_zscore",
    "txn_count_1h",
    "is_anomaly",
    "anomaly_type",
)


def build(result: RunResult) -> dict[str, Any]:
    """Assemble the full run artifact as a JSON-safe dict."""
    detectors = list(result.metrics.index)
    frame = result.frame
    alerts = frame[frame[[f"{name}_flag" for name in detectors]].any(axis=1)]
    ranked = alerts.sort_values("amount_zscore", ascending=False)

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "params": result.params,
        "summary": run_summary(frame, detectors),
        "timings_ms": result.timings,
        "metrics": result.metrics.to_dict(orient="index"),
        "recall_by_type": result.recall_by_type.to_dict(orient="index"),
        "alerts_total": int(len(alerts)),
        "alerts_included": int(min(len(alerts), MAX_ALERTS)),
        "alerts": [
            {
                **{field: _jsonable(row[field]) for field in ALERT_FIELDS},
                "detectors": [name for name in detectors if row[f"{name}_flag"]],
            }
            for _, row in ranked.head(MAX_ALERTS).iterrows()
        ],
    }


def write(report: dict[str, Any], directory: Path = DEFAULT_REPORT_DIR) -> Path:
    """Write ``report`` under ``directory`` and return the path.

    The filename carries the run's timestamp and seed, so repeated runs accumulate
    rather than silently overwriting the evidence of the last one.
    """
    directory.mkdir(parents=True, exist_ok=True)
    # Trim the timezone offset before it becomes a "+" in a filename.
    stamp = report["generated_at"][:19].replace(":", "").replace("-", "")
    base = f"run-{stamp}-seed{report['params']['seed']}"
    path = directory / f"{base}.json"
    # The stamp only resolves to the second, so two quick runs of the same seed would
    # land on one name. Suffix rather than overwrite: this function promises to
    # accumulate evidence, and quietly deleting the previous run would break that.
    attempt = 1
    while path.exists():
        path = directory / f"{base}-{attempt}.json"
        attempt += 1
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def format_console(result: RunResult) -> str:
    """Render a run as the terminal summary."""
    frame = result.frame
    detectors = list(result.metrics.index)
    summary = run_summary(frame, detectors)
    injected = summary["anomalies_injected"]

    lines = [
        f"data-sentinel — {summary['rows_processed']:,} transactions, "
        f"{frame['user_id'].nunique():,} users, seed {result.params['seed']}",
        f"{injected:,} injected anomalies ({injected / len(frame):.2%})",
        "",
        result.metrics.round(3).to_string(),
        "",
        "recall by anomaly type",
        result.recall_by_type.round(3).to_string(),
        "",
        f"detector agreement {summary['agreement']:.2f} "
        f"({summary['flagged_by_all']} of {summary['flagged_by_any']} alerts flagged by both)",
        "stage timings  " + "  ".join(f"{k}={v:.0f}ms" for k, v in result.timings.items()),
    ]
    # Naming the blind spots outright: a type nothing catches is the most useful line in
    # the report and the easiest one to miss in a table of small numbers.
    missed = result.recall_by_type.index[result.recall_by_type.max(axis=1) == 0]
    if len(missed):
        lines += ["", f"caught by no detector: {', '.join(missed)}"]
    return "\n".join(lines)


def _jsonable(value: object) -> object:
    """Coerce numpy scalars and timestamps into something ``json`` will accept."""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value.item() if hasattr(value, "item") else value
