"""Pipeline orchestration: land -> validate -> transform -> load -> audit.

One call to :func:`run_pipeline` rebuilds the whole warehouse from raw
artifacts. It is deliberately re-entrant -- it drops and recreates every
mart, so a rebuild is a clean function of the inputs rather than an append
that drifts over time.

The freshness table it writes is what ``eap check`` asserts against, and what
GitHub Actions fails on.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import text

from ..config import SETTINGS, Settings
from ..warehouse import execute, execute_many, migrate, rebuild, reset_engines, stable_key
from ..warehouse.engine import get_engine
from . import extract, transform
from .artifacts import generate_all
from .validate import RejectedRow, validate_rows

#: Staging datasets that exist purely as an audit trail and are not promoted
#: to a mart. `metric` is decomposed into the evaluation-run and drift facts;
#: `traffic` is the raw form of the prediction and serving facts.
STAGING_ONLY = {"metric", "traffic"}

STAGING_TABLE = {
    "window": "stg_windows_raw",
    "metric": "stg_metrics_raw",
    "traffic": "stg_traffic_raw",
    "eval_run": "stg_eval_run_raw",
    "prediction": "stg_predictions_raw",
    "serving": "stg_serving_raw",
    "drift": "stg_drift_raw",
    "model_card": "stg_eval_run_raw",
}


@dataclass
class PipelineReport:
    """Everything a caller (or CI) needs to judge a build."""

    pipeline_run_id: str
    status: str = "running"
    rows_landed: int = 0
    rows_accepted: int = 0
    rows_rejected: int = 0
    mart_counts: dict[str, int] = field(default_factory=dict)
    rejection_codes: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"pipeline run   {self.pipeline_run_id}  [{self.status}]",
            f"rows landed    {self.rows_landed}",
            f"rows accepted  {self.rows_accepted}",
            f"rows rejected  {self.rows_rejected}",
            "mart counts:",
        ]
        for mart, count in sorted(self.mart_counts.items()):
            lines.append(f"  {mart:<22} {count:>8}")
        if self.rejection_codes:
            lines.append("rejection codes:")
            for code, count in self.rejection_codes.items():
                lines.append(f"  {code:<34} {count:>6}")
        for warning in self.warnings:
            lines.append(f"warning: {warning}")
        for error in self.errors:
            lines.append(f"error: {error}")
        return "\n".join(lines)


def run_pipeline(
    db_path: Path | None = None,
    settings: Settings | None = None,
    regenerate_artifacts: bool = True,
    fail_fast: bool = False,
    verbose: bool = True,
) -> PipelineReport:
    """Rebuild the warehouse end to end from raw artifacts."""
    settings = settings or SETTINGS
    db_path = Path(db_path or settings.duckdb_path)
    settings.ensure_dirs()
    # `eap --db <path>` may point outside the configured warehouse directory,
    # so the parent has to exist before DuckDB can create the file.
    db_path.parent.mkdir(parents=True, exist_ok=True)
    reset_engines()

    pipeline_run_id = f"pipe_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    report = PipelineReport(pipeline_run_id=pipeline_run_id)

    if regenerate_artifacts:
        runs = generate_all(settings.generated_dir)
        if verbose:
            print(f"generated {len(runs)} fine-tune artifact sets")

    rebuild(db_path)
    if verbose:
        print(f"rebuilt schema at {db_path}")

    batches = extract.extract_all(settings.generated_dir, settings.telemetry_db_path)
    if not batches:
        report.status = "failed"
        report.errors.append("no artifacts found to ingest")
        return report

    # ---- stage 1: land + validate --------------------------------------
    accepted: dict[str, list[dict[str, Any]]] = {}
    origins: dict[tuple[str, str], str] = {}
    window_labels: dict[str, str] = {}
    rejections: list[RejectedRow] = []
    source_origins: dict[str, str] = {}

    for batch in batches:
        source_origins[batch.source_name] = batch.data_origin
        _land(db_path, batch, pipeline_run_id)

        result = validate_rows(batch.dataset_name, batch.rows, batch.source_name)
        report.rows_landed += result.rows_in
        report.rows_accepted += len(result.accepted)
        report.rows_rejected += result.rows_rejected

        if result.rejected:
            rejections.extend(result.rejected)
            if fail_fast:
                report.status = "failed"
                report.errors.append(
                    f"{result.rows_rejected} row(s) failed validation in "
                    f"{batch.dataset_name} from {batch.source_name}"
                )
                _finish(db_path, report, source_origins)
                return report

        accepted.setdefault(batch.dataset_name, []).extend(result.accepted)
        for row in result.accepted:
            if batch.dataset_name == "eval_run":
                origins[(row["experiment_id"], row["model_id"])] = batch.data_origin
            elif batch.dataset_name in {"prediction", "drift", "serving"}:
                origins.setdefault(
                    (row["experiment_id"], row["model_id"]), batch.data_origin
                )
            elif batch.dataset_name == "window":
                window_labels[row["window_id"]] = row["label"]

    # The ingestion run is recorded before its rejections: the rejection table
    # carries a foreign key onto it, so inserting them first would fail.
    _record_ingestion_run(db_path, report, source_origins)

    if rejections:
        _record_rejections(db_path, pipeline_run_id, rejections)
        for code, count in _count_codes(rejections).items():
            report.rejection_codes[code] = report.rejection_codes.get(code, 0) + count

    # ---- stage 2: transform --------------------------------------------
    classes = sorted(
        {
            row["true_label"]
            for row in accepted.get("prediction", [])
            if row.get("true_label")
        }
    )
    features = sorted({row["feature"] for row in accepted.get("drift", [])})

    dims = transform.build_dimensions(
        model_cards=accepted.get("model_card", []),
        eval_runs=accepted.get("eval_run", []),
        features=features,
        window_labels=window_labels,
        classes=classes,
    )
    marts = transform.MartRows()

    transform.build_evaluation_runs(
        marts, dims, accepted.get("eval_run", []), window_labels, origins
    )
    origin_index = transform.overall_index(marts)
    transform.build_predictions(
        marts, dims, accepted.get("prediction", []), origin_index, window_labels, origins
    )
    transform.build_class_evaluation_runs(
        marts, dims, accepted.get("prediction", []), origin_index, origins
    )
    transform.build_length_evaluation_runs(
        marts, accepted.get("prediction", []), origin_index, origins
    )
    transform.build_serving_requests(
        marts, dims, accepted.get("serving", []), origin_index, origins
    )
    transform.build_drift_measurements(
        marts, dims, accepted.get("drift", []), window_labels, origins
    )

    # ---- stage 3: load --------------------------------------------------
    _load_dimensions(db_path, dims)
    _load_marts(db_path, marts)

    report.mart_counts = {
        "fact_evaluation_run": len(marts.evaluation_run),
        "fact_prediction": len(marts.prediction),
        "fact_serving_request": len(marts.serving_request),
        "fact_drift_measurement": len(marts.drift_measurement),
        "dim_model": len(dims.models),
        "dim_experiment": len(dims.experiments),
        "dim_feature": len(dims.features),
        "dim_slice": len(dims.slices),
        "dim_date": len(dims.dates),
    }
    _write_freshness(db_path, report)

    report.status = "success"
    _finish(db_path, report, source_origins)
    if verbose:
        print(report.summary())
    return report


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _land(db_path: Path, batch: extract.SourceBatch, pipeline_run_id: str) -> None:
    """Write a batch to its staging table as text, without interpreting it."""
    table = STAGING_TABLE.get(batch.dataset_name)
    if table is None:
        return
    rows = [
        {
            "_ingest_run_id": pipeline_run_id,
            "_source_name": batch.source_name,
            "_source_row_id": str(row.get("request_id") or row.get("example_id") or i),
            "payload": json.dumps(row, default=str, sort_keys=True),
            "landed_at": datetime.now(timezone.utc).replace(tzinfo=None),
        }
        for i, row in enumerate(batch.rows)
    ]
    execute_many(
        db_path,
        f"INSERT INTO {table} (_ingest_run_id, _source_name, _source_row_id, payload, landed_at) "
        "VALUES (:_ingest_run_id, :_source_name, :_source_row_id, :payload, :landed_at)",
        rows,
    )


def _record_rejections(db_path: Path, pipeline_run_id: str, rejections: list[RejectedRow]) -> None:
    execute_many(
        db_path,
        "INSERT INTO ingestion_rejection (rejection_id, pipeline_run_id, source_name, "
        "dataset_name, source_row_id, error_count, error_codes, error_messages, "
        "raw_payload, rejected_at) VALUES (:rejection_id, :pipeline_run_id, :source_name, "
        ":dataset_name, :source_row_id, :error_count, :error_codes, :error_messages, "
        ":raw_payload, :rejected_at)",
        [
            {
                "rejection_id": stable_key(pipeline_run_id, r.source_name, r.dataset_name, r.source_row_id, i),
                "pipeline_run_id": pipeline_run_id,
                "source_name": r.source_name,
                "dataset_name": r.dataset_name,
                "source_row_id": r.source_row_id,
                "error_count": r.error_count,
                "error_codes": json.dumps(r.error_codes),
                "error_messages": json.dumps(r.error_messages),
                "raw_payload": r.raw_payload,
                "rejected_at": datetime.now(timezone.utc).replace(tzinfo=None),
            }
            for i, r in enumerate(rejections)
        ],
    )


def _record_ingestion_run(
    db_path: Path, report: PipelineReport, source_origins: dict[str, str]
) -> None:
    origin = "real" if "real" in source_origins.values() else "synthetic"
    execute(
        db_path,
        "INSERT INTO ingestion_run (pipeline_run_id, started_at, finished_at, status, "
        "source_name, data_origin, rows_landed, rows_accepted, rows_rejected, error) "
        "VALUES (:id, CAST(now() AS TIMESTAMP), NULL, :status, :src, :origin, :landed, "
        ":accepted, :rejected, :error)",
        {
            "id": report.pipeline_run_id,
            "status": report.status,
            "src": "+".join(sorted(source_origins)),
            "origin": origin,
            "landed": report.rows_landed,
            "accepted": report.rows_accepted,
            "rejected": report.rows_rejected,
            "error": "; ".join(report.errors) or None,
        },
    )


def _finish(db_path: Path, report: PipelineReport, source_origins: dict[str, str]) -> None:
    execute(
        db_path,
        "UPDATE ingestion_run SET finished_at = CAST(now() AS TIMESTAMP), status = :status, "
        "error = :error WHERE pipeline_run_id = :id",
        {
            "id": report.pipeline_run_id,
            "status": report.status,
            "error": "; ".join(report.errors) or None,
        },
    )
    del source_origins


def _load_dimensions(db_path: Path, dims: transform.DimensionRows) -> None:
    _upsert(db_path, "dim_model", list(dims.models.values()), _MODEL_COLS)
    _upsert(db_path, "dim_experiment", list(dims.experiments.values()), _EXPERIMENT_COLS)
    _upsert(db_path, "dim_feature", list(dims.features.values()), _FEATURE_COLS)
    _upsert(db_path, "dim_slice", list(dims.slices.values()), _SLICE_COLS)
    _upsert(db_path, "dim_date", list(dims.dates.values()), _DATE_COLS)


def _load_marts(db_path: Path, marts: transform.MartRows) -> None:
    _insert(db_path, "fact_evaluation_run", marts.evaluation_run, _EVAL_RUN_COLS)
    _insert(db_path, "fact_prediction", marts.prediction, _PREDICTION_COLS)
    _insert(db_path, "fact_serving_request", marts.serving_request, _SERVING_COLS)
    _insert(db_path, "fact_drift_measurement", marts.drift_measurement, _DRIFT_COLS)


def _insert(db_path: Path, table: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    collist = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    execute_many(db_path, f"INSERT INTO {table} ({collist}) VALUES ({placeholders})", rows)


def _upsert(db_path: Path, table: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Replace dimension members wholesale.

    Dimensions are small and rebuilt from scratch each run, so a delete +
    insert is both simpler and safer than a per-row upsert: a dimension can
    never retain a member that no longer has any supporting fact row.
    """
    if not rows:
        return
    execute(db_path, f"DELETE FROM {table}")
    _insert(db_path, table, rows, columns)


def _write_freshness(db_path: Path, report: PipelineReport) -> None:
    """Record per-mart freshness, which is what CI asserts on."""
    engine = get_engine(db_path)
    specs = [
        ("fact_evaluation_run", "date_id"),
        ("fact_prediction", "date_id"),
        ("fact_serving_request", "date_id"),
        ("fact_drift_measurement", "date_id"),
    ]
    rows = []
    for mart, date_col in specs:
        count, max_date = _mart_bounds(engine, mart, date_col)
        rows.append(
            {
                "mart_name": mart,
                "row_count": count,
                "max_date_id": max_date,
                "max_loaded_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "source_row_count": count,
                "is_stale": count == 0,
                "staleness_reason": "mart is empty" if count == 0 else None,
            }
        )
    execute_many(
        db_path,
        "INSERT OR REPLACE INTO mart_freshness (mart_name, row_count, max_date_id, "
        "max_loaded_at, source_row_count, is_stale, staleness_reason) VALUES "
        "(:mart_name, :row_count, :max_date_id, :max_loaded_at, :source_row_count, "
        ":is_stale, :staleness_reason)",
        rows,
    )


def _mart_bounds(engine, mart: str, date_col: str) -> tuple[int, Any]:
    with engine.connect() as conn:
        row = conn.execute(
            text(f"SELECT COUNT(*) AS n, MAX({date_col}) AS max_date FROM {mart}")
        ).fetchone()
    return int(row[0]), row[1]


def _count_codes(rejections: list[RejectedRow]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rejections:
        for code in r.error_codes:
            counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


_MODEL_COLS = [
    "model_id", "base_model", "arm", "method", "quantisation", "adapter_rank",
    "trainable_params", "total_params", "trainable_pct", "source_dataset",
    "created_at", "is_champion", "data_origin",
]
_EXPERIMENT_COLS = [
    "experiment_id", "name", "hypothesis", "owner", "started_at", "ended_at",
    "status", "data_source", "tags", "data_origin",
]
_FEATURE_COLS = ["feature_id", "feature_type", "source_system", "is_drift_tracked", "description"]
_SLICE_COLS = ["slice_id", "slice_type", "slice_name", "ordinal", "definition", "source_system"]
_DATE_COLS = ["date_id", "year", "quarter", "month", "week", "day_of_week", "is_weekend"]
_EVAL_RUN_COLS = [
    "evaluation_run_key", "experiment_id", "model_id", "slice_id", "date_id",
    "window_id", "n_samples", "n_correct", "accuracy", "accuracy_ci_low",
    "accuracy_ci_high", "macro_f1", "ece", "mce", "brier", "trainable_params",
    "seed", "evaluation_seconds", "source_system", "data_origin",
]
_PREDICTION_COLS = [
    "prediction_key", "evaluation_run_key", "experiment_id", "model_id", "date_id",
    "example_id", "window_id", "true_label", "predicted_label", "prob_true_label",
    "prob_predicted_label", "max_probability", "confidence_margin", "entropy",
    "is_correct", "is_abstained", "n_tokens", "doc_length_bucket_id", "regime_slice_id",
    "data_origin",
]
_SERVING_COLS = [
    "serving_request_key", "model_id", "date_id", "slice_id", "quantisation",
    "batch_size", "input_tokens", "latency_ms", "model_size_bytes", "peak_memory_mb",
    "throughput_tps", "request_index", "data_origin",
]
_DRIFT_COLS = [
    "drift_measurement_key", "feature_id", "model_id", "date_id", "slice_id",
    "window_id", "measurement_ts", "metric_name", "metric_value", "psi",
    "ks_statistic", "ks_pvalue", "js_divergence", "baseline_value", "current_value",
    "sample_size", "threshold_moderate", "threshold_severe", "severity", "data_origin",
]


def migrate_warehouse(db_path: Path | None = None, verbose: bool = True) -> None:
    """Apply pending migrations without rebuilding."""
    db_path = Path(db_path or SETTINGS.duckdb_path)
    applied = migrate(db_path, verbose=verbose)
    if verbose and not applied:
        print("warehouse schema already up to date")
