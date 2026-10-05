"""Warehouse quality gate.

``eap check`` is what GitHub Actions runs after rebuilding the warehouse. It
asserts four independent things, and CI fails if any of them fails:

1. **No rejected rows.** Validation rejections are counted, not tolerated.
2. **No orphaned facts.** Every fact key resolves to a dimension member, and
   every slice is referenced by at least one fact row.
3. **No stale mart.** Every mart is non-empty and its newest date is not
   behind the newest date present anywhere in the warehouse, which is what a
   partial or truncated load looks like.
4. **The last ingestion run succeeded.**

Each check returns a :class:`CheckResult` with a human-readable failure
reason, so a CI log says which invariant broke rather than just exiting 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .warehouse import query, require_warehouse

MART_NAMES = (
    "fact_evaluation_run",
    "fact_prediction",
    "fact_serving_request",
    "fact_drift_measurement",
)

#: mart -> (staging table it is derived from, timestamp field in that payload).
#:
#: Freshness must be judged per mart against the source it came from. A single
#: global frontier is wrong here: drift measurements legitimately extend past
#: the date of the evaluation run that produced the model, so comparing every
#: mart to the newest date in the warehouse would flag a perfectly complete
#: build as stale. What actually indicates a broken load is a mart that lags
#: the newest record in *its own* staging table.
FRESHNESS_SPECS: tuple[tuple[str, str, str], ...] = (
    ("fact_evaluation_run", "stg_eval_run_raw", "started_at"),
    ("fact_prediction", "stg_eval_run_raw", "started_at"),
    ("fact_serving_request", "stg_serving_raw", "ts"),
    ("fact_drift_measurement", "stg_drift_raw", "measured_at"),
)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "detail": self.detail,
            "evidence": self.evidence,
        }


@dataclass
class GateReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def summary(self) -> str:
        lines = []
        for check in self.checks:
            mark = "PASS" if check.passed else "FAIL"
            lines.append(f"[{mark}] {check.name}: {check.detail}")
        lines.append("")
        lines.append(
            "warehouse gate: PASS" if self.passed else f"warehouse gate: FAIL ({len(self.failures)} check(s))"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


def check_rejections(db_path: Path | str, max_rejections: int = 0) -> CheckResult:
    """Count rows rejected by validation since the last successful build."""
    path = require_warehouse(db_path)
    row = query(
        path,
        """
        SELECT COUNT(*) AS n_rejections,
               COALESCE(SUM(error_count), 0) AS n_errors
        FROM ingestion_rejection
        """,
    ).iloc[0]
    n_rejections = int(row["n_rejections"])

    by_code = query(
        path,
        """
        SELECT dataset_name, error_codes, COUNT(*) AS n
        FROM ingestion_rejection
        GROUP BY dataset_name, error_codes
        ORDER BY n DESC
        LIMIT 20
        """,
    )
    passed = n_rejections <= max_rejections
    return CheckResult(
        name="validation_rejections",
        passed=passed,
        detail=(
            f"{n_rejections} rejected row(s) (limit {max_rejections})"
            if passed
            else f"{n_rejections} rejected row(s) exceeds limit {max_rejections}"
        ),
        evidence={
            "n_rejections": n_rejections,
            "n_errors": int(row["n_errors"]),
            "by_dataset": by_code.to_dict(orient="records"),
        },
    )


def check_referential_integrity(db_path: Path | str) -> CheckResult:
    """No orphaned fact keys; every slice is referenced."""
    path = require_warehouse(db_path)

    orphans = query(
        path,
        """
        SELECT
          (SELECT COUNT(*) FROM fact_prediction p
             LEFT JOIN fact_evaluation_run r USING (evaluation_run_key)
            WHERE r.evaluation_run_key IS NULL)                        AS orphan_predictions,
          (SELECT COUNT(*) FROM fact_evaluation_run r
             LEFT JOIN dim_model m USING (model_id)
             LEFT JOIN dim_slice s USING (slice_id)
             LEFT JOIN dim_date d USING (date_id)
             LEFT JOIN dim_experiment e USING (experiment_id)
            WHERE m.model_id IS NULL OR s.slice_id IS NULL
               OR d.date_id IS NULL OR e.experiment_id IS NULL)       AS orphan_evaluation_runs,
          (SELECT COUNT(*) FROM fact_serving_request r
             LEFT JOIN dim_model m USING (model_id)
             LEFT JOIN dim_date d USING (date_id)
            WHERE m.model_id IS NULL OR d.date_id IS NULL)            AS orphan_serving_requests,
          (SELECT COUNT(*) FROM fact_drift_measurement r
             LEFT JOIN dim_feature f USING (feature_id)
             LEFT JOIN dim_date d USING (date_id)
            WHERE f.feature_id IS NULL OR d.date_id IS NULL)          AS orphan_drift_measurements,
          (SELECT COUNT(*) FROM fact_prediction p
             LEFT JOIN dim_slice l ON l.slice_id = p.doc_length_bucket_id
            WHERE p.doc_length_bucket_id IS NOT NULL
              AND l.slice_id IS NULL)                                 AS orphan_length_buckets,
          (SELECT COUNT(*) FROM fact_prediction p
             LEFT JOIN dim_slice g ON g.slice_id = p.regime_slice_id
            WHERE p.regime_slice_id IS NOT NULL
              AND g.slice_id IS NULL)                                 AS orphan_regimes,
          (SELECT COUNT(*) FROM dim_slice s
            WHERE NOT EXISTS (SELECT 1 FROM fact_evaluation_run r
                               WHERE r.slice_id = s.slice_id))        AS unreferenced_slices
        """,
    ).iloc[0]

    total = int(sum(int(v) for v in orphans if v is not None))
    return CheckResult(
        name="referential_integrity",
        passed=total == 0,
        detail=(
            "no orphaned fact rows and every slice is referenced"
            if total == 0
            else f"{total} orphan/unreferenced row(s) detected"
        ),
        evidence={k: (int(v) if v is not None else 0) for k, v in orphans.items()},
    )


def check_freshness(db_path: Path | str) -> CheckResult:
    """Every mart is populated and not lagging its own source."""
    path = require_warehouse(db_path)

    recorded = {
        row["mart_name"]: row
        for _, row in query(
            path, "SELECT mart_name, row_count, max_date_id FROM mart_freshness"
        ).iterrows()
    }

    problems: list[str] = []
    evidence: dict[str, Any] = {"marts": {}}

    for mart, staging_table, payload_field in FRESHNESS_SPECS:
        entry: dict[str, Any] = {}
        row = recorded.get(mart)
        if row is None:
            problems.append(f"{mart} is absent from mart_freshness")
            evidence["marts"][mart] = {"present": False}
            continue

        count = int(row["row_count"] or 0)
        mart_max = row["max_date_id"]
        entry.update(
            {
                "present": True,
                "row_count": count,
                "mart_max_date_id": str(mart_max) if mart_max is not None else None,
                "source_table": staging_table,
                "source_field": payload_field,
            }
        )

        if count == 0:
            problems.append(f"{mart} is empty")
            evidence["marts"][mart] = entry
            continue

        source = query(
            path,
            f"""
            SELECT MAX(TRY_CAST(json_extract_string(payload, '$.{payload_field}') AS TIMESTAMP)) AS src_max
            FROM {staging_table}
            """,
        ).iloc[0]["src_max"]
        source_max = pd.Timestamp(source).date() if source is not None else None
        entry["source_max_date_id"] = str(source_max) if source_max else None

        if source_max is None or mart_max is None:
            entry["lag_days_vs_source"] = None
        else:
            lag = (source_max - pd.Timestamp(mart_max).date()).days
            entry["lag_days_vs_source"] = lag
            if lag > 0:
                problems.append(
                    f"{mart} is {lag} day(s) behind {staging_table}"
                )
        evidence["marts"][mart] = entry

    return CheckResult(
        name="mart_freshness",
        passed=not problems,
        detail=(
            f"all {len(FRESHNESS_SPECS)} marts populated and current against their sources"
            if not problems
            else "; ".join(problems)
        ),
        evidence=evidence,
    )


def check_last_ingestion(db_path: Path | str) -> CheckResult:
    """The most recent pipeline run finished successfully."""
    path = require_warehouse(db_path)
    row = query(
        path,
        """
        SELECT pipeline_run_id, status, rows_landed, rows_accepted, rows_rejected, error
        FROM ingestion_run ORDER BY started_at DESC LIMIT 1
        """,
    )
    if row.empty:
        return CheckResult(
            name="last_ingestion_run",
            passed=False,
            detail="no ingestion run recorded",
            evidence={},
        )
    latest = row.iloc[0]
    passed = latest["status"] == "success"
    return CheckResult(
        name="last_ingestion_run",
        passed=passed,
        detail=(
            f"run {latest['pipeline_run_id']} status={latest['status']} "
            f"landed={latest['rows_landed']} accepted={latest['rows_accepted']} "
            f"rejected={latest['rows_rejected']}"
        ),
        evidence={
            "pipeline_run_id": latest["pipeline_run_id"],
            "status": latest["status"],
            "rows_landed": int(latest["rows_landed"]),
            "rows_accepted": int(latest["rows_accepted"]),
            "rows_rejected": int(latest["rows_rejected"]),
            "error": latest["error"],
        },
    )


def check_required_coverage(db_path: Path | str) -> CheckResult:
    """The warehouse actually contains enough for the analyses to mean anything."""
    path = require_warehouse(db_path)
    row = query(
        path,
        """
        SELECT
          (SELECT COUNT(DISTINCT arm) FROM dim_model WHERE arm <> 'none')          AS n_arms,
          (SELECT COUNT(DISTINCT quantisation) FROM dim_model)                     AS n_quantisations,
          (SELECT COUNT(DISTINCT slice_type) FROM dim_slice)                      AS n_slice_types,
          (SELECT COUNT(DISTINCT slice_type) FROM dim_slice)                      AS n_slice_types,
          (SELECT COUNT(DISTINCT feature_id) FROM fact_drift_measurement)         AS n_features,
          (SELECT COUNT(*) FROM fact_prediction WHERE is_correct IS NOT NULL)     AS n_labelled,
          (SELECT COUNT(DISTINCT true_label) FROM fact_prediction
             WHERE true_label IS NOT NULL)                                        AS n_classes
        """,
    ).iloc[0]

    arms = int(row["n_arms"])
    quantisations = int(row["n_quantisations"])
    features = int(row["n_features"])
    labelled = int(row["n_labelled"])
    classes = int(row["n_classes"])

    # Only these three families are always derivable from predictions. Drift
    # regimes require a source with labelled monitoring windows, so a build
    # without telemetry legitimately has none and is not a defect.
    required_families = {"overall", "class", "length_bucket"}
    present_families = set(
        query(path, "SELECT DISTINCT slice_type FROM dim_slice")["slice_type"]
    )
    missing = required_families - present_families

    problems = []
    if arms < 2:
        problems.append("fewer than 2 fine-tune arms loaded; arm comparison impossible")
    if quantisations < 2:
        problems.append("fewer than 2 quantisations loaded; quantisation comparison impossible")
    if missing:
        problems.append(
            "missing slice family/families: " + ", ".join(sorted(missing))
        )
    if features < 1:
        problems.append("no drift features loaded")
    if labelled < 100:
        problems.append(f"only {labelled} labelled predictions")
    if classes < 2:
        problems.append("fewer than 2 classes; per-class analysis is meaningless")

    return CheckResult(
        name="analytical_coverage",
        passed=not problems,
        detail=(
            f"{arms} arms, {quantisations} quantisations, {len(present_families)} slice families, "
            f"{features} drift features, {labelled} labelled predictions, {classes} classes"
        )
        if not problems
        else "; ".join(problems),
        evidence={
            "n_arms": arms,
            "n_quantisations": quantisations,
            "slice_families": sorted(present_families),
            "required_families": sorted(required_families),
            "has_drift_regimes": "drift_window" in present_families,
            "n_features": features,
            "n_labelled_predictions": labelled,
            "n_classes": classes,
        },
    )


def run_gate(db_path: Path | str, max_rejections: int = 0) -> GateReport:
    """Run every check and return a combined report."""
    return GateReport(
        checks=[
            check_last_ingestion(db_path),
            check_rejections(db_path, max_rejections=max_rejections),
            check_referential_integrity(db_path),
            check_freshness(db_path),
            check_required_coverage(db_path),
        ]
    )
