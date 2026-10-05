"""FastAPI service exposing the analytical queries and a read-only SQL endpoint.

Every analytical question the platform answers is available as a JSON
endpoint that returns the same structures the CLI prints, so the dashboard
and any external consumer see identical numbers.

``/sql`` accepts a single ``SELECT``/``WITH`` statement. It is deliberately
restricted rather than trusted: only one statement, read-only keywords only,
and a row cap. A warehouse behind a BI dashboard should not be a general
query engine for whoever finds the port.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..analytics import queries as Q
from ..config import SETTINGS
from ..quality import run_gate
from ..serialize import jsonable
from ..warehouse import query as run_query
from ..warehouse import require_warehouse

MAX_ROWS = 10_000

#: Anything that could mutate or escape the warehouse.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|copy|"
    r"install|load|pragma|set|reset|call|export|import|truncate|vacuum|"
    r"checkpoint|force)\b",
    re.IGNORECASE,
)
_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


class SqlRequest(BaseModel):
    sql: str = Field(..., min_length=1, description="A single read-only SELECT/WITH statement")
    limit: int = Field(MAX_ROWS, ge=1, le=MAX_ROWS)


class ErrorResponse(BaseModel):
    error: str


def _clean(value: Any) -> Any:
    """Make a query result JSON-safe.

    Shares :func:`eval_analytics.serialize.jsonable` with the CLI so both
    surfaces encode a number as a number.
    """
    return jsonable(value)


def _validate_sql(sql: str) -> None:
    stripped = sql.strip()
    if not _ALLOWED_START.match(stripped):
        raise HTTPException(400, "only SELECT or WITH statements are allowed")
    if _FORBIDDEN.search(stripped):
        raise HTTPException(400, "statement contains a write or DDL keyword")
    # A trailing semicolon followed by more SQL would smuggle a second
    # statement past a naive single-statement check.
    body = stripped.rstrip(";").strip()
    if not body:
        raise HTTPException(400, "empty statement")
    if not stripped.endswith(";") and ";" in body:
        raise HTTPException(400, "exactly one statement is allowed")


def create_app(db_path: Path | str | None = None) -> FastAPI:
    """Build the API bound to a warehouse file."""
    resolved = Path(db_path or SETTINGS.duckdb_path)

    app = FastAPI(
        title="Evaluation Analytics API",
        version="0.1.0",
        summary=(
            "SQL-backed evaluation and telemetry analytics. Read-only over a DuckDB "
            "star schema of model evaluation runs, per-example predictions, serving "
            "requests and drift measurements."
        ),
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )

    def _db() -> Path:
        try:
            return require_warehouse(resolved)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

    def _records(sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        frame = run_query(_db(), sql, params)
        return _clean(frame.to_dict(orient="records"))

    @app.get("/health")
    def health() -> dict[str, Any]:
        try:
            path = _db()
        except HTTPException as exc:
            return {"status": "unavailable", "detail": exc.detail}
        return {"status": "ok", "warehouse": path.name}

    @app.get("/meta/dimensions")
    def dimensions() -> dict[str, Any]:
        return {
            "models": _records("SELECT * FROM dim_model ORDER BY arm, quantisation"),
            "experiments": _records("SELECT * FROM dim_experiment ORDER BY experiment_id"),
            "features": _records("SELECT * FROM dim_feature ORDER BY feature_id"),
            "slices": _records("SELECT * FROM dim_slice ORDER BY slice_type, ordinal, slice_id"),
        }

    @app.get("/meta/arms")
    def arms() -> dict[str, Any]:
        return {"arms": _clean(Q.available_arms(_db()).to_dict(orient="records"))}

    @app.get("/meta/summary")
    def summary() -> dict[str, Any]:
        return _clean(Q.data_quality_summary(_db()))

    @app.get("/meta/quality-gate")
    def quality_gate() -> JSONResponse:
        gate = run_gate(_db())
        return JSONResponse(_clean(gate.to_dict()), status_code=200 if gate.passed else 503)

    # -- the six analytical questions ------------------------------------

    @app.get("/analysis/lora-vs-full")
    def lora_vs_full(
        arm_a: str = "lora",
        arm_b: str = "full_finetune",
        quantisation: str = "fp32",
        n_boot: int = Query(4000, ge=100, le=50_000),
    ) -> dict[str, Any]:
        try:
            payload = Q.lora_vs_full_finetune(
                _db(), arm_a=arm_a, arm_b=arm_b, quantisation=quantisation, n_boot=n_boot
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return _clean(payload)

    @app.get("/analysis/calibration")
    def calibration(
        quantisation: str = "fp32",
        arms: str = "lora,full_finetune",
        n_bins: int = Query(15, ge=2, le=50),
    ) -> dict[str, Any]:
        return _clean(
            Q.calibration_by_arm(
                _db(), arms=tuple(a.strip() for a in arms.split(",") if a.strip()),
                quantisation=quantisation, n_bins=n_bins,
            )
        )

    @app.get("/analysis/quantisation")
    def quantisation(
        reference: str = "fp32", target: str = "int8"
    ) -> dict[str, Any]:
        return _clean(
            Q.quantisation_comparison(
                _db(), reference_quantisation=reference, target_quantisation=target
            )
        )

    @app.get("/analysis/drift")
    def drift(
        feature: str | None = None,
        metric: str | None = None,
        moderate: float = Query(0.10, ge=0.0),
        severe: float = Query(0.25, ge=0.0),
    ) -> dict[str, Any]:
        return _clean(
            Q.drift_trend(
                _db(), feature=feature, metric=metric,
                metric_value=moderate, severe_value=severe,
            )
        )

    @app.get("/analysis/length-buckets")
    def length_buckets(quantisation: str = "fp32") -> dict[str, Any]:
        payload = _clean(Q.error_by_length_bucket(_db(), quantisation=quantisation))
        payload["confusions_lora"] = _clean(
            Q.confusion_by_length(_db(), arm="lora", slice_id="len_17_32")
        )
        return payload

    @app.get("/analysis/slice-losses")
    def slice_losses(
        arm: str = "lora",
        against: str = "full_finetune",
        quantisation: str = "fp32",
        n_boot: int = Query(2000, ge=100, le=50_000),
    ) -> dict[str, Any]:
        return _clean(
            Q.slice_losses(
                _db(), arm=arm, against=against, quantisation=quantisation, n_boot=n_boot
            )
        )

    # -- read-only SQL ----------------------------------------------------

    @app.post("/sql")
    def sql_endpoint(request: SqlRequest) -> dict[str, Any]:
        _validate_sql(request.sql)
        try:
            frame = run_query(_db(), request.sql)
        except Exception as exc:  # surfaced as 400: the caller's SQL is wrong
            raise HTTPException(400, f"query failed: {exc}") from exc
        truncated = len(frame) > request.limit
        return {
            "columns": list(frame.columns),
            "rows": _clean(frame.head(request.limit).to_dict(orient="records")),
            "row_count": int(len(frame)),
            "truncated": truncated,
            "limit": request.limit,
        }

    @app.get("/query/runs")
    def evaluation_runs(
        arm: str | None = None,
        slice_type: str | None = None,
        limit: int = Query(200, ge=1, le=MAX_ROWS),
    ) -> dict[str, Any]:
        clauses, params = [], {}
        if arm:
            clauses.append("m.arm = :arm")
            params["arm"] = arm
        if slice_type:
            clauses.append("s.slice_type = :slice_type")
            params["slice_type"] = slice_type
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = _records(
            f"""
            SELECT r.experiment_id, m.model_id, m.arm, m.quantisation,
                   m.trainable_params, s.slice_id, s.slice_type, s.slice_name,
                   r.n_samples, r.accuracy, r.accuracy_ci_low, r.accuracy_ci_high,
                   r.macro_f1, r.ece, r.mce, r.brier, r.date_id, r.data_origin
            FROM fact_evaluation_run r
            JOIN dim_model m ON m.model_id = r.model_id
            JOIN dim_slice s ON s.slice_id = r.slice_id
            {where}
            ORDER BY r.date_id DESC, m.arm, s.slice_type, s.ordinal
            LIMIT :limit
            """,
            {**params, "limit": limit},
        )
        return {"rows": rows, "row_count": len(rows)}

    return app


app = create_app()
