"""Validation with counted, attributable rejections.

Every staging record is checked against its Pydantic contract. Accepted rows
are returned for the mart load; rejected rows are returned with their error
codes so the pipeline can write them to ``ingestion_rejection``. Nothing is
dropped silently -- a row either becomes a typed mart row or becomes an
audited rejection.

A rejected row costs one landing-table row and never reaches a fact table, so
the accepted/rejected counts are what CI asserts on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .contracts import CONTRACTS

TRecord = TypeVar("TRecord", bound=BaseModel)


@dataclass
class RejectedRow:
    """One source row that failed its contract."""

    dataset_name: str
    source_name: str
    source_row_id: str | None
    error_count: int
    error_codes: list[str]
    error_messages: list[str]
    raw_payload: str


@dataclass
class ValidationResult:
    """Outcome of validating one dataset."""

    dataset_name: str
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[RejectedRow] = field(default_factory=list)

    @property
    def rows_in(self) -> int:
        return len(self.accepted) + len(self.rejected)

    @property
    def rows_rejected(self) -> int:
        return len(self.rejected)

    @property
    def rejection_rate(self) -> float:
        return self.rows_rejected / self.rows_in if self.rows_in else 0.0

    def error_code_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rejected:
            for code in row.error_codes:
                counts[code] = counts.get(code, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def validate_rows(
    dataset_name: str,
    rows: list[dict[str, Any]],
    source_name: str,
) -> ValidationResult:
    """Validate every row against ``CONTRACTS[dataset_name]``.

    Unknown dataset names are a programming error and raise immediately --
    silently landing unvalidated rows is exactly the failure this layer
    exists to prevent.
    """
    try:
        contract = CONTRACTS[dataset_name]
    except KeyError as exc:
        raise KeyError(
            f"no contract registered for dataset {dataset_name!r}; "
            f"known datasets: {sorted(CONTRACTS)}"
        ) from exc

    result = ValidationResult(dataset_name=dataset_name)
    for index, row in enumerate(rows):
        source_row_id = str(
            row.get("request_id")
            or row.get("window_id")
            or row.get("example_id")
            or row.get("metric_name")
            or index
        )
        try:
            model = contract.model_validate(row)
        except ValidationError as exc:
            errors = exc.errors()
            result.rejected.append(
                RejectedRow(
                    dataset_name=dataset_name,
                    source_name=source_name,
                    source_row_id=source_row_id,
                    error_count=len(errors),
                    error_codes=sorted({e["type"] for e in errors}),
                    error_messages=[_format_error(e) for e in errors],
                    raw_payload=_safe_json(row),
                )
            )
            continue
        except (TypeError, ValueError) as exc:
            # Non-ValidationError failures (e.g. unhashable inputs) are still
            # a rejection of this row, not a crash of the pipeline.
            result.rejected.append(
                RejectedRow(
                    dataset_name=dataset_name,
                    source_name=source_name,
                    source_row_id=source_row_id,
                    error_count=1,
                    error_codes=["unprocessable_row"],
                    error_messages=[f"{type(exc).__name__}: {exc}"],
                    raw_payload=_safe_json(row),
                )
            )
            continue

        result.accepted.append(model.model_dump(mode="json"))
    return result


def _format_error(error: dict[str, Any]) -> str:
    location = ".".join(str(part) for part in error["loc"]) or "<root>"
    return f"{location}: {error['msg']}"


def _safe_json(row: Any) -> str:
    try:
        return json.dumps(row, default=str, sort_keys=True)[:4000]
    except (TypeError, ValueError):
        return repr(row)[:4000]
