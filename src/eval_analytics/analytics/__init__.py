"""Analytical queries answering real questions about model quality."""

from .stats import (
    CalibrationBucket,
    CalibrationSummary,
    PairedResult,
    calibration,
    mcnemar_exact,
    paired_bootstrap,
)

__all__ = [
    "CalibrationBucket",
    "CalibrationSummary",
    "PairedResult",
    "calibration",
    "mcnemar_exact",
    "paired_bootstrap",
]
