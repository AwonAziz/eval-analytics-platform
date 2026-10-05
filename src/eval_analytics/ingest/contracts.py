"""Pydantic contracts for every staging record that reaches a mart.

These are the enforcement point of the ELT. Landing tables hold text so that a
malformed artifact is preserved verbatim; these models decide whether it is
allowed to become a typed mart row. A row that fails validation is recorded in
``ingestion_rejection`` with its machine-readable error codes and never
reaches a fact table.

Bounds are deliberately tight and mostly physical rather than statistical:
confidence must lie in [0, 1], latency cannot be negative, a test set cannot
be empty. That is enough to catch the failure modes that actually occur --
truncated writes, nulls where a number is required, percentages that arrived
as integers out of range, and probability rows that do not sum to one.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Strict(BaseModel):
    """Reject unknown keys so a renamed upstream field cannot pass silently."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


def _parse_ts(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"not an ISO-8601 timestamp: {value!r}") from exc


class TrafficRecord(_Strict):
    """One served request from the real telemetry stream.

    Real telemetry exposes only the top-1 confidence, not a full probability
    vector, so ``prob_true_label`` and ``entropy`` are legitimately absent
    here. The mart stores them as NULL rather than fabricating a distribution.
    """

    request_id: str = Field(min_length=1)
    ts: datetime
    window_id: str | None = None
    run_id: str | None = None
    text_snippet: str | None = None
    gold_intent: str | None = None
    pred_intent: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool | None = None
    in_scope: bool | None = None
    latency_ms: float | None = Field(default=None, ge=0.0)
    text_length_tokens: int | None = Field(default=None, gt=0)

    _ts = field_validator("ts", mode="before")(_parse_ts)


class MetricRecord(_Strict):
    """One named metric observation from a drift/quality window."""

    run_id: str = Field(min_length=1)
    window_id: str = Field(min_length=1)
    window_index: int = Field(ge=0)
    ts: datetime
    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    value: float | None = None
    baseline: float | None = None
    threshold_moderate: float | None = None
    threshold_severe: float | None = None
    severity: str | None = None
    unit: str | None = None

    _ts = field_validator("ts", mode="before")(_parse_ts)


class WindowRecord(_Strict):
    """A labelled monitoring window; becomes a drift-regime slice."""

    window_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    window_index: int = Field(ge=0)
    label: str = Field(min_length=1)
    started_at: datetime
    n_traffic: int | None = Field(default=None, ge=0)
    n_labeled: int | None = Field(default=None, ge=0)
    notes: str | None = None

    _ts = field_validator("started_at", mode="before")(_parse_ts)


class EvalRunRecord(_Strict):
    """Headline quality metrics for one (experiment, model) pair."""

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    #: 'none' covers production models that were not produced by a fine-tune
    #: (the calibrated classifier in the telemetry stream), so that they can
    #: be compared against the LoRA and full-fine-tune arms rather than
    #: excluded from the warehouse.
    arm: str = Field(pattern="^(lora|full_finetune|none)$")
    method: str = Field(min_length=1)
    quantisation: str = Field(min_length=1)
    adapter_rank: int | None = Field(default=None, gt=0)
    n_samples: int = Field(gt=0)
    n_correct: int | None = Field(default=None, ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    accuracy_ci_low: float | None = Field(default=None, ge=0.0, le=1.0)
    accuracy_ci_high: float | None = Field(default=None, ge=0.0, le=1.0)
    macro_f1: float | None = Field(default=None, ge=0.0, le=1.0)
    ece: float | None = Field(default=None, ge=0.0)
    mce: float | None = Field(default=None, ge=0.0)
    brier: float | None = Field(default=None, ge=0.0, le=1.0)
    temperature: float | None = Field(default=None, gt=0.0)
    trainable_params: int | None = Field(default=None, ge=0)
    total_params: int | None = Field(default=None, ge=0)
    seed: int | None = None
    #: Monitoring window this evaluation belongs to. Telemetry emits one
    #: evaluation per window, and the window's regime label becomes a
    #: drift-regime slice on the fact row.
    window_id: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None

    _started = field_validator("started_at", mode="before")(_parse_ts)
    _finished = field_validator("finished_at", mode="before")(_parse_ts)

    @model_validator(mode="after")
    def _check_bounds(self) -> EvalRunRecord:
        if self.n_correct is not None and self.n_correct > self.n_samples:
            raise ValueError(
                f"n_correct ({self.n_correct}) exceeds n_samples ({self.n_samples})"
            )
        lo, hi = self.accuracy_ci_low, self.accuracy_ci_high
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(f"accuracy_ci_low ({lo}) exceeds accuracy_ci_high ({hi})")
        if lo is not None and not (lo - 1e-9 <= self.accuracy <= hi + 1e-9):
            raise ValueError(
                f"accuracy {self.accuracy} falls outside CI [{lo}, {hi}]"
            )
        return self


class PredictionRecord(_Strict):
    """One example's prediction from a fine-tuned arm.

    ``true_label`` is optional because real telemetry contains a large volume
    of unlabeled production traffic. Those rows are worth keeping -- they
    drive confidence and length analysis -- so they land with a NULL label
    rather than being dropped or given an invented one.
    """

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    example_id: str = Field(min_length=1)
    true_label: str | None = None
    predicted_label: str = Field(min_length=1)
    prob_true_label: float | None = Field(default=None, ge=0.0, le=1.0)
    prob_predicted_label: float | None = Field(default=None, ge=0.0, le=1.0)
    max_probability: float = Field(ge=0.0, le=1.0)
    confidence_margin: float | None = Field(default=None, ge=0.0, le=1.0)
    entropy: float | None = Field(default=None, ge=0.0)
    n_tokens: int | None = Field(default=None, gt=0)
    window_id: str | None = None

    @model_validator(mode="after")
    def _check_consistency(self) -> PredictionRecord:
        if self.true_label is None and self.prob_true_label is not None:
            raise ValueError(
                "prob_true_label was supplied without a true_label"
            )
        if self.prob_predicted_label is not None and self.max_probability < self.prob_predicted_label - 1e-9:
            raise ValueError(
                f"max_probability ({self.max_probability}) is below the probability "
                f"assigned to the predicted label ({self.prob_predicted_label})"
            )
        if (
            self.prob_true_label is not None
            and self.prob_predicted_label is not None
            and self.true_label == self.predicted_label
            and abs(self.prob_true_label - self.prob_predicted_label) > 1e-6
        ):
            raise ValueError(
                "true and predicted labels match but their probabilities differ"
            )
        return self


class ServingRequestRecord(_Strict):
    """One timed serving request under a quantisation configuration."""

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    quantisation: str = Field(min_length=1)
    latency_ms: float = Field(ge=0.0)
    batch_size: int | None = Field(default=None, gt=0)
    input_tokens: int | None = Field(default=None, gt=0)
    model_size_bytes: int | None = Field(default=None, gt=0)
    peak_memory_mb: float | None = Field(default=None, ge=0.0)
    throughput_tps: float | None = Field(default=None, ge=0.0)
    request_index: int | None = Field(default=None, ge=0)
    ts: datetime | None = None

    _ts = field_validator("ts", mode="before")(_parse_ts)


class DriftRecord(_Strict):
    """One drift statistic for one feature in one measurement window.

    The grain is ``(feature, window, metric_name)`` rather than a fixed
    PSI/KS/JS triplet, because the sources do not agree on which statistics
    they compute: real telemetry emits PSI per tabular feature plus a family
    of embedding statistics (MMD2, SVD, Frechet), while the fine-tune reports
    emit PSI, KS and JS together. Carrying ``metric_name``/``metric_value``
    alongside the three denormalised columns lets one fact table hold both
    without inventing a KS value for a source that never measured one.
    """

    experiment_id: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    feature: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    metric_value: float
    window_id: str = Field(min_length=1)
    window_index: int = Field(ge=0)
    measured_at: datetime
    psi: float | None = Field(default=None, ge=0.0)
    ks_statistic: float | None = Field(default=None, ge=0.0, le=1.0)
    ks_pvalue: float | None = Field(default=None, ge=0.0, le=1.0)
    js_divergence: float | None = Field(default=None, ge=0.0, le=1.0)
    baseline_value: float | None = None
    current_value: float | None = None
    sample_size: int | None = Field(default=None, ge=0)
    threshold_moderate: float | None = Field(default=None, ge=0.0)
    threshold_severe: float | None = Field(default=None, ge=0.0)
    severity: str | None = None

    _ts = field_validator("measured_at", mode="before")(_parse_ts)

    @model_validator(mode="after")
    def _check_metric_agrees_with_denormalised_column(self) -> DriftRecord:
        expected = {
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "js_divergence": self.js_divergence,
        }.get(self.metric_name)
        if expected is not None and abs(expected - self.metric_value) > 1e-9:
            raise ValueError(
                f"metric_value ({self.metric_value}) disagrees with the "
                f"{self.metric_name} column ({expected})"
            )
        return self


class ModelCardRecord(_Strict):
    """Model identity and parameter budget."""

    model_id: str = Field(min_length=1)
    base_model: str = Field(min_length=1)
    arm: str = Field(pattern="^(lora|full_finetune)$")
    method: str = Field(min_length=1)
    adapter_rank: int | None = Field(default=None, gt=0)
    trainable_params: int = Field(ge=0)
    total_params: int = Field(ge=0)
    trainable_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    quantisation: str = Field(min_length=1)
    dataset: str | None = None
    created_at: datetime | None = None
    is_champion: bool | None = None

    _created = field_validator("created_at", mode="before")(_parse_ts)

    @model_validator(mode="after")
    def _check_params(self) -> ModelCardRecord:
        if self.trainable_params > self.total_params:
            raise ValueError(
                f"trainable_params ({self.trainable_params}) exceeds "
                f"total_params ({self.total_params})"
            )
        return self


CONTRACTS: dict[str, type[_Strict]] = {
    "traffic": TrafficRecord,
    "metric": MetricRecord,
    "window": WindowRecord,
    "eval_run": EvalRunRecord,
    "prediction": PredictionRecord,
    "serving": ServingRequestRecord,
    "drift": DriftRecord,
    "model_card": ModelCardRecord,
}
