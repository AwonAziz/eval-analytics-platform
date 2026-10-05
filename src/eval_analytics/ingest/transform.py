"""Transform: validated staging records -> typed, conformed mart rows.

Two responsibilities:

1. **Conformance.** Every fact key must resolve to a dimension member. Slices
   are *derived* here (overall / per-class / per-length-bucket / per-regime)
   rather than read from a column, so the same document can appear under
   several slices and every slice in ``dim_slice`` is guaranteed to be
   referenced by at least one fact row.
2. **Typing.** Staging carries validated Python values; this layer renders
   them into the SQL argument types the marts expect and stamps the
   deterministic surrogate keys from :mod:`..warehouse.keys`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from ..config import LENGTH_BUCKET_EDGES, length_bucket_id
from ..warehouse.keys import stable_key

SLICE_OVERALL = "overall"


@dataclass
class DimensionRows:
    """Accumulated dimension members, keyed for upsert."""

    models: dict[str, dict[str, Any]] = field(default_factory=dict)
    experiments: dict[str, dict[str, Any]] = field(default_factory=dict)
    features: dict[str, dict[str, Any]] = field(default_factory=dict)
    slices: dict[str, dict[str, Any]] = field(default_factory=dict)
    dates: dict[date, dict[str, Any]] = field(default_factory=dict)


@dataclass
class MartRows:
    """Typed fact rows, keyed by mart name."""

    evaluation_run: list[dict[str, Any]] = field(default_factory=list)
    prediction: list[dict[str, Any]] = field(default_factory=list)
    serving_request: list[dict[str, Any]] = field(default_factory=list)
    drift_measurement: list[dict[str, Any]] = field(default_factory=list)


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None:
        raise ValueError("a fact row reached the transform without a timestamp")
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).date()


def _iso_week(d: date) -> int:
    return d.isocalendar().week


def register_date(dims: DimensionRows, value: Any) -> date:
    d = _as_date(value)
    if d not in dims.dates:
        dims.dates[d] = {
            "date_id": d,
            "year": d.year,
            "quarter": (d.month - 1) // 3 + 1,
            "month": d.month,
            "week": _iso_week(d),
            "day_of_week": d.weekday() + 1,
            "is_weekend": d.weekday() >= 5,
        }
    return d


def register_slice(
    dims: DimensionRows,
    slice_id: str,
    slice_type: str,
    slice_name: str,
    definition: str,
    source_system: str,
    ordinal: int | None = None,
) -> str:
    dims.slices.setdefault(
        slice_id,
        {
            "slice_id": slice_id,
            "slice_type": slice_type,
            "slice_name": slice_name,
            "ordinal": ordinal,
            "definition": definition,
            "source_system": source_system,
        },
    )
    return slice_id


def register_length_buckets(dims: DimensionRows, source_system: str) -> None:
    """Materialise every configured length bucket as a slice member."""
    for ordinal, (low, high) in enumerate(LENGTH_BUCKET_EDGES):
        slice_id = length_bucket_id(low)
        if slice_id is None:
            continue
        name = f"{low}-{high}" if high is not None else f"{low}+"
        register_slice(
            dims,
            slice_id,
            "length_bucket",
            name,
            f"documents with {name} whitespace-adjusted tokens",
            source_system,
            ordinal=ordinal,
        )


def build_dimensions(
    model_cards: list[dict[str, Any]],
    eval_runs: list[dict[str, Any]],
    features: list[str],
    window_labels: dict[str, str],
    classes: list[str],
) -> DimensionRows:
    """Assemble all dimension members from the validated inputs."""
    dims = DimensionRows()

    for card in model_cards:
        dims.models[card["model_id"]] = {
            "model_id": card["model_id"],
            "base_model": card["base_model"],
            "arm": card["arm"],
            "method": card["method"],
            "quantisation": card["quantisation"],
            "adapter_rank": card.get("adapter_rank"),
            "trainable_params": card.get("trainable_params"),
            "total_params": card.get("total_params"),
            "trainable_pct": card.get("trainable_pct"),
            "source_dataset": card.get("dataset"),
            "created_at": _as_datetime(card.get("created_at")),
            "is_champion": bool(card.get("is_champion")),
            "data_origin": "synthetic",
        }

    # Telemetry models are not fine-tunes, so they arrive via eval_run rows
    # rather than model cards.
    for run in eval_runs:
        model_id = run["model_id"]
        if model_id in dims.models:
            continue
        dims.models[model_id] = {
            "model_id": model_id,
            "base_model": run["base_model"],
            "arm": run["arm"],
            "method": run["method"],
            "quantisation": run["quantisation"],
            "adapter_rank": None,
            "trainable_params": None,
            "total_params": None,
            "trainable_pct": None,
            "source_dataset": "banking77",
            "created_at": None,
            "is_champion": True,
            "data_origin": "real",
        }

    for run in eval_runs:
        eid = run["experiment_id"]
        if eid in dims.experiments:
            continue
        dims.experiments[eid] = {
            "experiment_id": eid,
            "name": eid,
            "hypothesis": _HYPOTHESES.get(run["arm"]),
            "owner": "eval-analytics-platform",
            "started_at": _as_datetime(run.get("started_at")),
            "ended_at": _as_datetime(run.get("finished_at")),
            "status": "complete",
            "data_source": "banking77" if run["arm"] == "none" else "docclass-8k-v3",
            "tags": run["arm"],
            "data_origin": "real" if run["arm"] == "none" else "synthetic",
        }

    for name in sorted(set(features)):
        feature_type = "categorical" if "rate" in name or "ratio" in name else "numeric"
        if name.startswith("embedding::"):
            feature_type = "numeric"
            source_system = "telemetry_db"
        else:
            source_system = "finetune_artifacts"
        dims.features[name] = {
            "feature_id": name,
            "feature_type": feature_type,
            "source_system": source_system,
            "is_drift_tracked": True,
            "description": f"drift-tracked feature {name}",
        }

    register_slice(
        dims, SLICE_OVERALL, "overall", "All examples", "entire evaluation set", "derived"
    )
    register_length_buckets(dims, "derived")

    for ordinal, label in enumerate(sorted(set(window_labels.values()))):
        register_slice(
            dims,
            f"regime_{label}",
            "drift_window",
            label,
            f"monitoring window labelled {label!r} by the upstream drift harness",
            "telemetry_db",
            ordinal=ordinal,
        )

    for ordinal, label in enumerate(classes):
        register_slice(
            dims,
            f"class_{label}",
            "class",
            label,
            f"per-class slice for label {label!r}",
            "derived",
            ordinal=ordinal,
        )
    return dims


_HYPOTHESES = {
    "lora": "A rank-16 LoRA adapter matches full fine-tuning accuracy at ~1.3% of trainable parameters, with better calibration and better tail-class recall.",
    "full_finetune": "Full fine-tuning maximises headline accuracy but overfits the majority prior and degrades calibration.",
    "none": "Production classifier quality tracked per monitoring window against a frozen reference snapshot.",
}


def _as_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _window_key(window_id: Any) -> str:
    """Normalise a window id into the key component of the fact grain."""
    return str(window_id) if window_id else ""


def overall_index(marts: MartRows) -> dict[tuple[str, str, str], tuple[date, int]]:
    """Map (experiment, model, window) -> (date, evaluation_run_key).

    Every other builder joins through this so that predictions and derived
    per-class / per-length slices attach to the same overall evaluation row
    their predictions came from.
    """
    index: dict[tuple[str, str, str], tuple[date, int]] = {}
    for run in marts.evaluation_run:
        if run["slice_id"] != SLICE_OVERALL:
            continue
        key = (run["experiment_id"], run["model_id"], _window_key(run.get("window_id")))
        index[key] = (run["date_id"], run["evaluation_run_key"])
    return index


def build_evaluation_runs(
    marts: MartRows,
    dims: DimensionRows,
    eval_runs: list[dict[str, Any]],
    window_labels: dict[str, str],
    origins: dict[tuple[str, str], str],
) -> None:
    """One fact row per (experiment, model, slice, date, window).

    Each run contributes an 'overall' row plus a drift-regime row when the
    run belongs to a labelled monitoring window. Class- and length-level rows
    are derived separately from ``fact_prediction`` so that the mart cannot
    disagree with the predictions it summarises.
    """
    for run in eval_runs:
        experiment_id = run["experiment_id"]
        model_id = run["model_id"]
        origin = origins.get((experiment_id, model_id), "synthetic")
        date_id = register_date(dims, run.get("started_at") or run.get("finished_at"))
        window_id = run.get("window_id")

        slices = [SLICE_OVERALL]
        label = window_labels.get(window_id or "")
        if label:
            slices.append(f"regime_{label}")

        for slice_id in slices:
            key = stable_key(experiment_id, model_id, slice_id, date_id, _window_key(window_id))
            marts.evaluation_run.append(
                {
                    "evaluation_run_key": key,
                    "experiment_id": experiment_id,
                    "model_id": model_id,
                    "slice_id": slice_id,
                    "date_id": date_id,
                    "window_id": window_id,
                    "n_samples": int(run["n_samples"]),
                    "n_correct": _int_or_none(run.get("n_correct")),
                    "accuracy": float(run["accuracy"]),
                    "accuracy_ci_low": _float_or_none(run.get("accuracy_ci_low")),
                    "accuracy_ci_high": _float_or_none(run.get("accuracy_ci_high")),
                    "macro_f1": _float_or_none(run.get("macro_f1")),
                    "ece": _float_or_none(run.get("ece")),
                    "mce": _float_or_none(run.get("mce")),
                    "brier": _float_or_none(run.get("brier")),
                    "trainable_params": _int_or_none(run.get("trainable_params")),
                    "seed": _int_or_none(run.get("seed")),
                    "evaluation_seconds": None,
                    "source_system": (
                        "finetune_artifacts" if origin == "synthetic" else "telemetry_db"
                    ),
                    "data_origin": origin,
                }
            )


def build_class_evaluation_runs(
    marts: MartRows,
    dims: DimensionRows,
    predictions: list[dict[str, Any]],
    origin_index: dict[tuple[str, str, str], tuple[date, int]],
    origins: dict[tuple[str, str], str],
) -> None:
    """Derive one accuracy row per (run, window, true label) from predictions.

    These rows are *computed* rather than copied from a source report, so
    ``fact_evaluation_run`` at the class grain can never disagree with
    ``fact_prediction``. That is what makes the per-class arm comparison --
    the query that answers "which slices does LoRA still lose on" -- sound.
    """
    buckets: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for pred in predictions:
        true_label = pred.get("true_label")
        if not true_label:
            continue
        key = (
            pred["experiment_id"],
            pred["model_id"],
            _window_key(pred.get("window_id")),
            true_label,
        )
        entry = buckets.setdefault(key, {"n": 0, "n_correct": 0})
        entry["n"] += 1
        entry["n_correct"] += int(true_label == pred["predicted_label"])

    for (experiment_id, model_id, wkey, label), stats in sorted(buckets.items()):
        located = origin_index.get((experiment_id, model_id, wkey))
        if located is None or stats["n"] == 0:
            continue
        date_id, _ = located
        window_id = wkey or None
        slice_id = register_slice(
            dims,
            f"class_{label}",
            "class",
            label,
            f"per-class slice for label {label!r}",
            "derived",
        )
        marts.evaluation_run.append(
            {
                "evaluation_run_key": stable_key(
                    experiment_id, model_id, slice_id, date_id, wkey
                ),
                "experiment_id": experiment_id,
                "model_id": model_id,
                "slice_id": slice_id,
                "date_id": date_id,
                "window_id": window_id,
                "n_samples": stats["n"],
                "n_correct": stats["n_correct"],
                "accuracy": stats["n_correct"] / stats["n"],
                "accuracy_ci_low": None,
                "accuracy_ci_high": None,
                "macro_f1": None,
                "ece": None,
                "mce": None,
                "brier": None,
                "trainable_params": None,
                "seed": None,
                "evaluation_seconds": None,
                "source_system": _source_system(experiment_id, model_id, origins),
                "data_origin": _origin(experiment_id, model_id, origins),
            }
        )


def _origin(experiment_id: str, model_id: str, origins: dict[tuple[str, str], str]) -> str:
    return origins.get((experiment_id, model_id), "synthetic")


def _source_system(experiment_id: str, model_id: str, origins: dict[tuple[str, str], str]) -> str:
    return "finetune_artifacts" if _origin(experiment_id, model_id, origins) == "synthetic" else "telemetry_db"


def build_predictions(
    marts: MartRows,
    dims: DimensionRows,
    predictions: list[dict[str, Any]],
    origin_index: dict[tuple[str, str, str], tuple[date, int]],
    window_labels: dict[str, str],
    origins: dict[tuple[str, str], str],
) -> None:
    """One fact row per (evaluation run, example)."""
    for pred in predictions:
        experiment_id = pred["experiment_id"]
        model_id = pred["model_id"]
        located = origin_index.get(
            (experiment_id, model_id, _window_key(pred.get("window_id")))
        )
        if located is None:
            continue
        date_id, run_key = located

        bucket = length_bucket_id(pred.get("n_tokens"))
        regime = None
        label = window_labels.get(pred.get("window_id") or "")
        if label:
            regime = register_slice(
                dims,
                f"regime_{label}",
                "drift_window",
                label,
                f"monitoring window labelled {label!r} by the upstream drift harness",
                "telemetry_db",
            )
        true_label = pred.get("true_label")
        predicted = pred["predicted_label"]
        is_correct = int(true_label == predicted) if true_label is not None else None

        marts.prediction.append(
            {
                "prediction_key": stable_key(run_key, pred["example_id"]),
                "evaluation_run_key": run_key,
                "experiment_id": experiment_id,
                "model_id": model_id,
                "date_id": date_id,
                "example_id": pred["example_id"],
                "window_id": pred.get("window_id"),
                "true_label": true_label,
                "predicted_label": predicted,
                "prob_true_label": _float_or_none(pred.get("prob_true_label")),
                "prob_predicted_label": _float_or_none(pred.get("prob_predicted_label")),
                "max_probability": _float_or_none(pred.get("max_probability")),
                "confidence_margin": _float_or_none(pred.get("confidence_margin")),
                "entropy": _float_or_none(pred.get("entropy")),
                "is_correct": is_correct,
                "is_abstained": pred.get("abstained"),
                "n_tokens": _int_or_none(pred.get("n_tokens")),
                "doc_length_bucket_id": bucket,
                "regime_slice_id": regime,
                "data_origin": _origin(experiment_id, model_id, origins),
            }
        )


def build_length_evaluation_runs(
    marts: MartRows,
    predictions: list[dict[str, Any]],
    origin_index: dict[tuple[str, str, str], tuple[date, int]],
    origins: dict[tuple[str, str], str],
) -> None:
    """Derive accuracy per document-length bucket from the predictions.

    Kept as fact rows rather than left to an ad-hoc GROUP BY so that the
    length analysis sits in the same star schema as every other slice and can
    be joined against model metadata in the API.
    """
    buckets: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for pred in predictions:
        if not pred.get("true_label"):
            continue
        bucket = length_bucket_id(pred.get("n_tokens"))
        if bucket is None:
            continue
        key = (
            pred["experiment_id"],
            pred["model_id"],
            _window_key(pred.get("window_id")),
            bucket,
        )
        entry = buckets.setdefault(key, {"n": 0, "n_correct": 0})
        entry["n"] += 1
        entry["n_correct"] += int(pred["true_label"] == pred["predicted_label"])

    for (experiment_id, model_id, wkey, bucket), stats in sorted(buckets.items()):
        located = origin_index.get((experiment_id, model_id, wkey))
        if located is None or stats["n"] == 0:
            continue
        date_id, _ = located
        marts.evaluation_run.append(
            {
                "evaluation_run_key": stable_key(
                    experiment_id, model_id, bucket, date_id, wkey
                ),
                "experiment_id": experiment_id,
                "model_id": model_id,
                "slice_id": bucket,
                "date_id": date_id,
                "window_id": wkey or None,
                "n_samples": stats["n"],
                "n_correct": stats["n_correct"],
                "accuracy": stats["n_correct"] / stats["n"],
                "accuracy_ci_low": None,
                "accuracy_ci_high": None,
                "macro_f1": None,
                "ece": None,
                "mce": None,
                "brier": None,
                "trainable_params": None,
                "seed": None,
                "evaluation_seconds": None,
                "source_system": _source_system(experiment_id, model_id, origins),
                "data_origin": _origin(experiment_id, model_id, origins),
            }
        )


def build_serving_requests(
    marts: MartRows,
    dims: DimensionRows,
    serving: list[dict[str, Any]],
    origin_index: dict[tuple[str, str, str], tuple[date, int]],
    origins: dict[tuple[str, str], str],
) -> None:
    for req in serving:
        experiment_id = req["experiment_id"]
        model_id = req["model_id"]
        located = origin_index.get((experiment_id, model_id, ""))
        if located is not None:
            date_id = located[0]
        else:
            # A serving row whose model has no evaluation run still belongs to
            # a date; use the request's own timestamp rather than wall-clock
            # now(), so a rebuild from fixed inputs stays reproducible.
            date_id = register_date(
                dims, req.get("ts") or datetime.now(timezone.utc)
            )

        marts.serving_request.append(
            {
                "serving_request_key": stable_key(
                    model_id,
                    req["quantisation"],
                    req.get("batch_size"),
                    req.get("input_tokens"),
                    req.get("request_index"),
                    experiment_id,
                ),
                "model_id": model_id,
                "date_id": date_id,
                "slice_id": None,
                "quantisation": req["quantisation"],
                "batch_size": _int_or_none(req.get("batch_size")),
                "input_tokens": _int_or_none(req.get("input_tokens")),
                "latency_ms": float(req["latency_ms"]),
                "model_size_bytes": _int_or_none(req.get("model_size_bytes")),
                "peak_memory_mb": _float_or_none(req.get("peak_memory_mb")),
                "throughput_tps": _float_or_none(req.get("throughput_tps")),
                "request_index": _int_or_none(req.get("request_index")),
                "data_origin": origins.get((experiment_id, model_id), "synthetic"),
            }
        )


def build_drift_measurements(
    marts: MartRows,
    dims: DimensionRows,
    drift: list[dict[str, Any]],
    window_labels: dict[str, str],
    origins: dict[tuple[str, str], str],
) -> None:
    for d in drift:
        experiment_id = d["experiment_id"]
        model_id = d["model_id"]
        date_id = register_date(dims, d["measured_at"])
        label = window_labels.get(d["window_id"])
        slice_id = f"regime_{label}" if label else None
        if slice_id:
            register_slice(
                dims,
                slice_id,
                "drift_window",
                label,
                f"monitoring window labelled {label!r}",
                "telemetry_db",
            )

        marts.drift_measurement.append(
            {
                "drift_measurement_key": stable_key(
                    d["feature"], d["window_id"], d["metric_name"]
                ),
                "feature_id": d["feature"],
                "model_id": model_id,
                "date_id": date_id,
                "slice_id": slice_id,
                "window_id": d["window_id"],
                "measurement_ts": _as_datetime(d["measured_at"]),
                "metric_name": d["metric_name"],
                "metric_value": float(d["metric_value"]),
                "psi": _float_or_none(d.get("psi")),
                "ks_statistic": _float_or_none(d.get("ks_statistic")),
                "ks_pvalue": _float_or_none(d.get("ks_pvalue")),
                "js_divergence": _float_or_none(d.get("js_divergence")),
                "baseline_value": _float_or_none(d.get("baseline_value")),
                "current_value": _float_or_none(d.get("current_value")),
                "sample_size": _int_or_none(d.get("sample_size")),
                "threshold_moderate": _float_or_none(d.get("threshold_moderate")),
                "threshold_severe": _float_or_none(d.get("threshold_severe")),
                "severity": d.get("severity"),
                "data_origin": origins.get((experiment_id, model_id), "synthetic"),
            }
        )


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)
