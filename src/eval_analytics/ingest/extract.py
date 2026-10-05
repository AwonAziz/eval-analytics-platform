"""Extraction: raw artifacts on disk -> staging records.

Two sources feed the warehouse:

* ``data/raw/generated`` -- the fine-tune artifact sets (``test_preds.npy``,
  ``eval_report.json``, ``serving_report.json``, ``drift_report.json``).
  Produced by :mod:`eval_analytics.ingest.artifacts` today, and by whatever
  real training run produces them tomorrow. Nothing downstream distinguishes
  the two.
* the upstream telemetry SQLite database -- the only source of genuinely
  measured rows in this warehouse.

Extraction is deliberately dumb: it reshapes bytes into dictionaries and
computes a few derived quantities (token counts from text, entropy from a
probability vector). It does not validate; that is :mod:`.validate`'s job.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

#: Words -> approximate subword tokens. Banking77-style utterances are short,
#: so a whitespace count with a 1.3x subword multiplier tracks the real
#: tokenizer closely enough for length bucketing.
WORDS_TO_TOKENS = 1.3

SOURCE_GENERATED = "finetune_artifacts"
SOURCE_TELEMETRY = "telemetry_db"


def estimate_tokens(text: str | None) -> int | None:
    """Approximate token count for a piece of text."""
    if not text or not text.strip():
        return None
    return max(1, int(round(len(text.split()) * WORDS_TO_TOKENS)))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _entropy(probs: np.ndarray) -> float:
    """Shannon entropy in nats, with 0*log(0) treated as 0."""
    safe = np.clip(probs, 1e-12, 1.0)
    return float(-(safe * np.log(safe)).sum())


# ---------------------------------------------------------------------------
# Fine-tune artifacts
# ---------------------------------------------------------------------------


def extract_generated(root: Path) -> dict[str, list[dict[str, Any]]]:
    """Read every artifact set under ``root``."""
    out: dict[str, list[dict[str, Any]]] = {
        "model_card": [],
        "eval_run": [],
        "prediction": [],
        "serving": [],
        "drift": [],
    }
    if not root.exists():
        return out

    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if not (run_dir / "eval_report.json").exists():
            continue
        report = _read_json(run_dir / "eval_report.json")
        experiment_id = report["experiment_id"]
        model_id = report["model_id"]

        card_path = run_dir / "model_card.json"
        card = _read_json(card_path) if card_path.exists() else {}
        quantisation = card.get("quantisation", report.get("quantisation", "fp32"))
        out["model_card"].append(
            {
                "model_id": card.get("model_id", model_id),
                "base_model": card.get("base_model", report["base_model"]),
                "arm": card.get("arm", report["arm"]),
                "method": card.get("method", report["arm"]),
                "adapter_rank": card.get("adapter_rank", report.get("adapter_rank")),
                "trainable_params": int(
                    card.get("trainable_params", report.get("trainable_params", 0))
                ),
                "total_params": int(card.get("total_params", report.get("total_params", 0))),
                "trainable_pct": card.get("trainable_pct"),
                "quantisation": quantisation,
                "dataset": card.get("dataset", report.get("dataset")),
                "created_at": card.get("created_at"),
                "is_champion": card.get("is_champion", False),
            }
        )

        out["eval_run"].append(
            {
                "experiment_id": experiment_id,
                "model_id": model_id,
                "base_model": report["base_model"],
                "arm": report["arm"],
                "method": report["arm"],
                "quantisation": quantisation,
                "adapter_rank": report.get("adapter_rank"),
                "n_samples": int(report["test_set_size"]),
                "n_correct": int(report["n_correct"]),
                "accuracy": report["accuracy"],
                "accuracy_ci_low": report.get("accuracy_ci_low"),
                "accuracy_ci_high": report.get("accuracy_ci_high"),
                "macro_f1": report.get("macro_f1"),
                "ece": report.get("ece"),
                "mce": report.get("mce"),
                "brier": report.get("brier"),
                "temperature": report.get("temperature"),
                "trainable_params": report.get("trainable_params"),
                "total_params": report.get("total_params"),
                "seed": report.get("seed"),
                "window_id": None,
                "started_at": report.get("started_at"),
                "finished_at": report.get("finished_at"),
            }
        )

        out["prediction"].extend(_extract_predictions(run_dir, experiment_id, model_id))

        serving_path = run_dir / "serving_report.json"
        if serving_path.exists():
            out["serving"].extend(_extract_serving(_read_json(serving_path), experiment_id, model_id))

        drift_path = run_dir / "drift_report.json"
        if drift_path.exists():
            drift = _read_json(drift_path)
            for m in drift["measurements"]:
                out["drift"].append(
                    {
                        "experiment_id": experiment_id,
                        "model_id": model_id,
                        "feature": m["feature"],
                        "metric_name": m["metric_name"],
                        "metric_value": m["metric_value"],
                        "window_id": m.get("window_id", f"{experiment_id}-w{m['window_index']:02d}"),
                        "window_index": int(m["window_index"]),
                        "measured_at": m["measured_at"],
                        "psi": m.get("psi"),
                        "ks_statistic": m.get("ks_statistic"),
                        "ks_pvalue": m.get("ks_pvalue"),
                        "js_divergence": m.get("js_divergence"),
                        "baseline_value": m.get("baseline_value"),
                        "current_value": m.get("current_value"),
                        "sample_size": m.get("sample_size"),
                        "threshold_moderate": m.get("threshold_moderate"),
                        "threshold_severe": m.get("threshold_severe"),
                        "severity": m.get("severity"),
                    }
                )
    return out


def _extract_predictions(
    run_dir: Path, experiment_id: str, model_id: str
) -> list[dict[str, Any]]:
    """Expand ``test_preds.npy`` into one record per example."""
    probs = np.load(run_dir / "test_preds.npy").astype(np.float64)
    true_labels = np.load(run_dir / "test_labels.npy", allow_pickle=True)
    example_ids = np.load(run_dir / "test_example_ids.npy", allow_pickle=True)
    lengths = np.load(run_dir / "test_token_lengths.npy")
    label_names = _read_json(run_dir / "classes.json")["label_names"]

    index_of = {name: i for i, name in enumerate(label_names)}
    rows: list[dict[str, Any]] = []

    # strict=True: a length mismatch between the probability matrix and the
    # label array would silently truncate the example set and quietly
    # corrupt every downstream paired statistic.
    for i, (probs_i, true_label) in enumerate(zip(probs, true_labels, strict=True)):
        order = np.argsort(probs_i)[::-1]
        pred_idx = int(order[0])
        pred_label = label_names[pred_idx]
        true_idx = index_of.get(str(true_label))
        top1, top2 = float(probs_i[order[0]]), float(probs_i[order[1]])
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_id": model_id,
                "example_id": str(example_ids[i]),
                "true_label": str(true_label),
                "predicted_label": pred_label,
                "prob_true_label": float(probs_i[true_idx]) if true_idx is not None else None,
                "prob_predicted_label": top1,
                "max_probability": top1,
                "confidence_margin": top1 - top2,
                "entropy": _entropy(probs_i),
                "n_tokens": int(lengths[i]),
                "window_id": None,
            }
        )
    return rows


def _extract_serving(
    report: dict[str, Any], experiment_id: str, model_id: str
) -> list[dict[str, Any]]:
    """Join per-request timings to the size/memory of their config."""
    by_quant = {c["quantisation"]: c for c in report.get("configs", [])}
    rows = []
    for req in report.get("requests", []):
        cfg = by_quant.get(req["quantisation"], {})
        rows.append(
            {
                "experiment_id": experiment_id,
                "model_id": model_id,
                "quantisation": req["quantisation"],
                "latency_ms": req["latency_ms"],
                "batch_size": req.get("batch_size"),
                "input_tokens": req.get("input_tokens"),
                "model_size_bytes": cfg.get("model_size_bytes"),
                "peak_memory_mb": cfg.get("peak_memory_mb"),
                "throughput_tps": None,
                "request_index": req.get("request_index"),
                "ts": report.get("generated_at"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Upstream telemetry (real, measured)
# ---------------------------------------------------------------------------

#: Telemetry drift metrics that are not per-feature PSI. They are still drift
#: measurements, so they land in the same fact with their own metric_name.
_EMBEDDING_METRICS = {
    "centroid_cosine_shift",
    "concept_gap",
    "domain_classifier_auc",
    "frechet_normalised",
    "mmd2",
    "ood_rate",
    "swd",
}


def extract_telemetry(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Read the upstream telemetry database."""
    out: dict[str, list[dict[str, Any]]] = {
        "window": [],
        "metric": [],
        "traffic": [],
        "eval_run": [],
        "prediction": [],
        "serving": [],
        "drift": [],
    }
    if not db_path.exists():
        return out

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        runs = {r["run_id"]: dict(r) for r in conn.execute("SELECT * FROM runs")}
        windows = [dict(w) for w in conn.execute("SELECT * FROM windows")]
        metrics = [dict(m) for m in conn.execute("SELECT * FROM metrics ORDER BY id")]

        def model_for(run_id: str) -> str:
            app_model = runs.get(run_id, {})
            raw = (app_model or {}).get("app_model") or f"unknown-{run_id}"
            return raw.split(":")[0]

        def encoder_for(run_id: str) -> str:
            return ((runs.get(run_id) or {}).get("encoder") or "unknown").split("@")[0]

        for w in windows:
            out["window"].append(
                {
                    "window_id": w["window_id"],
                    "run_id": w["run_id"],
                    "window_index": int(w["window_index"]),
                    "label": w["label"] or "unlabelled",
                    "started_at": w["started_at"],
                    "n_traffic": w["n_traffic"],
                    "n_labeled": w["n_labeled"],
                    "notes": w["notes"],
                }
            )

        window_meta = {
            w["window_id"]: {
                "index": int(w["window_index"]),
                "run_id": w["run_id"],
                "n_labeled": w["n_labeled"] or 0,
                "label": w["label"] or "unlabelled",
                "started_at": w["started_at"],
            }
            for w in windows
        }

        for m in metrics:
            out["metric"].append(
                {
                    "run_id": m["run_id"],
                    "window_id": m["window_id"],
                    "window_index": int(m["window_index"]),
                    "ts": m["ts"],
                    "category": m["category"],
                    "name": m["name"],
                    "value": m["value"],
                    "baseline": m["baseline"],
                    "threshold_moderate": m["threshold_moderate"],
                    "threshold_severe": m["threshold_severe"],
                    "severity": m["severity"],
                    "unit": m["unit"],
                }
            )
            if m["value"] is None:
                continue

            name = m["name"]
            meta = window_meta.get(m["window_id"], {})
            common = {
                "experiment_id": m["run_id"],
                "model_id": model_for(m["run_id"]),
                "window_id": m["window_id"],
                "window_index": int(m["window_index"]),
                "measured_at": m["ts"],
                "baseline_value": m["baseline"],
                "current_value": m["value"],
                "threshold_moderate": m["threshold_moderate"],
                "threshold_severe": m["threshold_severe"],
                "severity": m["severity"],
                "sample_size": meta.get("n_labeled"),
            }

            if name.startswith("psi::"):
                out["drift"].append(
                    {
                        **common,
                        "feature": name.split("::", 1)[1],
                        "metric_name": "psi",
                        "metric_value": m["value"],
                        "psi": m["value"],
                        "ks_statistic": None,
                        "ks_pvalue": None,
                        "js_divergence": None,
                    }
                )
            elif name in _EMBEDDING_METRICS:
                out["drift"].append(
                    {
                        **common,
                        "feature": f"embedding::{name}",
                        "metric_name": name,
                        "metric_value": m["value"],
                        "psi": None,
                        "ks_statistic": None,
                        "ks_pvalue": None,
                        "js_divergence": None,
                    }
                )

        # Per-window quality becomes one fact_evaluation_run row per window.
        quality_names = {"accuracy", "macro_f1", "ece", "brier"}
        by_window: dict[str, dict[str, float]] = {}
        for m in metrics:
            if m["category"] == "quality" and m["name"] in quality_names and m["value"] is not None:
                by_window.setdefault(m["window_id"], {})[m["name"]] = m["value"]

        for window_id, values in by_window.items():
            meta = window_meta.get(window_id, {})
            n_samples = meta.get("n_labeled") or 0
            if n_samples <= 0:
                continue
            accuracy = values.get("accuracy")
            out["eval_run"].append(
                {
                    "experiment_id": meta.get("run_id"),
                    "model_id": model_for(meta.get("run_id")),
                    "base_model": encoder_for(meta.get("run_id")),
                    "arm": "none",
                    "method": "calibrated_logreg",
                    "quantisation": "fp32",
                    "adapter_rank": None,
                    "n_samples": int(n_samples),
                    "n_correct": int(round(accuracy * n_samples)) if accuracy is not None else None,
                    "accuracy": accuracy if accuracy is not None else 0.0,
                    "accuracy_ci_low": None,
                    "accuracy_ci_high": None,
                    "macro_f1": values.get("macro_f1"),
                    "ece": values.get("ece"),
                    "mce": None,
                    "brier": values.get("brier"),
                    "temperature": None,
                    "trainable_params": None,
                    "total_params": None,
                    "seed": None,
                    "window_id": window_id,
                    "started_at": meta.get("started_at"),
                    "finished_at": None,
                }
            )

        for t in conn.execute("SELECT * FROM traffic ORDER BY id"):
            tokens = estimate_tokens(t["text_snippet"])
            out["traffic"].append(
                {
                    "request_id": t["request_id"] or f"row_{t['id']}",
                    "ts": t["ts"],
                    "window_id": t["window_id"],
                    "run_id": t["run_id"],
                    "text_snippet": t["text_snippet"],
                    "gold_intent": t["gold_intent"],
                    "pred_intent": t["pred_intent"],
                    "confidence": t["confidence"],
                    "abstained": bool(t["abstained"]) if t["abstained"] is not None else None,
                    "in_scope": bool(t["in_scope"]) if t["in_scope"] is not None else None,
                    "latency_ms": t["app_latency_ms"],
                    "text_length_tokens": tokens,
                }
            )
            if not t["pred_intent"]:
                continue

            # Every served request becomes a prediction row; unlabeled
            # production traffic keeps a NULL true_label instead of a guess.
            out["prediction"].append(
                {
                    "experiment_id": t["run_id"],
                    "model_id": model_for(t["run_id"]),
                    "example_id": t["request_id"] or f"row_{t['id']}",
                    "true_label": t["gold_intent"],
                    "predicted_label": t["pred_intent"],
                    "prob_true_label": None,
                    "prob_predicted_label": t["confidence"],
                    "max_probability": t["confidence"],
                    "confidence_margin": None,
                    "entropy": None,
                    "n_tokens": tokens,
                    "window_id": t["window_id"],
                }
            )

            if t["app_latency_ms"] is not None:
                out["serving"].append(
                    {
                        "experiment_id": t["run_id"],
                        "model_id": model_for(t["run_id"]),
                        "quantisation": "as_served",
                        "latency_ms": t["app_latency_ms"],
                        "batch_size": None,
                        "input_tokens": tokens,
                        "model_size_bytes": None,
                        "peak_memory_mb": None,
                        "throughput_tps": None,
                        "request_index": int(t["id"]),
                        "ts": t["ts"],
                    }
                )
    finally:
        conn.close()
    return out


@dataclass
class SourceBatch:
    """All rows for one dataset from exactly one source.

    Origin is tracked per batch rather than per dataset because the same
    dataset legitimately arrives from two places: the real telemetry stream
    and the generated fine-tune artifacts. Collapsing them would lose the
    ``real``/``synthetic`` distinction that every fact row carries.
    """

    source_name: str
    data_origin: str
    dataset_name: str
    rows: list[dict[str, Any]]

    @property
    def row_count(self) -> int:
        return len(self.rows)


def extract_all(
    generated_dir: Path, telemetry_db: Path | None
) -> list[SourceBatch]:
    """Extract every source into origin-tagged batches."""
    batches: list[SourceBatch] = []

    for dataset, rows in extract_generated(generated_dir).items():
        if rows:
            batches.append(
                SourceBatch(
                    source_name=SOURCE_GENERATED,
                    data_origin="synthetic",
                    dataset_name=dataset,
                    rows=rows,
                )
            )

    if telemetry_db and Path(telemetry_db).is_file():
        for dataset, rows in extract_telemetry(Path(telemetry_db)).items():
            if rows:
                batches.append(
                    SourceBatch(
                        source_name=SOURCE_TELEMETRY,
                        data_origin="real",
                        dataset_name=dataset,
                        rows=rows,
                    )
                )

    return batches
