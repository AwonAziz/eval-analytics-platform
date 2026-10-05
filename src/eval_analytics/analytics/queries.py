"""The analytical queries -- the reason the warehouse exists.

Each function answers one real question about model quality and returns
plain Python structures (dicts and lists of dicts) so the same code backs the
SQL API, the dashboard and the tests.

Every function takes ``db_path`` and issues SQL against the marts; nothing
here recomputes a metric that the warehouse already stores, except the
statistics that require resampling or a significance test, which are computed
here from ``fact_prediction`` and checked against the stored value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..warehouse import query
from .stats import (
    DEFAULT_N_BOOTSTRAP,
    calibration,
    paired_bootstrap,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_ARMS = {"lora": "distilbert-base-uncased-lora-r16", "full_finetune": "distilbert-base-uncased-full-finetune"}


def available_arms(db_path: Path | str) -> pd.DataFrame:
    """Every arm currently in the warehouse, with its headline metrics."""
    return query(
        db_path,
        """
        SELECT m.arm, m.model_id, m.quantisation, m.trainable_params,
               r.n_samples, r.accuracy, r.ece, r.macro_f1, r.data_origin
        FROM v_model_quality_headline r
        JOIN dim_model m ON m.model_id = r.model_id
        WHERE r.n_samples > 0
        ORDER BY m.arm, m.quantisation
        """,
    )


def _arm_predictions(
    db_path: Path | str,
    arm: str,
    quantisation: str = "fp32",
    only_labelled: bool = True,
) -> pd.DataFrame:
    """Aligned per-example correctness for one arm.

    The two arms must be scored on the same documents, so this returns rows
    joined on ``example_id``; an inner join would silently shrink the sample
    if a future artifact set had a different test set, which would corrupt
    every paired statistic downstream. The caller checks ``n``.
    """
    sql = """
        SELECT p.example_id, p.true_label, p.predicted_label, p.is_correct,
               p.max_probability, p.prob_true_label, p.confidence_margin,
               p.entropy, p.n_tokens, p.doc_length_bucket_id
        FROM fact_prediction p
        JOIN dim_model m ON m.model_id = p.model_id
        WHERE m.arm = :arm AND m.quantisation = :quantisation
    """
    if only_labelled:
        sql += " AND p.is_correct IS NOT NULL"
    return query(db_path, sql, {"arm": arm, "quantisation": quantisation})


def _assert_paired(a: pd.DataFrame, b: pd.DataFrame, arm_a: str, arm_b: str) -> None:
    """Refuse to report paired statistics over different document sets."""
    if len(a) == 0 or len(b) == 0:
        raise ValueError(
            f"no labelled predictions for {arm_a} and/or {arm_b}; "
            "cannot compute a paired comparison"
        )
    only_a = set(a["example_id"]) - set(b["example_id"])
    only_b = set(b["example_id"]) - set(a["example_id"])
    if only_a or only_b:
        raise ValueError(
            f"arms are not scored on the same documents: "
            f"{len(only_a)} only in {arm_a}, {len(only_b)} only in {arm_b}"
        )


# ---------------------------------------------------------------------------
# 1. LoRA vs full fine-tune
# ---------------------------------------------------------------------------


def lora_vs_full_finetune(
    db_path: Path | str,
    arm_a: str = "lora",
    arm_b: str = "full_finetune",
    quantisation: str = "fp32",
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 20261005,
) -> dict[str, Any]:
    """Paired accuracy delta with bootstrap CI and exact McNemar p, overall
    and per class.

    The headline answers "is the arm difference real?"; the per-class table
    answers "where is it real?" -- an overall delta can be a wash between a
    large head-class win and a tail-class collapse, which is exactly the shape
    of the LoRA-vs-full-fine-tune trade-off.
    """
    a = _arm_predictions(db_path, arm_a, quantisation)
    b = _arm_predictions(db_path, arm_b, quantisation)
    _assert_paired(a, b, arm_a, arm_b)

    merged = a.merge(
        b[["example_id", "is_correct", "predicted_label", "max_probability"]],
        on="example_id",
        suffixes=("_a", "_b"),
        validate="one_to_one",
    )

    overall = paired_bootstrap(
        merged["is_correct_a"], merged["is_correct_b"], n_boot=n_boot, seed=seed
    )

    per_class = []
    for label, group in merged.groupby("true_label", sort=True):
        res = paired_bootstrap(
            group["is_correct_a"], group["is_correct_b"], n_boot=n_boot, seed=seed
        )
        row = {"true_label": label, **res.to_dict()}
        row["winner"] = (
            arm_a if res.delta > 0 else arm_b if res.delta < 0 else "tie"
        )
        row["supported_classes"] = len(group)
        per_class.append(row)

    conf = query(
        db_path,
        """
        SELECT m.arm, m.trainable_params, m.total_params, m.trainable_pct
        FROM dim_model m WHERE m.arm IN :arms
        """.replace(":arms", f"('{arm_a}', '{arm_b}')"),
    )
    param_lookup = {
        row["arm"]: {
            "trainable_params": row["trainable_params"],
            "total_params": row["total_params"],
            "trainable_pct": row["trainable_pct"],
        }
        for _, row in conf.iterrows()
    }

    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "quantisation": quantisation,
        "n_examples": int(len(merged)),
        "overall": overall.to_dict(),
        "per_class": per_class,
        "classes_won_by": {
            arm_a: sum(1 for r in per_class if r["winner"] == arm_a),
            arm_b: sum(1 for r in per_class if r["winner"] == arm_b),
        },
        "parameter_efficiency": {
            "trainable_params_a": param_lookup.get(arm_a, {}).get("trainable_params"),
            "trainable_params_b": param_lookup.get(arm_b, {}).get("trainable_params"),
            "trainable_param_ratio": _ratio(
                param_lookup.get(arm_a, {}).get("trainable_params"),
                param_lookup.get(arm_b, {}).get("trainable_params"),
            ),
            "accuracy_cost_per_param": _ratio(
                overall.delta,
                (param_lookup.get(arm_a, {}).get("trainable_params") or 0)
                - (param_lookup.get(arm_b, {}).get("trainable_params") or 0),
            ),
        },
    }


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or not denominator:
        return None
    return float(numerator) / float(denominator)


# ---------------------------------------------------------------------------
# 2. Calibration by arm
# ---------------------------------------------------------------------------


def calibration_by_arm(
    db_path: Path | str,
    arms: tuple[str, ...] = ("lora", "full_finetune"),
    quantisation: str = "fp32",
    n_bins: int = 15,
) -> dict[str, Any]:
    """Why is one arm's ECE lower than the other's?

    Each arm's ECE is decomposed into per-bucket contributions, so the
    answer is "these bins, weighted like this" rather than a bare pair of
    numbers. The stored ``fact_evaluation_run.ece`` is returned alongside the
    recomputed value so a drift between them is visible.
    """
    summaries = []
    for arm in arms:
        rows = _arm_predictions(db_path, arm, quantisation)
        if rows.empty:
            continue
        summary = calibration(
            model_id=rows["model_id"].iloc[0] if "model_id" in rows else arm,
            arm=arm,
            quantisation=quantisation,
            confidence=rows["max_probability"],
            correct=rows["is_correct"],
            prob_true=rows["prob_true_label"],
            n_bins=n_bins,
        )
        stored = query(
            db_path,
            """
            SELECT r.ece, r.mce, r.brier, r.accuracy, r.n_samples
            FROM fact_evaluation_run r
            JOIN dim_model m ON m.model_id = r.model_id
            JOIN dim_slice s ON s.slice_id = r.slice_id
            WHERE m.arm = :arm AND m.quantisation = :quantisation
              AND s.slice_type = 'overall' AND r.data_origin = 'synthetic'
            LIMIT 1
            """,
            {"arm": arm, "quantisation": quantisation},
        )
        summary.model_id = summary.model_id or arm
        entry = summary.to_dict()
        if not stored.empty:
            row = stored.iloc[0]
            entry["stored_ece"] = float(row["ece"]) if pd.notna(row["ece"]) else None
            entry["ece_matches_mart"] = (
                abs((entry["stored_ece"] or 0) - entry["ece"]) < 1e-6
            )
        summaries.append(entry)

    headline = {}
    for entry in summaries:
        top = sorted(entry["buckets"], key=lambda b: -b["ece_contribution"])[:3]
        headline[entry["arm"]] = {
            "ece": entry["ece"],
            "overconfidence": entry["overconfidence"],
            "worst_buckets": [
                {
                    "range": f"{b['bin_low']:.2f}-{b['bin_high']:.2f}",
                    "n": b["n"],
                    "accuracy": b["accuracy"],
                    "avg_confidence": b["avg_confidence"],
                    "gap": b["gap"],
                    "share_of_ece": b["share_of_ece"],
                }
                for b in top
            ],
        }

    return {"arms": summaries, "headline": headline, "n_bins": n_bins}


# ---------------------------------------------------------------------------
# 3. Quantisation: latency, size, agreement
# ---------------------------------------------------------------------------


def quantisation_comparison(
    db_path: Path | str,
    reference_quantisation: str = "fp32",
    target_quantisation: str = "int8",
) -> dict[str, Any]:
    """INT8 vs FP32 on three axes: latency, footprint, and agreement.

    Agreement is the one that matters for a release decision: a 3.8x smaller
    model is only worth shipping if it still makes the same predictions. The
    two builds are paired on ``base_model``/``arm``/``adapter_rank`` so an
    unrelated model can never be compared against them by accident.
    """
    latency = query(
        db_path,
        """
        SELECT s.quantisation,
               COUNT(*)                                        AS n_requests,
               median(s.latency_ms)                            AS p50_latency_ms,
               quantile_cont(s.latency_ms, 0.99)               AS p99_latency_ms,
               quantile_cont(s.latency_ms, 0.95)               AS p95_latency_ms,
               mean(s.latency_ms)                              AS mean_latency_ms,
               avg(s.model_size_bytes)                         AS model_size_bytes,
               any_value(s.data_origin)                        AS data_origin
        FROM fact_serving_request s
        GROUP BY s.quantisation
        ORDER BY median(s.latency_ms)
        """,
    )

    by_tokens = query(
        db_path,
        """
        SELECT s.quantisation, s.input_tokens,
               COUNT(*)                          AS n_requests,
               median(s.latency_ms)              AS p50_latency_ms,
               quantile_cont(s.latency_ms, 0.99) AS p99_latency_ms
        FROM fact_serving_request s
        WHERE s.batch_size = 1 AND s.input_tokens IS NOT NULL
        GROUP BY s.quantisation, s.input_tokens
        ORDER BY s.input_tokens, s.quantisation
        """,
    )

    agreement = _prediction_agreement(
        db_path, reference_quantisation, target_quantisation
    )

    ref = latency[latency["quantisation"] == reference_quantisation]
    tgt = latency[latency["quantisation"] == target_quantisation]
    ratios = None
    if not ref.empty and not tgt.empty:
        r = ref.iloc[0]
        t = tgt.iloc[0]
        ratios = {
            "p50_speedup": _ratio(r["p50_latency_ms"], t["p50_latency_ms"]),
            "p99_speedup": _ratio(r["p99_latency_ms"], t["p99_latency_ms"]),
            "size_reduction": _ratio(r["model_size_bytes"], t["model_size_bytes"]),
        }

    return {
        "latency_by_quantisation": latency.to_dict(orient="records"),
        "latency_by_input_tokens": by_tokens.to_dict(orient="records"),
        "agreement": agreement,
        "ratios": ratios,
        "reference_quantisation": reference_quantisation,
        "target_quantisation": target_quantisation,
    }


def _prediction_agreement(
    db_path: Path | str, reference_quantisation: str, target_quantisation: str
) -> dict[str, Any]:
    """Agreement between two quantisations of the *same* logical model.

    The two builds are matched on model identity -- base model, arm and
    adapter rank -- not merely on quantisation. Filtering only by quantisation
    would happily pair an FP32 *full fine-tune* against an INT8 *LoRA* and
    report the difference as quantisation error.
    """
    paired = query(
        db_path,
        """
        WITH models AS (
            SELECT model_id, base_model, arm, adapter_rank, quantisation
            FROM dim_model
            WHERE data_origin = 'synthetic'
        ),
        ref AS (
            SELECT m.base_model, m.arm, m.adapter_rank, m.model_id,
                   p.example_id, p.predicted_label AS ref_label, p.is_correct AS ref_correct
            FROM fact_prediction p
            JOIN models m ON m.model_id = p.model_id
            WHERE m.quantisation = :ref_quant
        ),
        tgt AS (
            SELECT m.base_model, m.arm, m.adapter_rank, m.model_id,
                   p.example_id, p.predicted_label AS tgt_label, p.is_correct AS tgt_correct
            FROM fact_prediction p
            JOIN models m ON m.model_id = p.model_id
            WHERE m.quantisation = :tgt_quant
        )
        SELECT ref.base_model, ref.arm, ref.adapter_rank,
               ref.model_id AS reference_model, tgt.model_id AS target_model,
               COUNT(*)                                                   AS n_paired,
               SUM(CASE WHEN ref.ref_label = tgt.tgt_label THEN 1 ELSE 0 END) AS n_agree,
               SUM(CASE WHEN ref.ref_correct = 1 AND tgt.tgt_correct = 0 THEN 1 ELSE 0 END) AS reference_only_right,
               SUM(CASE WHEN ref.ref_correct = 0 AND tgt.tgt_correct = 1 THEN 1 ELSE 0 END) AS target_only_right,
               AVG(CASE WHEN ref.ref_label = tgt.tgt_label THEN 1.0 ELSE 0.0 END) AS agreement_rate
        FROM ref
        JOIN tgt
          ON tgt.example_id = ref.example_id
         AND tgt.base_model = ref.base_model
         AND tgt.arm = ref.arm
         AND (tgt.adapter_rank IS NOT DISTINCT FROM ref.adapter_rank)
        GROUP BY ref.base_model, ref.arm, ref.adapter_rank,
                 ref.model_id, tgt.model_id
        ORDER BY ref.model_id
        """,
        {"ref_quant": reference_quantisation, "tgt_quant": target_quantisation},
    )

    if paired.empty:
        return {
            "available": False,
            "reason": (
                f"no prediction pairs found for quantisation "
                f"{reference_quantisation} vs {target_quantisation}"
            ),
        }

    result = paired.to_dict(orient="records")
    for entry in result:
        entry["disagreements"] = entry["n_paired"] - entry["n_agree"]
    # The summary is the first pair record rather than a separately derived
    # dict, so `disagreements` cannot drift out of agreement with its inputs.
    return {"available": True, "pairs": result, "summary": result[0]}


# ---------------------------------------------------------------------------
# 4. Drift trend per feature
# ---------------------------------------------------------------------------


def drift_trend(
    db_path: Path | str,
    feature: str | None = None,
    metric: str | None = None,
    metric_value: float = 0.10,
    severe_value: float = 0.25,
) -> dict[str, Any]:
    """PSI / KS / JS per feature over time, with threshold crossings.

    The slope is the part worth reporting: a feature sitting at a high PSI is
    less urgent than one climbing steadily, and only the trend says which.
    """
    where = []
    params: dict[str, Any] = {"moderate": metric_value, "severe": severe_value}
    if feature:
        where.append("d.feature_id = :feature")
        params["feature"] = feature
    if metric:
        where.append("d.metric_name = :metric")
        params["metric"] = metric
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    series = query(
        db_path,
        f"""
        SELECT d.feature_id, f.feature_type, d.window_id, d.measurement_ts, d.date_id,
               MAX(CASE WHEN d.metric_name = 'psi'           THEN d.metric_value END) AS psi,
               MAX(CASE WHEN d.metric_name = 'ks_statistic'  THEN d.metric_value END) AS ks_statistic,
               MAX(CASE WHEN d.metric_name = 'js_divergence' THEN d.metric_value END) AS js_divergence,
               ANY_VALUE(d.metric_name) AS metric_name,
               ANY_VALUE(d.metric_value) AS metric_value,
               MAX(d.severity)   AS severity,
               MAX(d.sample_size) AS sample_size,
               ANY_VALUE(d.data_origin) AS data_origin
        FROM fact_drift_measurement d
        JOIN dim_feature f ON f.feature_id = d.feature_id
        {clause}
        GROUP BY d.feature_id, f.feature_type, d.window_id, d.measurement_ts, d.date_id
        ORDER BY d.feature_id, d.measurement_ts
        """,
        params,
    )

    summaries = []
    for feature_id, group in series.groupby("feature_id", sort=True):
        ordered = group.sort_values("measurement_ts")
        psi = ordered["psi"].dropna()
        latest = ordered.iloc[-1]
        slope = float(np.polyfit(np.arange(len(psi)), psi.to_numpy(), 1)[0]) if len(psi) > 1 else 0.0

        crossings = _threshold_crossings(ordered, "psi", metric_value, severe_value)
        psi_first = float(psi.iloc[0]) if len(psi) else None
        psi_latest = float(psi.iloc[-1]) if len(psi) else None
        # The trend label follows the endpoint change, not the fitted slope.
        # A least-squares slope can be positive on a series that ends lower
        # than it started (a V shape), and calling that "rising" would
        # mislead an on-call engineer reading an alert.
        change = (psi_latest - psi_first) if (psi_first is not None and psi_latest is not None) else None
        summaries.append(
            {
                "feature_id": feature_id,
                "feature_type": latest["feature_type"],
                "data_origin": latest["data_origin"],
                "n_windows": int(len(group)),
                "first_ts": ordered.iloc[0]["measurement_ts"],
                "last_ts": latest["measurement_ts"],
                "psi_first": psi_first,
                "psi_latest": psi_latest,
                "psi_change": change,
                "psi_max": float(psi.max()) if len(psi) else None,
                "psi_slope_per_window": slope,
                "ks_latest": float(latest["ks_statistic"]) if pd.notna(latest["ks_statistic"]) else None,
                "js_latest": float(latest["js_divergence"]) if pd.notna(latest["js_divergence"]) else None,
                "severity_latest": latest["severity"],
                "crossed_moderate": crossings["moderate"],
                "crossed_severe": crossings["severe"],
                "trend": (
                    "rising" if change is not None and change > 0.01
                    else "falling" if change is not None and change < -0.01
                    else "flat"
                ),
            }
        )

    alerts = [
        s
        for s in summaries
        if s["crossed_moderate"] or s["trend"] == "rising" or (s["psi_slope_per_window"] or 0) > 0.0005
    ]
    alerts.sort(key=lambda s: (-(s["psi_change"] or 0), -(s["psi_slope_per_window"] or 0)))

    return {
        "thresholds": {"moderate": metric_value, "severe": severe_value},
        "features": summaries,
        "series": series.to_dict(orient="records"),
        "alerts": alerts,
    }


def _threshold_crossings(
    group: pd.DataFrame, column: str, moderate: float, severe: float
) -> dict[str, Any]:
    """First window in which a feature crossed each threshold."""
    values = group[column].dropna()
    if values.empty:
        return {"moderate": None, "severe": None}

    def first_above(threshold: float) -> dict[str, Any] | None:
        hits = group[group[column] > threshold]
        if hits.empty:
            return None
        row = hits.iloc[0]
        return {
            "window_id": row["window_id"],
            "measured_at": row["measurement_ts"],
            "value": float(row[column]),
        }

    return {"moderate": first_above(moderate), "severe": first_above(severe)}


# ---------------------------------------------------------------------------
# 5. Error rate by document-length bucket
# ---------------------------------------------------------------------------


def error_by_length_bucket(
    db_path: Path | str,
    quantisation: str = "fp32",
) -> dict[str, Any]:
    """Error rate against document length, per arm.

    Also reports each bucket's error rate *relative to its neighbours*, which
    is what identifies a genuine anomaly. A smooth decline in accuracy as
    documents get shorter is ordinary; a bucket that is worse than the trend
    through it is a specific, fixable defect.
    """
    per_arm = query(
        db_path,
        """
        SELECT m.arm, m.model_id, s.slice_id, s.slice_name, s.ordinal,
               COUNT(*)                                   AS n_samples,
               SUM(p.is_correct)                          AS n_correct,
               AVG(p.is_correct)                          AS accuracy,
               AVG(p.max_probability)                     AS mean_confidence,
               SUM(CASE WHEN p.is_correct = 0 THEN 1 ELSE 0 END) AS n_errors
        FROM fact_prediction p
        JOIN dim_model m ON m.model_id = p.model_id
        JOIN dim_slice s ON s.slice_id = p.doc_length_bucket_id
        WHERE s.slice_type = 'length_bucket'
          AND m.quantisation = :quantisation
          AND p.true_label IS NOT NULL
        GROUP BY m.arm, m.model_id, s.slice_id, s.slice_name, s.ordinal
        ORDER BY m.arm, s.ordinal
        """,
        {"quantisation": quantisation},
    )

    arms: dict[str, Any] = {}
    for arm, group in per_arm.groupby("arm", sort=True):
        ordered = group.sort_values("ordinal")
        rows = ordered.to_dict(orient="records")
        for row in rows:
            row["error_rate"] = 1.0 - row["accuracy"]
            row["n_errors"] = int(row["n_errors"])

        # Excess error against the mean of the two neighbouring buckets.
        for i, row in enumerate(rows):
            neighbours = []
            if i > 0:
                neighbours.append(rows[i - 1]["error_rate"])
            if i < len(rows) - 1:
                neighbours.append(rows[i + 1]["error_rate"])
            baseline = float(np.mean(neighbours)) if neighbours else float("nan")
            row["neighbour_error_rate"] = baseline
            row["excess_error_vs_neighbours"] = (
                row["error_rate"] - baseline if neighbours else None
            )
            row["lift"] = (
                row["error_rate"] / baseline
                if neighbours and baseline > 0
                else None
            )

        worst = max(rows, key=lambda r: r["excess_error_vs_neighbours"] or -1)
        arms[arm] = {
            "buckets": rows,
            "worst_bucket": worst["slice_name"],
            "worst_bucket_excess_error": worst["excess_error_vs_neighbours"],
            "worst_bucket_lift": worst["lift"],
        }

    return {"arms": arms, "quantisation": quantisation}


def confusion_by_length(
    db_path: Path | str,
    arm: str = "lora",
    quantisation: str = "fp32",
    slice_id: str | None = None,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    """Which true/predicted label pairs dominate the worst length bucket.

    Ties the length finding to a specific mechanism: if one labelled
    confusion dominates a bucket, the bucket has a content problem, not a
    size problem.
    """
    params: dict[str, Any] = {"arm": arm, "quantisation": quantisation, "top_n": top_n}
    clause = "AND p.doc_length_bucket_id = :slice_id" if slice_id else ""
    if slice_id:
        params["slice_id"] = slice_id

    return query(
        db_path,
        f"""
        SELECT p.true_label, p.predicted_label,
               COUNT(*)                        AS n,
               COUNT(*) * 1.0 / NULLIF(SUM(COUNT(*)) OVER (), 0) AS share_of_bucket_errors,
               AVG(p.max_probability)          AS mean_confidence
        FROM fact_prediction p
        JOIN dim_model m ON m.model_id = p.model_id
        WHERE m.arm = :arm AND m.quantisation = :quantisation
          AND p.is_correct = 0
          {clause}
        GROUP BY p.true_label, p.predicted_label
        ORDER BY n DESC
        LIMIT :top_n
        """,
        params,
    ).to_dict(orient="records")


# ---------------------------------------------------------------------------
# 6. Which slices does an arm lose on?
# ---------------------------------------------------------------------------


def slice_losses(
    db_path: Path | str,
    arm: str = "lora",
    against: str = "full_finetune",
    quantisation: str = "fp32",
    n_boot: int = 2000,
    minimum_examples: int = 30,
) -> dict[str, Any]:
    """Every slice on which ``arm`` is behind ``against``, and by how much.

    Reads the slice projection once and pivots per slice, so all three slice
    families come from a single pass. Each measured slice gets a paired
    bootstrap CI and an exact McNemar p; slices with too few examples to judge
    are returned separately rather than reported as small wins or losses.
    """
    long = query(
        db_path,
        """
        SELECT slice_id, slice_type, slice_name, ordinal, example_id,
               MAX(CASE WHEN arm = :arm THEN is_correct END) AS correct_arm,
               MAX(CASE WHEN arm = :against THEN is_correct END) AS correct_against
        FROM v_prediction_slice
        WHERE arm IN (:arm, :against) AND quantisation = :quantisation
          AND is_correct IS NOT NULL
        GROUP BY slice_id, slice_type, slice_name, ordinal, example_id
        """,
        {"arm": arm, "against": against, "quantisation": quantisation},
    )

    by_family: dict[str, list[dict[str, Any]]] = {
        "class": [],
        "length_bucket": [],
        "drift_window": [],
    }
    thin: list[dict[str, Any]] = []

    if long.empty:
        return {
            "arm": arm,
            "against": against,
            "quantisation": quantisation,
            "overall": None,
            "losses": [],
            "wins": [],
            "too_few_examples": [],
            "by_family": by_family,
            "summary": {"n_slices_measured": 0, "n_losses": 0, "n_wins": 0, "significant_losses": 0},
        }

    for (slice_id, slice_type, slice_name), group in long.groupby(
        ["slice_id", "slice_type", "slice_name"], sort=True
    ):
        paired = group.dropna(subset=["correct_arm", "correct_against"])
        if paired.empty:
            continue

        entry = {
            "slice_id": slice_id,
            "slice_name": slice_name,
            "slice_type": slice_type,
            "n_examples": int(len(paired)),
            "accuracy_arm": float(paired["correct_arm"].mean()),
            "accuracy_against": float(paired["correct_against"].mean()),
            "delta": float(paired["correct_arm"].mean() - paired["correct_against"].mean()),
        }

        if len(paired) < minimum_examples:
            entry["status"] = "too_few_examples"
            thin.append(entry)
            continue

        res = paired_bootstrap(
            paired["correct_arm"], paired["correct_against"], n_boot=n_boot, seed=20261005
        )
        entry.update(
            {
                "status": "measured",
                "ci_low": res.ci_low,
                "ci_high": res.ci_high,
                "p_value": res.p_value,
                "significant": res.significant,
                "n_arm_only_right": res.n_a_only,
                "n_against_only_right": res.n_b_only,
            }
        )
        by_family.setdefault(slice_type, []).append(entry)

    for rows in by_family.values():
        rows.sort(key=lambda r: r["delta"])

    losses = [r for rows in by_family.values() for r in rows if r["delta"] < 0]
    wins = [r for rows in by_family.values() for r in rows if r["delta"] > 0]
    losses.sort(key=lambda r: r["delta"])
    wins.sort(key=lambda r: -r["delta"])
    thin.sort(key=lambda r: r["n_examples"])

    overall = query(
        db_path,
        """
        SELECT
          (SELECT AVG(is_correct) FROM v_prediction_slice
            WHERE arm = :arm AND quantisation = :q AND slice_type = 'overall'
              AND is_correct IS NOT NULL) AS accuracy_arm,
          (SELECT AVG(is_correct) FROM v_prediction_slice
            WHERE arm = :against AND quantisation = :q AND slice_type = 'overall'
              AND is_correct IS NOT NULL) AS accuracy_against
        """,
        {"arm": arm, "against": against, "q": quantisation},
    )

    overall_row = None
    if not overall.empty and pd.notna(overall.iloc[0]["accuracy_arm"]):
        row = overall.iloc[0]
        overall_row = {
            "accuracy_arm": float(row["accuracy_arm"]),
            "accuracy_against": float(row["accuracy_against"]),
            "delta": float(row["accuracy_arm"] - row["accuracy_against"]),
        }

    return {
        "arm": arm,
        "against": against,
        "quantisation": quantisation,
        "overall": overall_row,
        "losses": losses,
        "wins": wins,
        "too_few_examples": thin,
        "by_family": by_family,
        "summary": {
            "n_slices_measured": len(losses) + len(wins),
            "n_losses": len(losses),
            "n_wins": len(wins),
            "significant_losses": sum(1 for r in losses if r.get("significant")),
        },
    }


# ---------------------------------------------------------------------------
# Supporting views
# ---------------------------------------------------------------------------


def data_quality_summary(db_path: Path | str) -> dict[str, Any]:
    """Counts and provenance across the whole warehouse."""
    counts = query(
        db_path,
        """
        SELECT 'fact_evaluation_run' AS object, COUNT(*) AS rows FROM fact_evaluation_run
        UNION ALL SELECT 'fact_prediction', COUNT(*) FROM fact_prediction
        UNION ALL SELECT 'fact_serving_request', COUNT(*) FROM fact_serving_request
        UNION ALL SELECT 'fact_drift_measurement', COUNT(*) FROM fact_drift_measurement
        UNION ALL SELECT 'ingestion_rejection', COUNT(*) FROM ingestion_rejection
        ORDER BY object
        """,
    )
    provenance = query(
        db_path,
        """
        SELECT data_origin,
               COUNT(*)                                   AS predictions,
               SUM(CASE WHEN true_label IS NULL THEN 1 ELSE 0 END) AS unlabelled,
               MIN(date_id) AS first_date, MAX(date_id) AS last_date
        FROM fact_prediction
        GROUP BY data_origin
        ORDER BY data_origin
        """,
    )
    latest_run = query(
        db_path,
        """
        SELECT pipeline_run_id, status, rows_landed, rows_accepted, rows_rejected,
               started_at, finished_at
        FROM ingestion_run ORDER BY started_at DESC LIMIT 1
        """,
    )
    return {
        "counts": counts.to_dict(orient="records"),
        "provenance": provenance.to_dict(orient="records"),
        "latest_ingestion_run": (
            latest_run.iloc[0].to_dict() if not latest_run.empty else None
        ),
    }
