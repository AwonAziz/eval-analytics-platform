"""Streamlit dashboard for the evaluation and telemetry warehouse.

Live Plotly charts over the same analytical layer the SQL API serves, so the
dashboard and the API can never disagree. Reads the DuckDB file directly --
no HTTP hop -- and caches query results so redrawing a chart is instant.

Run with::

    streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from eval_analytics.analytics import queries as Q
from eval_analytics.config import SETTINGS
from eval_analytics.quality import run_gate

DB_PATH = Path(os.environ.get("EAP_DUCKDB_PATH", SETTINGS.duckdb_path))

CHECK_MARK = "\u2713"
ELLIPSIS = "\u2026"
TIMES = "\u00d7"
ARROW = "\u2192"
MIDDOT = "\u00b7"
DASH = "\u2014"
EN_DASH = "\u2013"
MINUS = "\u2212"

ACCENT = {
    "lora": "#4C8BF5",
    "full_finetune": "#E4572E",
    "none": "#7F8C8D",
}
QUANT_COLOR = {"fp32": "#4C8BF5", "fp16": "#54A24B", "int8": "#E4572E"}
GRID = {"showgrid": False, "zeroline": False}

st.set_page_config(
    page_title="Eval Analytics",
    page_icon="\N{BAR CHART}",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def _db() -> str:
    return str(DB_PATH)


@st.cache_data(show_spinner="running analysis...", ttl=300)
def arms() -> pd.DataFrame:
    return Q.available_arms(_db())


@st.cache_data(show_spinner="running analysis...", ttl=300)
def lora_vs_full(arm_a: str, arm_b: str, quantisation: str, n_boot: int) -> dict:
    return Q.lora_vs_full_finetune(
        _db(), arm_a=arm_a, arm_b=arm_b, quantisation=quantisation, n_boot=n_boot
    )


@st.cache_data(show_spinner="running analysis...", ttl=300)
def calibration(quantisation: str, n_bins: int) -> dict:
    return Q.calibration_by_arm(_db(), quantisation=quantisation, n_bins=n_bins)


@st.cache_data(show_spinner="running analysis...", ttl=300)
def quantisation() -> dict:
    return Q.quantisation_comparison(_db())


@st.cache_data(show_spinner="running analysis...", ttl=300)
def drift() -> dict:
    return Q.drift_trend(_db())


@st.cache_data(show_spinner="running analysis...", ttl=300)
def length_buckets(quantisation: str) -> dict:
    return Q.error_by_length_bucket(_db(), quantisation=quantisation)


@st.cache_data(show_spinner="running analysis...", ttl=300)
def confusions(arm: str, slice_id: str) -> list[dict]:
    return Q.confusion_by_length(_db(), arm=arm, slice_id=slice_id)


@st.cache_data(show_spinner="running analysis...", ttl=300)
def slice_losses(arm: str, against: str, n_boot: int) -> dict:
    return Q.slice_losses(_db(), arm=arm, against=against, n_boot=n_boot)


@st.cache_data(show_spinner=False, ttl=300)
def quality() -> dict:
    return run_gate(_db()).to_dict()


@st.cache_data(show_spinner=False, ttl=300)
def summary() -> dict:
    return Q.data_quality_summary(_db())


def _fmt_p(value: float | None) -> str:
    if value is None:
        return ELLIPSIS
    if value < 1e-4:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _require_warehouse() -> None:
    if not DB_PATH.exists():
        st.error(
            f"No warehouse at `{DB_PATH}`.\n\n"
            "Build one first:\n\n```\neap build\n```"
        )
        st.stop()


def sidebar() -> dict:
    st.sidebar.title("Controls")
    fine_tune_arms = sorted(
        arms().query("data_origin == 'synthetic'")["arm"].unique().tolist()
    )
    default_a = "lora" if "lora" in fine_tune_arms else (
        fine_tune_arms[0] if fine_tune_arms else ""
    )

    arm_a = st.sidebar.selectbox(
        "Baseline arm",
        fine_tune_arms,
        index=fine_tune_arms.index(default_a) if default_a in fine_tune_arms else 0,
    )
    arm_b = st.sidebar.selectbox(
        "Compared arm",
        [a for a in fine_tune_arms if a != arm_a] or fine_tune_arms,
        index=0,
    )
    quantisation = st.sidebar.selectbox("Quantisation", ["fp32", "fp16", "int8"], index=0)
    n_boot = st.sidebar.slider("Bootstrap resamples", 500, 10_000, 4000, step=500)
    n_bins = st.sidebar.slider("Calibration bins", 5, 30, 15)

    st.sidebar.divider()
    st.sidebar.caption(f"Warehouse: `{DB_PATH.name}`")
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "quantisation": quantisation,
        "n_boot": n_boot,
        "n_bins": n_bins,
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


def page_overview(cfg: dict) -> None:
    st.title("Warehouse overview")
    data = summary()
    gate = quality()

    counts = {row["object"]: row["rows"] for row in data["counts"]}
    rejections = counts.get("ingestion_rejection", 0)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Predictions", f"{counts.get('fact_prediction', 0):,}")
    c2.metric("Evaluation runs", f"{counts.get('fact_evaluation_run', 0):,}")
    c3.metric("Serving requests", f"{counts.get('fact_serving_request', 0):,}")
    c4.metric("Drift measurements", f"{counts.get('fact_drift_measurement', 0):,}")
    c5.metric(
        "Rejected rows",
        f"{rejections:,}",
        delta=f"0 {CHECK_MARK}" if rejections == 0 else "investigate",
        delta_color="normal" if rejections == 0 else "inverse",
    )

    st.subheader("Data provenance")
    prov = pd.DataFrame(data["provenance"])
    if not prov.empty:
        prov = prov.copy()
        prov["predictions"] = prov["predictions"].map(lambda v: f"{v:,}")
        st.dataframe(prov, width="stretch", hide_index=True)
        st.caption(
            "Real rows are measured production telemetry. Synthetic rows come from "
            "deterministic fine-tune artifacts. Both are labelled per row via "
            "`data_origin`, so any figure can be filtered back to measured data only."
        )

    st.subheader("Quality gate")
    gate_df = pd.DataFrame(
        [
            {
                "check": c["name"],
                "passed": "PASS" if c["passed"] else "FAIL",
                "detail": c["detail"],
            }
            for c in gate["checks"]
        ]
    )
    st.dataframe(gate_df, width="stretch", hide_index=True)

    st.subheader("Headline model quality")
    frame = arms()
    if "synthetic" in set(frame["data_origin"]):
        frame = frame[frame["data_origin"] == "synthetic"]
    view = frame.copy()
    for col in ("accuracy", "ece", "macro_f1"):
        if col in view:
            view[col] = view[col].map(lambda v: round(float(v), 4) if pd.notna(v) else None)
    st.dataframe(view, width="stretch", hide_index=True)


def page_arm_comparison(cfg: dict) -> None:
    arm_a, arm_b = cfg["arm_a"], cfg["arm_b"]
    st.title(f"Paired comparison: {arm_a} vs {arm_b}")
    st.caption(
        "Both arms are scored on the same documents, so the accuracy difference is a "
        "paired statistic. The confidence interval comes from a paired bootstrap over "
        "documents; the p-value is an exact McNemar test on the discordant pairs."
    )

    result = lora_vs_full(arm_a, arm_b, cfg["quantisation"], cfg["n_boot"])
    overall = result["overall"]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Paired examples", f"{overall['n']:,}")
    k2.metric(f"{arm_a} accuracy", f"{overall['mean_a']:.4f}")
    k3.metric(f"{arm_b} accuracy", f"{overall['mean_b']:.4f}")
    k4.metric(
        "Delta",
        f"{overall['delta']:+.4f}",
        delta=f"CI [{overall['ci_low']:+.4f}, {overall['ci_high']:+.4f}]",
        delta_color="normal" if overall["significant"] else "off",
    )
    k5.metric(
        "McNemar p",
        _fmt_p(overall["p_value"]),
        delta="significant" if overall["significant"] else "not significant",
        delta_color="normal" if overall["significant"] else "off",
    )

    eff = result["parameter_efficiency"]
    if eff.get("trainable_param_ratio") and eff.get("trainable_params_b"):
        st.info(
            f"**Parameter efficiency.** {arm_a} trains "
            f"{eff['trainable_params_a']:,} parameters against {eff['trainable_params_b']:,} "
            f"for {arm_b} {DASH} {eff['trainable_param_ratio'] * 100:.2f}% of the budget {DASH} "
            f"giving up {abs(overall['delta']) * 100:.2f} points of accuracy."
        )

    st.subheader("Per-class accuracy delta, with 95% bootstrap CI")
    per_class = pd.DataFrame(result["per_class"]).sort_values("delta")
    fig = go.Figure()
    colors = [
        ACCENT.get(arm_a, "#4C8BF5") if d > 0
        else ACCENT.get(arm_b, "#E4572E") if d < 0
        else "#999999"
        for d in per_class["delta"]
    ]
    fig.add_trace(
        go.Bar(
            x=per_class["delta"],
            y=per_class["true_label"],
            orientation="h",
            marker_color=colors,
            error_x=dict(
                type="data",
                symmetric=False,
                array=per_class["ci_high"] - per_class["delta"],
                arrayminus=per_class["delta"] - per_class["ci_low"],
            ),
            text=[f"n={n}" for n in per_class["n"]],
            textposition="outside",
            customdata=np.column_stack(
                [
                    per_class["ci_low"],
                    per_class["ci_high"],
                    per_class["p_value"],
                ]
            )
            if len(per_class)
            else [],
            hovertemplate=(
                "%{y}<br>delta %{x:+.4f}"
                "<br>CI [%{customdata[0]:+.4f}, %{customdata[1]:+.4f}]"
                "<br>p=%{customdata[2]:.4f}<br>%{text}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_color="#666", line_width=1)
    fig.update_layout(
        height=max(360, 42 * len(per_class)),
        xaxis_title=f"accuracy delta ({arm_a} {MINUS} {arm_b})",
        yaxis_title=None,
        showlegend=False,
        margin=dict(l=10, r=60, t=10, b=40),
    )
    fig.update_xaxes(zeroline=True, zerolinecolor="#666")
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, width="stretch")

    won = result["classes_won_by"]
    st.caption(
        f"**{won.get(arm_a, 0)} classes won** by {arm_a} {MIDDOT} "
        f"**{won.get(arm_b, 0)}** by {arm_b}. "
        "A positive delta means the baseline arm is ahead on that class."
    )
    st.dataframe(
        per_class[
            [
                "true_label", "n", "mean_a", "mean_b", "delta",
                "ci_low", "ci_high", "p_value", "significant",
            ]
        ],
        width="stretch",
        hide_index=True,
    )


def page_calibration(cfg: dict) -> None:
    st.title("Calibration by arm")
    st.caption(
        "Expected calibration error is the confidence-weighted gap between stated "
        "confidence and observed accuracy. Each bar below contributes "
        f"`n {TIMES} gap / N` to the arm's ECE, so the buckets that dominate the metric "
        "are visible directly."
    )

    result = calibration(cfg["quantisation"], cfg["n_bins"])

    for entry in result["arms"]:
        arm = entry["arm"]
        colour = ACCENT.get(arm, "#4C8BF5")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"{arm} ECE", f"{entry['ece']:.6f}")
        k2.metric("Accuracy", f"{entry['accuracy']:.4f}")
        k3.metric("Mean confidence", f"{entry['mean_confidence']:.4f}")
        k4.metric(
            "Overconfidence",
            f"{entry['overconfidence']:+.4f}",
            help="mean confidence minus accuracy; positive means the model overstates confidence",
        )

        left, right = st.columns([3, 2])
        bins = pd.DataFrame(entry["buckets"])

        with left:
            if not bins.empty:
                fig = px.scatter(
                    bins,
                    x="avg_confidence",
                    y="accuracy",
                    size="n",
                    color_discrete_sequence=[colour],
                    size_max=42,
                    custom_data=["bin_low", "bin_high", "n", "gap"],
                    labels={"avg_confidence": "mean confidence", "accuracy": "observed accuracy"},
                )
                lo, hi = float(bins["bin_low"].min()), float(bins["bin_high"].max())
                fig.add_trace(
                    go.Scatter(
                        x=[lo, hi], y=[lo, hi], mode="lines",
                        name="perfect calibration",
                        line=dict(dash="dash", color="#999"),
                    )
                )
                fig.update_traces(
                    marker=dict(line=dict(width=1, color="white")),
                    hovertemplate=(
                        "conf %{x:.3f}<br>acc %{y:.3f}<br>gap %{customdata[3]:.4f}"
                        "<br>bucket %{customdata[0]:.2f}-{customdata[1]:.2f}"
                        "<br>n=%{customdata[2]}<extra></extra>"
                    ),
                )
                fig.update_layout(
                    height=380, xaxis_range=[lo, hi], yaxis_range=[lo, hi],
                    showlegend=False, margin=dict(l=10, r=10, t=10, b=40),
                )
                fig.update_xaxes(**GRID)
                fig.update_yaxes(**GRID)
                st.plotly_chart(fig, width="stretch")

        with right:
            if not bins.empty:
                contrib = bins.sort_values("ece_contribution", ascending=False).head(8)
                fig = px.bar(
                    contrib,
                    x="ece_contribution",
                    y="bin_low",
                    orientation="h",
                    color_discrete_sequence=[colour],
                    labels={
                        "ece_contribution": "contribution to ECE",
                        "bin_low": "confidence bucket (lower bound)",
                    },
                    custom_data=["bin_high", "share_of_ece"],
                    text=contrib["share_of_ece"].map(lambda v: f"{v * 100:.0f}%"),
                )
                fig.update_traces(
                    textposition="outside",
                    hovertemplate=(
                        "bucket %{customdata[0]:.2f}-{customdata[1]:.2f}"
                        "<br>contributes %{x:.5f} (%{customdata[2] * 100:.1f}% of ECE)"
                        "<extra></extra>"
                    ),
                )
                fig.update_layout(height=380, showlegend=False, margin=dict(l=10, r=40, t=10, b=40))
                fig.update_xaxes(**GRID)
                fig.update_yaxes(**GRID)
                st.plotly_chart(fig, width="stretch")
        st.divider()

    stored = [(e["arm"], e.get("stored_ece")) for e in result["arms"] if e.get("stored_ece")]
    if stored:
        arms_txt = f" {MIDDOT} ".join(f"{a}: {v:.6f}" for a, v in stored)
        st.caption(f"Recomputed ECE matches `fact_evaluation_run.ece` {DASH} {arms_txt}")


def page_quantisation(cfg: dict) -> None:
    st.title("Quantisation: latency, footprint, agreement")
    result = quantisation()

    ratios = result.get("ratios") or {}
    k1, k2, k3 = st.columns(3)
    if ratios:
        k1.metric("p50 speedup", f"{ratios['p50_speedup']:.2f}{TIMES}")
        k2.metric("p99 speedup", f"{ratios['p99_speedup']:.2f}{TIMES}")
        k3.metric("Size reduction", f"{ratios['size_reduction']:.2f}{TIMES}")

    latency = pd.DataFrame(result["latency_by_quantisation"])
    # The real telemetry series measures end-to-end application latency, which
    # includes far more than the model call and is not comparable to the
    # model-only benchmark. Mixing them on one axis would flatten the real
    # comparison into nothing.
    if not latency.empty and "data_origin" in latency:
        bench = latency[latency["data_origin"] == "synthetic"].copy()
        served = latency[latency["data_origin"] != "synthetic"].copy()
    else:
        bench, served = latency, latency.iloc[0:0]

    left, right = st.columns(2)

    with left:
        if not bench.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=bench["quantisation"], y=bench["p50_latency_ms"], name="p50",
                    marker_color=[QUANT_COLOR.get(q, "#999") for q in bench["quantisation"]],
                    hovertemplate="%{x}<br>p50 %{y:.3f} ms<extra></extra>",
                )
            )
            fig.add_trace(
                go.Bar(
                    x=bench["quantisation"], y=bench["p99_latency_ms"], name="p99",
                    marker_color=[QUANT_COLOR.get(q, "#999") for q in bench["quantisation"]],
                    opacity=0.45,
                    hovertemplate="%{x}<br>p99 %{y:.3f} ms<extra></extra>",
                )
            )
            fig.update_layout(
                barmode="group", height=360, title="Model-call latency",
                yaxis_title="milliseconds",
            )
            fig.update_xaxes(**GRID)
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, width="stretch")

    with right:
        if not bench.empty:
            sizes = bench.copy()
            sizes["size_mb"] = sizes["model_size_bytes"] / 1e6
            fig = px.bar(
                sizes, x="quantisation", y="size_mb", color="quantisation",
                color_discrete_map=QUANT_COLOR,
                labels={"size_mb": "model size (MB)"},
                text="size_mb",
            )
            fig.update_traces(texttemplate="%{text:.1f} MB", textposition="outside")
            fig.update_layout(height=360, showlegend=False)
            fig.update_xaxes(**GRID)
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, width="stretch")

    st.subheader("Latency by input length (batch size 1)")
    by_tokens = pd.DataFrame(result["latency_by_input_tokens"])
    if not by_tokens.empty:
        fig = px.line(
            by_tokens, x="input_tokens", y="p50_latency_ms", color="quantisation",
            color_discrete_map=QUANT_COLOR, markers=True,
            labels={"input_tokens": "input tokens", "p50_latency_ms": "p50 latency (ms)"},
        )
        fig.update_layout(height=360)
        fig.update_xaxes(**GRID)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, width="stretch")

    agreement = result.get("agreement", {})
    st.subheader("Prediction agreement")
    if agreement.get("available"):
        for pair in agreement["pairs"]:
            st.markdown(
                f"**{pair['reference_model']}** vs **{pair['target_model']}** "
                f"on {pair['n_paired']:,} paired examples"
            )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Agreement", f"{pair['agreement_rate'] * 100:.3f}%")
            c2.metric("Disagreements", f"{pair['disagreements']:,}")
            c3.metric("FP32-only correct", f"{pair['reference_only_right']:,}")
            c4.metric(
                f"{result['target_quantisation']}-only correct",
                f"{pair['target_only_right']:,}",
            )
            fig = go.Figure(
                go.Bar(
                    x=[
                        pair["reference_only_right"],
                        pair["n_paired"] - pair["n_agree"],
                        pair["target_only_right"],
                    ],
                    y=["FP32 right", "different label", f"{result['target_quantisation']} right"],
                    orientation="h",
                    marker_color=["#4C8BF5", "#BBB", "#E4572E"],
                    text=[
                        pair["reference_only_right"],
                        pair["n_paired"] - pair["n_agree"],
                        pair["target_only_right"],
                    ],
                    textposition="outside",
                )
            )
            fig.update_layout(height=220, showlegend=False, margin=dict(l=10, r=40, t=10, b=40))
            fig.update_xaxes(**GRID)
            fig.update_yaxes(**GRID)
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(agreement.get("reason", "agreement not available"))

    if not served.empty:
        with st.expander("Real end-to-end serving latency (measured)"):
            st.caption(
                "From production telemetry. This is whole-application latency including "
                "retrieval and response assembly, so it is not comparable to the "
                "model-only benchmark above and is shown on its own scale."
            )
            st.dataframe(
                served[
                    ["quantisation", "n_requests", "p50_latency_ms", "p99_latency_ms", "mean_latency_ms"]
                ],
                width="stretch",
                hide_index=True,
            )


def page_drift(cfg: dict) -> None:
    st.title("Drift trend per feature")
    st.caption(
        "PSI, KS and JS per feature across monitoring windows. The trend label follows the "
        "endpoint change; the fitted slope is reported alongside it because a least-squares "
        "slope can be positive on a series that ends lower than it started."
    )

    result = drift()
    thresholds = result["thresholds"]

    features = pd.DataFrame(result["features"])
    if features.empty:
        st.warning("no drift measurements")
        return

    c1, c2 = st.columns([2, 1])
    selected = c1.selectbox("Feature", sorted(features["feature_id"].tolist()))
    origins = sorted(features["data_origin"].dropna().unique().tolist())
    origin_filter = c2.multiselect("Data origin", origins, default=origins)
    if origin_filter:
        features = features[features["data_origin"].isin(origin_filter)]

    series = pd.DataFrame(result["series"])
    if origin_filter:
        series = series[series["data_origin"].isin(origin_filter)]

    detail = features[features["feature_id"] == selected]
    if not detail.empty:
        row = detail.iloc[0]
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric(f"PSI first {ARROW} latest", f"{row['psi_first']:.4f} {ARROW} {row['psi_latest']:.4f}")
        k2.metric("Change", f"{row['psi_change']:+.4f}")
        k3.metric("Slope / window", f"{row['psi_slope_per_window']:+.5f}")
        k4.metric("Trend", str(row["trend"]))
        k5.metric("Severity", str(row["severity_latest"]))

    st.subheader(f"{selected} over time")
    sub = series[series["feature_id"] == selected].sort_values("measurement_ts")
    if not sub.empty:
        fig = go.Figure()
        for metric, label in (("psi", "PSI"), ("ks_statistic", "KS"), ("js_divergence", "JS")):
            if sub[metric].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=sub["measurement_ts"], y=sub[metric], mode="lines+markers",
                        name=label, marker=dict(size=5),
                        hovertemplate=f"{label} %{{y:.4f}}<br>%{{x}}<extra></extra>",
                    )
                )
        fig.add_hline(
            y=thresholds["moderate"], line_dash="dash", line_color="#E0A800",
            annotation_text="moderate",
        )
        fig.add_hline(
            y=thresholds["severe"], line_dash="dash", line_color="#E4572E",
            annotation_text="severe",
        )
        fig.update_layout(height=400, yaxis_title="statistic")
        fig.update_xaxes(**GRID)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, width="stretch")

    st.subheader(f"All features {DASH} latest PSI and slope")
    top = features.sort_values("psi_slope_per_window", ascending=False).head(20)
    fig = px.bar(
        top, x="psi_slope_per_window", y="feature_id", orientation="h",
        color="trend",
        color_discrete_map={"rising": "#E4572E", "falling": "#4C8BF5", "flat": "#B0B0B0"},
        labels={"psi_slope_per_window": "PSI slope per window", "feature_id": "feature"},
        custom_data=["psi_latest", "psi_first", "data_origin"],
    )
    fig.add_vline(x=0, line_color="#666")
    fig.update_traces(
        hovertemplate=(
            "%{y}<br>slope %{x:+.5f}/window"
            f"<br>PSI %{{customdata[1]:.4f}} {ARROW} %{{customdata[0]:.4f}}"
            "<br>%{customdata[2]}<extra></extra>"
        )
    )
    fig.update_layout(
        height=max(380, 22 * len(top)), showlegend=True,
        margin=dict(l=10, r=20, t=10, b=40),
    )
    fig.update_xaxes(**GRID)
    fig.update_yaxes(**GRID)
    st.plotly_chart(fig, width="stretch")

    alerts = pd.DataFrame(result["alerts"])
    if not alerts.empty:
        st.subheader("Alerts")
        st.dataframe(
            alerts[
                [
                    "feature_id", "psi_first", "psi_latest", "psi_change",
                    "psi_slope_per_window", "trend", "severity_latest", "data_origin",
                ]
            ],
            width="stretch",
            hide_index=True,
        )


def page_length(cfg: dict) -> None:
    st.title("Error rate by document-length bucket")
    st.caption(
        "Each bucket is also compared against the mean error rate of its two neighbours. "
        "A steady decline in accuracy as documents get shorter is ordinary; a bucket that "
        "is worse than the trend through it has a specific, fixable cause."
    )

    result = length_buckets(cfg["quantisation"])

    k1, k2 = st.columns(2)
    for col, arm in zip((k1, k2), result["arms"], strict=False):
        detail = result["arms"][arm]
        col.metric(f"{arm} worst bucket", detail["worst_bucket"])
        lift = detail.get("worst_bucket_lift")
        excess = detail.get("worst_bucket_excess_error")
        col.metric(
            "Error lift vs neighbours",
            f"{lift:.2f}{TIMES}" if lift else ELLIPSIS,
            delta=f"excess {excess:+.4f}" if excess is not None else None,
            delta_color="inverse",
        )

    rows = []
    for arm, detail in result["arms"].items():
        for bucket in detail["buckets"]:
            rows.append({"arm": arm, **_omit(bucket, ["arm", "model_id", "slice_id"])})
    frame = pd.DataFrame(rows)

    if not frame.empty:
        pivot = frame.pivot_table(
            index=["ordinal", "slice_name"], columns="arm", values="error_rate"
        ).reset_index().sort_values("ordinal")
        arm_cols = [c for c in pivot.columns if c in set(frame["arm"])]
        fig = px.line(
            pivot, x="slice_name", y=arm_cols, markers=True,
            color_discrete_map=ACCENT,
            labels={"slice_name": "document length bucket (tokens)", "value": "error rate"},
        )
        fig.update_layout(height=380)
        fig.update_xaxes(**GRID)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, width="stretch")

        fig2 = px.bar(
            frame, x="slice_name", y="excess_error_vs_neighbours", color="arm",
            color_discrete_map=ACCENT, barmode="group",
            labels={
                "slice_name": "document length bucket (tokens)",
                "excess_error_vs_neighbours": "excess error vs neighbouring buckets",
            },
            custom_data=["n_samples", "error_rate", "mean_confidence"],
        )
        fig2.add_hline(y=0, line_color="#666")
        fig2.update_traces(
            hovertemplate=(
                "%{x} (%{customdata[0]} examples)<br>error %{customdata[1]:.4f}"
                "<br>mean confidence %{customdata[2]:.4f}<br>excess %{y:+.4f}<extra></extra>"
            )
        )
        fig2.update_layout(height=380, barmode="group")
        fig2.update_xaxes(**GRID)
        fig2.update_yaxes(**GRID)
        st.plotly_chart(fig2, width="stretch")

    st.subheader(f"What goes wrong in the 17{EN_DASH}32 token bucket")
    conf = pd.DataFrame(confusions(cfg["arm_a"], "len_17_32"))
    if not conf.empty:
        conf = conf.copy()
        conf["share_of_bucket_errors"] = conf["share_of_bucket_errors"] * 100
        fig = px.bar(
            conf.head(10), x="n", y="true_label", orientation="h", color="predicted_label",
            labels={
                "n": "misclassified documents",
                "true_label": "true label",
                "predicted_label": "predicted as",
            },
            custom_data=["share_of_bucket_errors", "mean_confidence"],
        )
        fig.update_traces(
            hovertemplate=(
                "%{y} {ARROW} predicted label"
                "<br>%{customdata[0]:.1f}% of bucket errors"
                "<br>mean confidence %{customdata[1]:.4f}<extra></extra>"
            )
        )
        fig.update_layout(
            height=max(340, 34 * min(len(conf), 10)),
            yaxis_title=None, margin=dict(l=10, r=20, t=10, b=40),
        )
        fig.update_xaxes(**GRID)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "The concentration on a few label pairs is the finding: short documents in this "
            "band share a single cue term between paired labels, so the error is a content "
            "ambiguity rather than a length effect."
        )


def page_slices(cfg: dict) -> None:
    arm_a, arm_b = cfg["arm_a"], cfg["arm_b"]
    st.title(f"Slice analysis: where {arm_a} loses")
    st.caption(
        "Every class, length bucket and drift regime, with a paired bootstrap CI and an "
        "exact McNemar test. Slices with fewer than 30 paired examples are held back as "
        "unmeasured rather than reported as small wins or losses."
    )

    result = slice_losses(arm_a, arm_b, cfg["n_boot"])
    tally = result["summary"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Slices measured", tally["n_slices_measured"])
    k2.metric("Losses", tally["n_losses"])
    k3.metric("Significant losses", tally["significant_losses"])
    k4.metric("Wins", tally["n_wins"])

    for family, rows in result["by_family"].items():
        if not rows:
            continue
        st.subheader(f"{family.replace('_', ' ').title()} slices")
        frame = pd.DataFrame(rows)
        marks = (
            frame.get("significant", pd.Series([False] * len(frame)))
            .fillna(False)
            .astype(bool)
            .map({True: f"  {CHECK_MARK}", False: ""})
            .fillna("")
        )
        frame["label"] = frame["slice_name"].astype(str) + marks

        fig = px.bar(
            frame.sort_values("delta"),
            x="delta",
            y="label",
            orientation="h",
            color="delta",
            color_continuous_scale=["#E4572E", "#BDBDBD", "#4C8BF5"],
            color_continuous_midpoint=0,
            labels={"delta": f"accuracy delta ({arm_a} {MINUS} {arm_b})"},
            custom_data=[
                "n_examples", "accuracy_arm", "accuracy_against", "p_value", "significant",
            ],
        )
        for row in frame.itertuples():
            fig.add_shape(
                type="line",
                x0=row.ci_low, x1=row.ci_high, y0=row.label, y1=row.label,
                line=dict(color="#333", width=2),
            )
        fig.add_vline(x=0, line_color="#666")
        fig.update_traces(
            hovertemplate=(
                "%{y}<br>delta %{x:+.4f}<br>%{customdata[0]} examples"
                "<br>%{customdata[1]:.4f} vs %{customdata[2]:.4f}"
                "<br>p=%{customdata[3]:.4f} significant=%{customdata[4]}<extra></extra>"
            )
        )
        fig.update_layout(
            height=max(240, 34 * len(frame)), showlegend=False,
            coloraxis_showscale=False, margin=dict(l=10, r=20, t=10, b=40),
        )
        fig.update_xaxes(**GRID)
        fig.update_yaxes(**GRID)
        st.plotly_chart(fig, width="stretch")

        st.dataframe(
            frame[
                [
                    "slice_name", "n_examples", "accuracy_arm", "accuracy_against",
                    "delta", "ci_low", "ci_high", "p_value", "significant",
                ]
            ],
            width="stretch",
            hide_index=True,
        )

    if result["too_few_examples"]:
        with st.expander(f"{len(result['too_few_examples'])} slice(s) too small to judge"):
            st.dataframe(
                pd.DataFrame(result["too_few_examples"])[
                    [
                        "slice_type", "slice_name", "n_examples",
                        "accuracy_arm", "accuracy_against", "delta",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )


def _omit(record: dict, keys: list[str]) -> dict:
    return {k: v for k, v in record.items() if k not in keys}


def main() -> None:
    _require_warehouse()
    cfg = sidebar()

    pages = {
        "Overview": page_overview,
        "Arm comparison": page_arm_comparison,
        "Calibration": page_calibration,
        "Quantisation": page_quantisation,
        "Drift": page_drift,
        "Length buckets": page_length,
        "Slices": page_slices,
    }
    choice = st.sidebar.radio("View", list(pages))
    pages[choice](cfg)


if __name__ == "__main__":
    main()
