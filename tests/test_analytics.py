"""Statistics and the analytical queries."""

from __future__ import annotations

import numpy as np
import pytest

from eval_analytics.analytics import queries as Q
from eval_analytics.analytics.stats import (
    calibration,
    mcnemar_exact,
    paired_bootstrap,
)


class TestMcNemar:
    def test_no_discordant_pairs_is_not_significant(self):
        assert mcnemar_exact(0, 0) == 1.0

    def test_known_value(self):
        # Exact two-sided binomial p for 3 vs 12 discordant pairs:
        # 2 * P(X <= 3) with X ~ Binomial(15, 0.5) = 2 * 0.017578125.
        assert mcnemar_exact(3, 12) == pytest.approx(0.03515625, abs=1e-9)

    def test_symmetry(self):
        assert mcnemar_exact(4, 20) == pytest.approx(mcnemar_exact(20, 4))


class TestPairedBootstrap:
    def test_identical_inputs_give_a_zero_delta_and_a_degenerate_interval(self):
        a = np.array([1, 0, 1, 1, 0] * 40)
        result = paired_bootstrap(a, a, n_boot=500)
        assert result.delta == 0.0
        assert result.ci_low == 0.0
        assert result.ci_high == 0.0
        assert result.n_a_only == 0 and result.n_b_only == 0
        assert not result.significant

    def test_interval_brackets_the_observed_delta(self):
        rng = np.random.default_rng(7)
        a = (rng.random(800) < 0.80).astype(int)
        b = (rng.random(800) < 0.70).astype(int)
        result = paired_bootstrap(a, b, n_boot=2000)
        assert result.ci_low <= result.delta <= result.ci_high
        assert result.delta == pytest.approx(0.10, abs=0.04)
        assert result.significant

    def test_discordant_counts_are_exact(self):
        # pairs: (both), (a only), (b only), (neither), (a only), (neither)
        a = np.array([1, 1, 0, 0, 1, 0])
        b = np.array([1, 0, 1, 0, 0, 0])
        result = paired_bootstrap(a, b, n_boot=200)
        assert result.n_both == 1
        assert result.n_neither == 2
        assert result.n_a_only == 2
        assert result.n_b_only == 1
        assert result.n == 6

    def test_mismatched_lengths_are_refused(self):
        with pytest.raises(ValueError, match="must align"):
            paired_bootstrap([1, 0, 1], [1, 0], n_boot=100)

    def test_empty_input_is_refused(self):
        with pytest.raises(ValueError, match="at least one example"):
            paired_bootstrap([], [], n_boot=100)

    def test_is_seed_reproducible(self):
        rng = np.random.default_rng(11)
        a = (rng.random(400) < 0.8).astype(int)
        b = (rng.random(400) < 0.75).astype(int)
        first = paired_bootstrap(a, b, n_boot=1000, seed=99)
        second = paired_bootstrap(a, b, n_boot=1000, seed=99)
        assert first.ci_low == second.ci_low
        assert first.ci_high == second.ci_high


class TestCalibration:
    def test_a_perfectly_calibrated_set_scores_zero_ece(self):
        # A bin is calibrated when its observed accuracy equals its mean
        # confidence -- not when every example in it is correct. So the 0.9 bin
        # holds 900 examples of which 810 are right, and the 0.1 bin holds 100
        # of which 10 are right.
        conf = np.array([0.9] * 900 + [0.1] * 100)
        correct = np.array([1] * 810 + [0] * 90 + [1] * 10 + [0] * 90)
        summary = calibration("m", "lora", "fp32", conf, correct, correct.astype(float))
        assert summary.ece == pytest.approx(0.0, abs=1e-9)
        assert summary.mce == pytest.approx(0.0, abs=1e-9)
        assert summary.accuracy == pytest.approx(0.82)
        assert summary.overconfidence == pytest.approx(0.0, abs=1e-9)

    def test_contributions_sum_to_ece(self):
        rng = np.random.default_rng(3)
        conf = rng.random(500)
        correct = (rng.random(500) < conf).astype(int)
        summary = calibration("m", "lora", "fp32", conf, correct, None)
        total = sum(b.ece_contribution for b in summary.buckets)
        assert total == pytest.approx(summary.ece)

    def test_overconfidence_is_mean_confidence_minus_accuracy(self):
        conf = np.full(100, 0.9)
        correct = np.array([1] * 60 + [0] * 40)
        summary = calibration("m", "lora", "fp32", conf, correct, None)
        assert summary.overconfidence == pytest.approx(0.9 - 0.6)


class TestArmComparison:
    def test_arms_are_paired_on_the_same_documents(self, db_path: str):
        result = Q.lora_vs_full_finetune(db_path, n_boot=500)
        assert result["n_examples"] == 2400
        assert set(result["overall"]) >= {"delta", "ci_low", "ci_high", "p_value"}

    def test_per_class_covers_every_class_with_a_label(self, db_path: str):
        result = Q.lora_vs_full_finetune(db_path, n_boot=200)
        labels = {row["true_label"] for row in result["per_class"]}
        assert len(labels) == 8
        assert all("ci_low" in row and "p_value" in row for row in result["per_class"])

    def test_comparison_refuses_an_arm_with_no_labelled_predictions(self, db_path: str):
        """An arm absent from the warehouse must fail loudly, not return zeros."""
        with pytest.raises(ValueError, match="no labelled predictions"):
            Q.lora_vs_full_finetune(
                db_path, arm_a="lora", arm_b="no_such_arm", quantisation="fp32"
            )

    def test_pairing_guard_rejects_different_document_sets(self):
        """The guard itself: arms scored on different documents must not pair."""
        import pandas as pd

        from eval_analytics.analytics.queries import _assert_paired

        a = pd.DataFrame({"example_id": ["x1", "x2", "x3"]})
        b = pd.DataFrame({"example_id": ["x1", "x2", "x9"]})
        with pytest.raises(ValueError, match="not scored on the same documents"):
            _assert_paired(a, b, "lora", "full_finetune")

    def test_pairing_guard_accepts_an_identical_document_set(self):
        import pandas as pd

        from eval_analytics.analytics.queries import _assert_paired

        frame = pd.DataFrame({"example_id": ["x1", "x2"]})
        _assert_paired(frame, frame.copy(), "lora", "full_finetune")


class TestCalibrationQuery:
    def test_recomputed_ece_matches_the_mart(self, db_path: str):
        result = Q.calibration_by_arm(db_path)
        for entry in result["arms"]:
            assert entry["ece_matches_mart"], entry
            assert entry["ece"] == pytest.approx(entry["stored_ece"], abs=1e-6)

    def test_every_bucket_contribution_is_a_share_of_total(self, db_path: str):
        result = Q.calibration_by_arm(db_path)
        for entry in result["arms"]:
            shares = sum(b["share_of_ece"] for b in entry["buckets"])
            assert shares == pytest.approx(1.0, abs=1e-6)


class TestQuantisationQuery:
    def test_int8_is_cheaper_and_smaller_than_fp32(self, db_path: str):
        result = Q.quantisation_comparison(db_path)
        ratios = result["ratios"]
        assert ratios["p50_speedup"] > 1.0
        assert ratios["size_reduction"] > 1.0

    def test_agreement_pairs_two_precisions_of_one_model(self, db_path: str):
        result = Q.quantisation_comparison(db_path)
        agreement = result["agreement"]
        assert agreement["available"]
        pair = agreement["summary"]
        # Same base model, arm and adapter rank: two builds of one model.
        assert pair["reference_model"] != pair["target_model"]
        assert pair["agreement_rate"] > 0.98
        assert pair["n_paired"] == 2400

    def test_agreement_is_not_comparing_unrelated_models(self, db_path: str):
        """A full-finetune FP32 model must never be treated as INT8 LoRA."""
        result = Q.quantisation_comparison(
            db_path, reference_quantisation="fp32", target_quantisation="int8"
        )
        for pair in result["agreement"]["pairs"]:
            assert "full-finetune" not in pair["reference_model"]


class TestDriftQuery:
    def test_every_feature_reports_a_slope_and_a_trend(self, db_path: str):
        result = Q.drift_trend(db_path)
        assert result["features"]
        for feature in result["features"]:
            assert "psi_slope_per_window" in feature
            assert feature["trend"] in {"rising", "falling", "flat"}

    def test_pivot_exposes_all_three_statistics(self, db_path: str):
        result = Q.drift_trend(db_path)
        synthetic = [
            f for f in result["features"] if f["data_origin"] == "synthetic"
        ]
        assert synthetic
        for feature in synthetic:
            assert feature["psi_latest"] is not None
            assert feature["ks_latest"] is not None
            assert feature["js_latest"] is not None

    def test_can_filter_to_one_feature(self, db_path: str):
        result = Q.drift_trend(db_path, feature="upper_ratio")
        assert len(result["features"]) == 1
        assert result["features"][0]["feature_id"] == "upper_ratio"


class TestLengthBucketQuery:
    def test_every_arm_reports_ordered_buckets(self, db_path: str):
        result = Q.error_by_length_bucket(db_path)
        assert result["arms"]
        for detail in result["arms"].values():
            ordinals = [b["ordinal"] for b in detail["buckets"]]
            assert ordinals == sorted(ordinals)

    def test_buckets_carry_neighbour_comparisons(self, db_path: str):
        result = Q.error_by_length_bucket(db_path)
        for detail in result["arms"].values():
            interior = [
                b for b in detail["buckets"] if b["excess_error_vs_neighbours"] is not None
            ]
            assert interior
            assert any(b["lift"] for b in interior)

    def test_confusions_are_ordered_by_volume(self, db_path: str):
        rows = Q.confusion_by_length(db_path, arm="lora", slice_id="len_17_32")
        assert rows
        counts = [r["n"] for r in rows]
        assert counts == sorted(counts, reverse=True)
        assert all(r["true_label"] != r["predicted_label"] for r in rows)


class TestSliceLosses:
    def test_slices_are_split_into_wins_and_losses(self, db_path: str):
        result = Q.slice_losses(db_path, n_boot=300)
        tally = result["summary"]
        assert tally["n_slices_measured"] > 0
        assert tally["n_losses"] + tally["n_wins"] == tally["n_slices_measured"]
        assert all(row["delta"] < 0 for row in result["losses"])
        assert all(row["delta"] > 0 for row in result["wins"])

    def test_measured_slices_carry_inference(self, db_path: str):
        result = Q.slice_losses(db_path, n_boot=300)
        for row in result["losses"] + result["wins"]:
            assert row["status"] == "measured"
            assert row["ci_low"] <= row["ci_high"]
            assert 0.0 <= row["p_value"] <= 1.0
            assert "significant" in row

    def test_families_are_present(self, db_path: str):
        result = Q.slice_losses(db_path, n_boot=300)
        # The projection view also emits the 'overall' slice, which is a
        # legitimate slice type rather than a family of its own.
        assert {"class", "length_bucket", "drift_window"} <= set(result["by_family"])


class TestDataQualitySummary:
    def test_counts_and_provenance_are_reported(self, db_path: str):
        summary = Q.data_quality_summary(db_path)
        names = {row["object"] for row in summary["counts"]}
        assert {"fact_prediction", "fact_evaluation_run"} <= names
        assert summary["latest_ingestion_run"]["status"] == "success"
